import torch
from torch import nn
import torch.nn.functional as F
import math

from copy import deepcopy
import einops as eo

from ..utils import freeze
from ..configs import TransformerConfig
from ..nn.resnet import SquareToLandscape, LandscapeToSquare

from ..nn.dit import DiT, FinalLayer

from ..nn.embeddings import LearnedPosEnc
from ..nn.embeddings import TimestepEmbedding, StepEmbedding

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

def find_nearest_square(size):
    h,w = size
    avg = (h + w) / 2
    return 2 ** int(round(torch.log2(torch.tensor(avg)).item()))

def vid_pixel_shuffle(x, shuffle_factor):
    b,n = x.shape[:2]
    x = eo.rearrange(x, 'b n ... -> (b n) ...')
    x = F.pixel_shuffle(x, shuffle_factor)
    x = eo.rearrange(x, '(b n) ... -> b n ...', n = n)
    return x

def vid_pixel_unshuffle(x, shuffle_factor):
    b,n = x.shape[:2]
    x = eo.rearrange(x, 'b n ... -> (b n) ...')
    x = F.pixel_unshuffle(x, shuffle_factor)
    x = eo.rearrange(x, '(b n) ... -> b n ...', n = n)
    return x

class DiffusionDecoderCore(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        size = config.sample_size
        patch_size = config.patch_size
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        n_tokens = size[0] // patch_size[0] * size[1] // patch_size[1]

        self.proj_in = nn.Linear(patch_size[0] * patch_size[1] * config.channels, config.d_model, bias = False)

        self.proj_out = nn.Linear(config.d_model, patch_size[0] * patch_size[1] * config.channels, bias = False)

        self.ts_embed = TimestepEmbedding(config.d_model)
        
        self.proj_in_z = nn.Linear(config.latent_channels, config.d_model)

        self.rope_impl = getattr(config, "rope_impl", None)
        if self.rope_impl is None:
            raise ValueError("rope_impl must be set; learned positional encodings are no longer supported.")

        self.final = FinalLayer(config, skip_proj = True)

        config.causal = True
        self.blocks = DiT(config)
        self.config = config

        self.shuffle_factor = getattr(config, "shuffle_factor", 1)
        self.p_y, self.p_x = patch_size
        self.n_p_y = config.sample_size[0] // self.p_y
        self.n_p_x = config.sample_size[1] // self.p_x
        self.n_frames = config.n_frames

        # Learned null embedding for CFG
        self.null_emb = nn.Parameter(torch.zeros(config.latent_channels, config.latent_size, config.latent_size))

    def forward(self, x, z, ts):
        # x is [b,n,c,h,w]
        # z is [b,n,c,h,w] but different size cause latent
        # ts is [b,n] in [0,1] - per-frame timesteps

        if self.shuffle_factor > 1:
            x = vid_pixel_unshuffle(x, self.shuffle_factor)

        cond = self.ts_embed(ts)

        x = eo.rearrange(
            x,
            'b t c (n_p_y p_y) (n_p_x p_x) -> b (t n_p_y n_p_x) (p_y p_x c)',
            n_p_y = self.n_p_y,
            n_p_x = self.n_p_x
        )

        z = eo.rearrange(
            z,
            'b t c h w -> b (t h w) c'
        )

        x = self.proj_in(x) # -> [b,n,d]
        z = self.proj_in_z(z)
        
        n = x.shape[1]
        x = torch.cat([x,z],dim=1)

        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.final(x, cond)
        x = self.proj_out(x)

        x = eo.rearrange(
            x,
            'b (t n_p_y n_p_x) (p_y p_x c) -> b t c (n_p_y p_y) (n_p_x p_x)',
            n_p_y = self.n_p_y,
            n_p_x = self.n_p_x,
            p_y = self.p_y,
            p_x = self.p_x
        )

        if self.shuffle_factor > 1:
            x = vid_pixel_shuffle(x, self.shuffle_factor)

        return x

class CausalDiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)

        self.ts_mu = 0.0
        self.ts_sigma = 1.0
        self.cfg_prob = getattr(config, "cfg_prob", 0.0)
        self.prev_noise = getattr(config, "prev_noise", 0.0)

    @torch.no_grad()
    def sample_timesteps(self, b, n, device, dtype):
        # Sample ts in [0, 1] from a normal distribution, then sigmoid
        # Per-frame timesteps: [B, N]
        ts = torch.randn(b, n, device=device, dtype=dtype)
        ts = ts * self.ts_sigma - self.ts_mu
        ts = ts.sigmoid()
        return ts

    def forward(self, x, z):
        # x is [b,n,c,h,w]
        # z is [b,n,c,h,w] with latent size
        b, n = x.shape[:2]

        with torch.no_grad():
            # Sample per-frame timesteps [B,N]
            ts = self.sample_timesteps(b, n, x.device, x.dtype)

            eps = torch.randn_like(x)
            # Expand timesteps to match x shape [B,N,C,H,W]
            ts_exp = ts.view(b, n, 1, 1, 1)

            lerpd = x * (1. - ts_exp) + ts_exp * eps
            target = eps - x

            # CFG per-frame: randomly dropout individual frames
            if self.cfg_prob > 0:
                mask = torch.rand(b, n, device=z.device) < self.cfg_prob  # [B,N]
                z = z.clone()
                # Expand mask to match z dimensions [B,N,C,H,W]
                mask_exp = mask.view(b, n, 1, 1, 1)
                # Expand null_emb to broadcast [1,1,C,H,W]
                null_exp = self.core.null_emb.view(1, 1, *self.core.null_emb.shape)
                z = torch.where(mask_exp, null_exp, z)

        pred = self.core(lerpd, z, ts)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss

if __name__ == "__main__":
    from ..configs import Config
    import torch
    import torch.nn.functional as F

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cfg = Config.from_yaml("configs/waypoint_1/wp1_diffdec.yml").model

    from diffusers import AutoencoderTiny
    vae = AutoencoderTiny.from_pretrained("madebyollin/taef1")
    vae = vae.float().to(device)  # bfloat16 not supported on mps

    model = DiffusionDecoderCore(cfg).float().to(device)
    x = torch.randn(1,3,720,1280).float().to(device)
    z = torch.randn(1,64,16,16).float().to(device)

    with torch.no_grad():
        proxy = vae.encoder(x)
        y = model(proxy, z, torch.tensor([0.5]).to(device).float())
        rec = vae.decoder(y)
        print(rec.shape)
