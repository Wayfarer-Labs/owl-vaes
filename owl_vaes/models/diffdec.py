import torch
from torch import nn
import torch.nn.functional as F
import math

from copy import deepcopy

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
        self.final = FinalLayer(config, skip_proj = True)

        self.blocks = DiT(config)
        self.config = config

        self.shuffle_factor = getattr(config, "shuffle_factor", 1)
        self.p_y, self.p_x = patch_size
        self.n_p_y = config.sample_size[0] // self.p_y
        self.n_p_x = config.sample_size[1] // self.p_x

        # Learned null embedding for CFG
        self.null_emb = nn.Parameter(torch.zeros(config.latent_channels, config.latent_size, config.latent_size))

    def forward(self, x, z, ts):
        # x is [b,c,h,w]
        # z is [b,c,h,w] but different size cause latent
        # ts is [b,] in [0,1]
        # d is [b,] in [1,2,4,...,128]

        if self.shuffle_factor > 1:
            x = F.pixel_unshuffle(x, self.shuffle_factor)

        cond = self.ts_embed(ts)

        # Convert from image format [b,c,h,w] to patches [b,n_patches,patch_size*patch_size*c]
        b, c, h, w = x.shape

        x = x.view(b, c, self.n_p_y, self.p_y, self.n_p_x, self.p_x)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, self.n_p_y * self.n_p_x, self.p_y * self.p_x * c)

        x = self.proj_in(x) # -> [b,n,d]
        # No learned positional encoding

        # Flatten spatial dimensions: [b,c,h,w] -> [b,h*w,c]
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        z = self.proj_in_z(z)
        # No learned positional encoding
        
        n = x.shape[1]
        x = torch.cat([x,z],dim=1)

        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.final(x, cond)
        x = self.proj_out(x)
        # Convert from patches back to image format [b,n_patches,patch_size*patch_size*c] -> [b,c,h,w]
        b, n_patches, patch_dim = x.shape
        c = patch_dim // (self.p_y * self.p_x)
        x = x.view(b, self.n_p_y, self.n_p_x, self.p_y, self.p_x, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, self.n_p_y * self.p_y, self.n_p_x * self.p_x).contiguous()

        if self.shuffle_factor > 1:
            x = F.pixel_shuffle(x, self.shuffle_factor)

        return x

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)

        self.ts_mu = 0.4
        self.ts_sigma = 1.0
        self.cfg_prob = getattr(config, "cfg_prob", 0.0)

    @torch.no_grad()
    def sample_timesteps(self, b, device, dtype):
        # Sample ts in [0, 1] from a normal distribution, then sigmoid
        ts = torch.randn(b, device=device, dtype=dtype)
        ts = ts * self.ts_sigma - self.ts_mu
        ts = ts.sigmoid()
        return ts

    def forward(self, x, z):
        with torch.no_grad():
            ts = self.sample_timesteps(len(x), x.device, x.dtype)

            eps = torch.randn_like(x)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            lerpd = x * (1. - ts_exp) + ts_exp * eps
            target = eps - x

            # Apply CFG dropout: replace z with null embedding with probability cfg_prob
            if self.cfg_prob > 0:
                mask = torch.rand(len(z), device=z.device) < self.cfg_prob
                z = z.clone()
                z[mask] = self.core.null_emb.unsqueeze(0)

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
