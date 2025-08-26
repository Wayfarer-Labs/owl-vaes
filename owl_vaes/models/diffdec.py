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
        n_tokens = size[0] // config.patch_size * size[1] // config.patch_size

        self.proj_in = nn.Linear(config.patch_size * config.patch_size * config.channels, config.d_model, bias = False)

        self.proj_out = nn.Linear(config.d_model, config.patch_size * config.patch_size * config.channels, bias = False)

        self.ts_embed = TimestepEmbedding(config.d_model)
        
        self.proj_in_z = nn.Linear(config.latent_channels, config.d_model)

        self.rope_impl = getattr(config, "rope_impl", None)
        if self.rope_impl is None:
            self.pos_enc_x = LearnedPosEnc(n_tokens, config.d_model)
            self.pos_enc_z = LearnedPosEnc(config.latent_size**2, config.d_model)
        
        self.final = FinalLayer(config, skip_proj = True)

        self.p = config.patch_size
        self.n_p_y = config.sample_size[0] // self.p
        self.n_p_x = config.sample_size[1] // self.p

        if config.backbone == "dit":
            self.blocks = DiT(config)
        elif config.backbone == "hdit":
            from ..nn.hdit import HDiT
            self.blocks = HDiT(config)
        self.config = config

    def forward(self, x, z, ts):
        # x is [b,c,h,w]
        # z is [b,c,h,w] but different size cause latent
        # ts is [b,] in [0,1]
        # d is [b,] in [1,2,4,...,128]

        cond = self.ts_embed(ts)

        # Convert from image format [b,c,h,w] to patches [b,n_patches,patch_size*patch_size*c]
        b, c, h, w = x.shape
        x = x.view(b, c, self.n_p_y, self.p, self.n_p_x, self.p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, self.n_p_y * self.n_p_x, self.p * self.p * c)
        x = self.proj_in(x) # -> [b,n,d]
        if self.rope_impl is None:
            x = self.pos_enc_x(x)

        # Flatten spatial dimensions: [b,c,h,w] -> [b,h*w,c]
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        z = self.proj_in_z(z)
        if self.rope_impl is None:
            z = self.pos_enc_z(z)
        
        n = x.shape[1]
        x = torch.cat([x,z],dim=1)

        x = self.blocks(x, cond)
        x = x[:,:n]

        x = self.final(x, cond)
        x = self.proj_out(x)
        # Convert from patches back to image format [b,n_patches,patch_size*patch_size*c] -> [b,c,h,w]
        b, n_patches, patch_dim = x.shape
        c = patch_dim // (self.p * self.p)
        x = x.view(b, self.n_p_y, self.n_p_x, self.p, self.p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(b, c, self.n_p_y * self.p, self.n_p_x * self.p).contiguous()

        return x

class DiffusionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = DiffusionDecoderCore(config)

        self.ts_mu = 0.4
        self.ts_sigma = 1.0

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

        pred = self.core(lerpd, z, ts)
        diff_loss = F.mse_loss(pred, target)

        return diff_loss


if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/diffdec.yml").model
    model = DiffusionDecoderCore(cfg).bfloat16().cuda()
    z = torch.randn(1,128,4,4).bfloat16().cuda()
    with torch.no_grad():
        y = model.sample(z)
        print(y.shape)
    print(cfg)
