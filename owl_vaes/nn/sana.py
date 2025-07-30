import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from .attn import Attn
from ..configs import TransformerConfig
from .normalization import LayerNorm

"""
Building blocks for SANA modules and residuals
"""

class SpaceToChannel(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        if ch_in == ch_out:
            self.reps = 4
        else:
            self.reps = 2

    def forward(self, x):
        # [c,2h,2w] -> [4c,h,w]
        x = F.pixel_unshuffle(x, downscale_factor=2)
        # [4c,h,w] -> [2c,h,w]
        b, c, h, w = x.shape
        x = x.view(b, self.reps, c//self.reps, h, w)
        x = x.mean(dim=1)
        return x

class ChannelToSpace(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        if ch_in == ch_out:
            self.reps = 4
        else:
            self.reps = 2

    def forward(self, x):
        # [4c, h, w] -> [c, 2h, 2w]
        x = F.pixel_shuffle(x, upscale_factor=2)
        # [c, 2h, 2w] -> [2c, 2h, 2w]
        b, c, h, w = x.shape
        x = x.repeat(1, self.reps, 1, 1)
        return x

class ResidualAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()

        head_dim = 16
        n_heads = ch // head_dim
        self.n_heads = n_heads

        attn_cfg = TransformerConfig(
            n_heads = n_heads,
            d_model = ch
        )
        self.norm = LayerNorm(ch)
        self.attn = Attn(attn_cfg)
        self.layerscale = nn.Parameter(torch.ones(1)*1.0e-6)

    def forward(self, x):
        res = x.clone()

        x = eo.rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = self.attn(x)
        x = res + self.layerscale * x
        return x

class SanaAttn(nn.Module):
    def __init__(self, ch):
        super().__init__()

        config = TransformerConfig(
            d_model = ch,
            n_heads = ch // 64
        )

        self.norm1 = LayerNorm(ch)
        self.norm2 = LayerNorm(ch)

        self.attn = Attn(config)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch * 4),
            nn.SiLU(),
            nn.Linear(ch * 4, ch)
        )

    def forward(self, x):

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [b, c, h, w] -> [b, hw, c]

        res = x.clone()

        x = self.norm1(x)
        x = self.attn(x)
        x = res + x

        res2 = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)

        x = x + res2

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [b, hw, c] -> [b, c, h, w]
        return x