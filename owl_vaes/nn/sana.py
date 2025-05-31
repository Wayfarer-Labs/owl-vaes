import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..nn.normalization import LayerNorm
from ..configs import TransformerConfig

from .attn import Attn

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
        x = eo.reduce(x, 'b (reps c) h w -> b c h w', reps = self.reps, reduction = 'mean')
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
        x = eo.repeat(x, 'b c h w -> b (reps c) h w', reps=self.reps)
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
