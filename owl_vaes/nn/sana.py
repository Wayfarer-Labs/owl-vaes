import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import einops as eo

from .attn import Attn
from ..configs import TransformerConfig
from .normalization import LayerNorm
from .resnet import WeightNormConv2d

"""
Building blocks for SANA modules and residuals
"""

class SpaceToChannel(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out // 4, 3, 1, 1)

    def forward(self, x):
        x = self.proj(x.contiguous()) # [c_in, h, w] -> [c_out // 4, h, w]
        x = F.pixel_unshuffle(x, 2).contiguous() # [c_out // 4, h, w] -> [c_out, h // 2, w // 2]
        return x

class ChannelToSpace(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, 4 * ch_out, 3, 1, 1)

    def forward(self, x):
        x = self.proj(x.contiguous()) # [c_in, h, w] -> [4 * c_out, h, w]
        x = F.pixel_shuffle(x, 2).contiguous() # [4 * c_out, h, w] -> [c_out, 2h, 2w]
        return x

class ChannelAverage(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.grps = ch_in // ch_out
        self.scale = (self.grps) ** 0.5
    
    def forward(self, x):
        res = x.clone()
        x = self.proj(x.contiguous()) # [b, ch_out, h, w]

        # Residual goes through channel avg
        res = res.view(res.shape[0], self.grps, res.shape[1] // self.grps, res.shape[2], res.shape[3]).contiguous()
        res = res.mean(dim=1) * self.scale # [b, ch_out, h, w]
        
        return res + x

class ChannelDuplication(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.reps = ch_out // ch_in
        self.scale = (self.reps) ** -0.5

    def forward(self, x):
        res = x.clone()
        x = self.proj(x.contiguous())

        res = res.repeat_interleave(self.reps, dim = 1).contiguous() * self.scale

        return res + x

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