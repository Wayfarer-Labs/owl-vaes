import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import einops as eo

from .resnet import WeightNormConv2d
from .normalization import LayerNorm

"""
Video VAE building blocks
"""

class TemporalDownsample(nn.Module):
    def __init__(self, ch):
        self.proj = WeightNormConv2d(2*ch, ch, 3, 1, 1)

    def forward(self, x):
        # x is [b,t,c,h,w], we output [b,t//2,c,h,w]
        b = x.shape[0]
        x = eo.rearrange(x, 'b (t two) c h w -> (b t) (two c) h w', two = 2)
        x = self.proj(x)
        x = eo.rearrange(x, '(b t) c h w -> b t c h w', b = b)
        return x

class TemporalUpsample(nn.Module):
    def __init__(self, ch):
        self.proj = WeightNormConv2d(ch, 2*ch, 3, 1, 1)

    def forward(self, x):
        # x is [b,t,c,h,w], we output [b,t*2,c,h,w]
        b = x.shape[0]
        x = eo.rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.proj(x)
        x = eo.rearrange(x, '(b t) (two c) h w -> b (t two) c h w', b = b, two = 2)
        return x

"""
Smart patching doesn't require patch size
Figures out a patch size to fit some desired token count
Assumes the incoming videos are square
Token count is number of patches per side
"""

class BatchedWeightNormConv2D(WeightNormConv2d):
    """
    Handles reshape in/out for temporal dim
    """
    def forward(self, x):
        b,t = x.shape[:2]
        x = eo.rearrange(x, 'b t c h w -> (b t) c h w')
        x = super().forward(x)
        x = eo.rearrange(x, '(b t) c h w -> b t c h w', b = b)
        return x

class SmartPatchIn(nn.Module):
    def __init__(self, ch, d_model, desired_token_count = 16):
        super().__init__()

        self.n_p = desired_token_count
        self.proj = BatchedWeightNormConv2D(ch, d_model, 3, 1, 1)
    
    def forward(self, x):
        # x is [b,t,c,h,w]
        # pool to downsample to desired_token_count
        x = self.proj(x)
        x = F.interpolate(x, size=(self.n_p, self.n_p))
        return eo.rearrange(
            x, 
            'b t c h w -> b (t h w) c',
        )
        return x


class SmartPatchOut(nn.Module):
    def __init__(self, ch, d_model):
        super().__init__()

        self.proj = BatchedWeightNormConv2D(d_model, ch, 3, 1, 1)
    
    def forward(self, x, h, w):
        # x is [b,t,c,h,w]
        x = eo.rearrange(
            x,
            'b (t h w) c -> b t c h w',
        )
        x = F.interpolate(x, size = (h, w))
        x = self.proj(x)
        return x

def get_frame_causal_attn_mask(
    config,
    cache_frame_offset = 0,
    batch_size = None,
    device = None,
    max_q_len = 1024,
    max_kv_len = 1024,
    kernel_size = None,
):
    if kernel_size is not None:
        k_y, k_x = int_to_tuple(kernel_size)

    def frame_idx(idx):
        return idx // (config.tokens_per_frame)
    
    def row_idx(idx):
        idx = idx % (config.tokens_per_frame)
        idx = idx // int(config.tokens_per_frame ** .5)
        return idx
    
    def col_idx(idx):
        idx = idx % (config.tokens_per_frame)
        idx = idx % int(config.tokens_per_frame ** .5)
        return idx

    def can_attend_to(b,h,idx_i, idx_j):
        frame_idx_i = frame_idx(idx_i)
        row_idx_i = row_idx(idx_i)
        col_idx_i = col_idx(idx_i)

        frame_idx_j = frame_idx(idx_j)
        row_idx_j = row_idx(idx_j)
        col_idx_j = col_idx(idx_j)
        if kernel_size is None:
            return frame_idx_j <= frame_idx_i
        else:
            causal_mask = (frame_idx_j <= frame_idx_i)
            image_nbr_mask = (torch.abs(row_idx_j - row_idx_i) <= k_y) & (torch.abs(col_idx_j - col_idx_i) <= k_x)
            return causal_mask & image_nbr_mask

class CausalFrameAttn(nn.Module):
    """
    Attend to past frames
    """
    def __init__(self, ch, config):
        super().__init__()

        self.norm = LayerNorm(config.d_model)
        self.proj_in = SmartPatchIn(ch)
        self.proj_out = SmartPatchOut(ch)
        self.attn = Attn(config)
    
    def forward(self, x, kv_cache = None, attn_mask = None):
        # x is [b,t,c,h,w]
        _, _, _, h, w = x.shape
        res = x.clone()
        x = self.proj_in(x)
        x = self.norm(x)
        # x is now [b,n,d]
        x = self.attn(x, kv_cache, attn_mask)
        x = self.proj_out(x, h, w)
        x = x + res
        return x
        
