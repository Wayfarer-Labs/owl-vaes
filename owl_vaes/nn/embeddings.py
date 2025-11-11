import torch
from torch import nn
import math
from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)
import einops as eo

from .mlp import MLPSimple

class ImageRoPE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        pass

    def forward(self, q, k):
        # q k both [b,h,n,d]
        pass

class LearnedPosEnc(nn.Module):
    def __init__(self, n_seq, dim):
        super().__init__()

        self.p = nn.Parameter(torch.randn(n_seq,dim)*0.02)

    def forward(self, x):
        b,n,d = x.shape
        p = self.p.unsqueeze(0).expand(b, -1, -1)
        return x + p

class RoPEEmbedding(nn.Module):
    """
    Video RoPE embedding for when latents are 3D [n,h,w]
    """
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        dim_head = config.d_model // config.n_heads
        self.pos_emb = RotaryEmbedding(
            dim = dim_head//2
        )

    def forward(self, q, k):
        # q k both [b,h,n,d]
        # k can be longer for caching
        q,k = self.pos_emb.rotate_queries_with_cached_keys(q,k)
        return q,k
        
        return q, k
class SinCosEmbedding(nn.Module):
    def __init__(self, d, theta=300, mult=1000):
        super().__init__()
        self.d = d # Assume this is even
        self.theta = theta
        self.mult = mult

    def forward(self, t):
        # Assume t is already a tensor for compilation
        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = t * self.mult
        half = self.d // 2

        inds = torch.arange(half, device=t.device, dtype=t.dtype)
        freqs = (
            -math.log(self.theta) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim=-1)
        return embs

class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512, mult = 1000):
        super().__init__()

        self.mlp = MLPSimple(d_in, dim_out=d_out)
        self.sincos = SinCosEmbedding(d_in, theta=300, mult=mult)

    def forward(self, t):
        # Assume t is already a tensor for compilation
        if t.ndim == 0:
            t = t.unsqueeze(0)
        # Remove assert for compilation compatibility

        if t.ndim > 1: # [b,n]
            b,n = t.shape
            t_flat = eo.rearrange(t, 'b n -> (b n)')
            embs = self.sincos(t_flat)
            embs = eo.rearrange(embs, '(b n) d -> b n d', b = b, n = n)
        else:
            embs = self.sincos(t)
        return self.mlp(embs)

class StepEmbedding(nn.Module):
    def __init__(self, d_out, d_in=512, max_steps=128):
        super().__init__()

        self.mlp = MLPSimple(d_in, dim_out=d_out)
        self.max_steps = max_steps
        mult = 1000 / (math.log2(max_steps) - 1)  # Adjusted to map [1,128] -> [0,1000]
        self.sincos = SinCosEmbedding(d_in, theta=300, mult=mult)

    def forward(self, steps):
        # Assume steps is already a tensor for compilation
        if steps.ndim == 0:
            steps = steps.unsqueeze(0)

        # Map steps to [1, log2(max_steps)-1] to make 1 and 128 map to same value
        t = (math.log2(self.max_steps) - 1 - torch.log2(steps.float())).to(steps.dtype)
        # Wrap around so 0 maps to same as max
        t = t % (math.log2(self.max_steps) - 1)
        embs = self.sincos(t)
        return self.mlp(embs)
