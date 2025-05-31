import torch
from torch import nn

import einops as eo

from ..configs import TransformerConfig

class ImageRoPE(nn.Module):
    def __init__(self, config : TransformerConfig):
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
        p = eo.repeat(self.p, 'n d -> b n d', b = b)
        return x + p
