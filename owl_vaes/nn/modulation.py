import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm

class AdaLN(nn.Module):
    def __init__(self, dim_in, dim_out = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        self.fc = nn.Linear(dim_in, 2 * dim_out)
        self.norm = LayerNorm(dim_out)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        b,n,d = x.shape

        y = F.silu(cond)
        ab = self.fc(y) # [b,2d]
        ab = ab[:,None,:].expand(b,n,-1)
        a,b = ab.chunk(2,dim=-1) # each [b,n,d]

        x = self.norm(x) * (1. + a) + b
        return x

class Gate(nn.Module):
    def __init__(self, dim_in, dim_out = None):
        if dim_out is None:
            dim_out = dim_in

        super().__init__()

        self.fc_c = nn.Linear(dim_in, dim_out)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        b,n,d = x.shape

        y = F.silu(cond)
        c = self.fc_c(y) # [b,d]
        c = c[:,None,:].expand(b,n,-1) # [b,n,d]

        return c * x

class FiLM(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.alpha_beta = nn.Linear(dim_in, 2 * dim_out, bias = False)
        self.alpha_beta.weight.data.zero_()


    def forward(self, x, cond):
        # x is [bn,c,h,w]
        # cond is [b,n,d]
        bn,c,h,w = x.shape
        bs = cond.shape[0]

        y = F.silu(cond)
        ab = self.alpha_beta(y) # [b,n,2d]
        ab = ab.reshape(bn,ab.shape[-1]) # [bn,2d]
        a,b = ab.chunk(2,dim=-1) # each [bn,d]

        a = a[:,:,None,None] # [bn,c,h,w]
        b = b[:,:,None,None]

        x = x * (1. + a) + b
        return x