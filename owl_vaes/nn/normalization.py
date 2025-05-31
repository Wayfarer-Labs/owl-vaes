import torch
from torch import nn


class GroupNorm(nn.Module):
    def __init__(self, dim, groups = 32):
        super().__init__()
        self.groups = groups
        self.gain = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, self.groups, c // self.groups, h, w)

        var = x.float().var(dim=(2,3,4), keepdim=True, unbiased=False)
        std = (var + 1.0e-6).sqrt()
        x = x / std.to(x.dtype)

        x = x.reshape(b, c, h, w)
        return x * (1. + self.gain[None,:,None,None])

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # small init to default to no gain
        self.gain = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        b,h,n,d = x.shape
        gain = self.gain[None,None,None,:] # [1,1,1,d]

        gain = (1. + gain)
        rms = (x.float().pow(2).mean(-1,keepdim=True)+1.0e-6).rsqrt().to(x.dtype)

        return x * rms * gain

class RMSNorm2d(RMSNorm):
    def forward(self, x):
        # x is [b,c,h,w]
        x = x.permute(0,2,3,1) # -> [b,h,w,c]
        x = super().forward(x)
        x = x.permute(0,3,1,2)
        return x

class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q_norm = RMSNorm(dim)
        self.k_norm = RMSNorm(dim)

    def forward(self, q, k):
        return self.q_norm(q), self.k_norm(k)

def LayerNorm(dim, elementwise_affine=False, **kwargs):
    return nn.LayerNorm(dim, elementwise_affine=elementwise_affine, **kwargs)
