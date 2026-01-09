"""
Outlier penalty from Meta Movie Gen
"""

import torch
import einops as eo

def outlier_penalty_loss(z, r = 3, eps = 1.0e-6):
    # z is [b,c,h,w] or [b,t,c,h,w]
    rebatch = False
    if z.ndim == 5:
        b,t,c,h,w = z.shape
        orig_shape = (c,h,w)
        z = z.view(b*t,*orig_shape)
        rebatch = True

    orig_dtype = z.dtype
    z = z.float()

    mu = z.mean(dim=(2,3), keepdim=True)
    sigma = z.std(dim=(2,3), keepdim=True)

    loss = (z - mu.detach() + eps).norm(dim=1) - r * sigma.detach().norm(dim=1)
    loss = loss.clamp(min=0.0)
    loss = loss.to(orig_dtype)
    loss = loss.mean()

    return loss