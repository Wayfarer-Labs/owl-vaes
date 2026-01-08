"""
Outlier penalty from Meta Movie Gen
"""

import torch
import einops as eo

def outlier_penalty_loss(z, r = 3):
    # z is [b,c,h,w] or [b,t,c,h,w]
    rebatch = False
    if z.ndim == 5:
        b,t,c,h,w = z.shape
        orig_shape = (c,h,w)
        z = z.view(b*t,*orig_shape)
        rebatch = True

    mu = z.mean(dim=(2,3), keepdim=True)
    sigma = z.std(dim=(2,3), keepdim=True)

    loss = (z - mu).norm(dim=1) - r * sigma.norm(dim=1)
    loss = loss.clamp(min=0.0)
    loss = loss.mean()

    return loss