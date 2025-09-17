import torch
import einops as eo
from .schedulers import get_sd3_euler

class SameNoiseSampler:
    def __init__(self, n_steps):
        self.n_steps = n_steps

    @torch.no_grad()
    def __call__(self, encoder, scale, denoiser, samples):
        # samples is a video [n,c,h,w]
        z = encoder(samples)[0] / scale
        
        rgb = samples[:,:3]
        noisy = torch.randn_like(rgb[0])
        noisy = eo.repeat(noisy, 'c h w -> n c h w', n=z.shape[0])

        dt = get_sd3_euler(self.n_steps)
        t = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)

        for dt_i in dt:
            v = denoiser(noisy, z, t)
            noisy = noisy - dt_i * v
            t = t - dt_i

        return noisy