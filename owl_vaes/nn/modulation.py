import torch
from torch import nn
import torch.nn.functional as F

from .normalization import LayerNorm

def int_to_tuple(x):
    if isinstance(x, int):
        return (x,x)
    elif isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        try:
            return tuple(x)

class _AdaLN(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim_in = config.d_model
        dim_out = config.d_model

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

class VideoAdaLN(nn.Module):
    """
    AdaLN for video and latent data
    Assumes latents are after entire video in sequence
    """
    def __init__(self, config):
        super().__init__()

        dim_in = config.d_model
        dim_out = config.d_model

        self.fc = nn.Linear(dim_in, 2 * dim_out)
        self.norm = LayerNorm(dim_out)
        self.n_frames = config.n_frames

        h,w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)
        self.n_p_y = h // p_y
        self.n_p_x = w // p_x

        self.n_latent = config.latent_size
    
    def broadcast_to_video_and_latent(self, x):
        # x is [b,n_frames,d]
        x_1 = eo.repeat(
            x,
            'b n_frames d -> b (n_frames h w) d',
            h = self.n_p_y,
            w = self.n_p_x
        )
        x_2 = eo.repeat(
            x,
            'b n_frames d -> b (n_frames h w) d',
            h = self.n_latent,
            w = self.n_latent
        )
        x = torch.cat([x_1, x_2], dim = 1).contiguous()
        return x

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,n_frames]
        y = F.silu(cond)
        ab = self.fc(y) # [b,n_frames,2d]
        a,b = ab.chunk(2,dim=-1) # each [b,n_frames,d]
        a = self.broadcast_to_video_and_latent(a)
        b = self.broadcast_to_video_and_latent(b)

        x = self.norm(x) * (1. + a) + b
        return x

def AdaLN(config):
    if hasattr(config, 'n_frames') and config.n_frames > 1 and config.causal:
        return VideoAdaLN(config)
    else:
        return _AdaLN(config)

class _Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim_in = config.d_model
        dim_out = config.d_model

        self.fc_c = nn.Linear(dim_in, dim_out)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        b,n,d = x.shape

        y = F.silu(cond)
        c = self.fc_c(y) # [b,d]
        c = c[:,None,:].expand(b,n,-1) # [b,n,d]

        return c * x

class VideoGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        dim_in = config.d_model
        dim_out = config.d_model
        
        self.fc_c = nn.Linear(dim_in, dim_out)
        h, w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)
        self.n_p_y = h // p_y
        self.n_p_x = w // p_x
        self.n_latent = config.latent_size

    def broadcast_to_video_and_latent(self, x):
        # x is [b,n_frames,d]
        x_1 = eo.repeat(
            x,
            'b n_frames d -> b (n_frames h w) d',
            h = self.n_p_y,
            w = self.n_p_x
        )
        x_2 = eo.repeat(
            x,
            'b n_frames d -> b (n_frames h w) d',
            h = self.n_latent,
            w = self.n_latent
        )
        x = torch.cat([x_1, x_2], dim = 1).contiguous()
        return x
    
    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,n_frames]
        y = F.silu(cond)
        c = self.fc_c(y) # [b,n_frames,d]
        c = self.broadcast_to_video_and_latent(c)
        return c * x

class Gate(config):
    if hasattr(config, 'n_frames') and config.n_frames > 1 and config.causal:
        return VideoGate(config)
    else:
        return _Gate(config)

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