import torch
from torch import nn
import torch.nn.functional as F
import einops as eo

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from ..utils import int_to_tuple
            
def get_rope_impl(impl_name):
    impl_name = impl_name.lower()
    if impl_name == "simple":
        return SimpleRoPE
    elif impl_name == "image":
        return ImageRoPE
    elif impl_name == "image+latent":
        return ImageRoPEWithLatent
    elif impl_name == "video+latent":
        return VideoRoPEWithLatents
    elif impl_name == "video":
        return VideoRoPE
    else:
        raise ValueError(f"Invalid rope implementation: {impl_name}")

class SimpleRoPE(nn.Module):
    """
    1D Rotary Positional Embedding for sequence data.
    """
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        n_heads = config.n_heads
        self.dim_head = d_model // n_heads
        self.rope = RotaryEmbedding(self.dim_head, freqs_for="lang", theta = 300)

    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k

    def apply(self, x):
        # x: [b, h, n, d]
        b, h, n, d = x.shape
        x_rope = self.rope.rotate_queries_or_keys(x.float()).to(x.dtype)
        return x_rope

class ImageRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        h, w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)

        n_p_y = h // p_y
        n_p_x = w // p_x

        dim_head = config.d_model // config.n_heads
        rope_emb = RotaryEmbedding(
            dim_head // 4,
            freqs_for = 'pixel',
            max_freq = 256
        )
        freqs = rope_emb.get_axial_freqs(
            n_p_y,
            n_p_x
        )
        self.register_buffer('freqs', freqs, persistent=False)
        self.n_p_y = n_p_y
        self.n_p_x = n_p_x

    def apply(self, x):
        # Assume x is [b,h,n,d], must reshape to [b,h,n_p_y,n_p_x,d]
        b,h,n,d = x.shape
        x = x.view(b,h,self.n_p_y,self.n_p_x,d)
        x = apply_rotary_emb(self.freqs.detach().double(), x.double()).to(x.dtype)
        x = x.view(b,h,n,d)
        return x
    
    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k

class ImageRoPEWithLatent(nn.Module):
    """
    Same as above but assumes the last N tokens are latent
    """
    def __init__(self, config):
        super().__init__()

        h, w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)

        n_p_y = h // p_y
        n_p_x = w // p_x

        n_p_y_rope = n_p_y + config.latent_size
        n_p_x_rope = n_p_x + config.latent_size

        dim_head = config.d_model // config.n_heads
        rope_emb = RotaryEmbedding(
            dim_head // 4,
            freqs_for = 'pixel',
            max_freq = min(n_p_y_rope, n_p_x_rope) * 0.8
        )
        freqs = rope_emb.get_axial_freqs(
            n_p_y_rope,
            n_p_x_rope
        )
        self.register_buffer('freqs', freqs, persistent=False)
        self.n_p_y = n_p_y
        self.n_p_x = n_p_x
        self.n_image = n_p_y * n_p_x
        self.n_latent = config.latent_size
        self.n_heads = config.n_heads

    def apply(self, x):
        x_image = x[:,:,:self.n_image]
        x_latent = x[:,:,self.n_image:]
        
        # Assume x is [b,h,n,d], must reshape to [b,h,n_p_y,n_p_x,d]
        b,h,n,d = x_image.shape
        x_image = x_image.view(b,h,self.n_p_y,self.n_p_x,d)
        x_latent = x_latent.view(b,h,self.n_latent,self.n_latent,d)

        # We want a "big image" where the latent is the bottom right corner
        # This means adding padding to x_image on the right and bottom
        # The latent is rolled into bottom padding
        pad_right = torch.zeros((b,h,self.n_p_y,self.n_latent,d), dtype = x.dtype, device = x.device)
        pad_bottom = torch.zeros((b,h,self.n_latent,self.n_p_x,d), dtype = x.dtype, device = x.device)
        pad_bottom = torch.cat([pad_bottom, x_latent], dim = 3)
        
        # Now add padding and padding + latent
        x_image = torch.cat([x_image, pad_right], dim = 3)
        x_image = torch.cat([x_image, pad_bottom], dim = 2) # [h,w] -> [h+l,w+l]

        x_image = apply_rotary_emb(self.freqs.detach().float(), x_image.float()).to(x_image.dtype)
        x_latent = x_image[:,:,self.n_p_y:,self.n_p_x:].contiguous()
        x_image = x_image[:,:,:self.n_p_y,:self.n_p_x].contiguous()

        x_image = x_image.view(b,h,self.n_image,d)
        x_latent = x_latent.view(b,h,self.n_latent**2,d)
        x = torch.cat([x_image, x_latent], dim = 2)

        return x
    
    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k

class VideoRoPEWithLatents(nn.Module):
    def __init__(self, config):
        super().__init__()

        h,w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)
        n_frames = config.n_frames

        n_p_y = h // p_y
        n_p_x = w // p_x
        n_p_y_rope = n_p_y + config.latent_size
        n_p_x_rope = n_p_x + config.latent_size

        dim_head = config.d_model // config.n_heads
        rope_emb = RotaryEmbedding(
            dim_head // 8,
            freqs_for = 'pixel',
            max_freq = min(n_p_y_rope, n_p_x_rope) * 0.8
        )
        freqs = rope_emb.get_axial_freqs(
            n_frames,
            n_p_y_rope,
            n_p_x_rope
        )
        self.register_buffer('freqs', freqs, persistent=False)
        self.n_p_y = n_p_y
        self.n_p_x = n_p_x
        self.n_latent = config.latent_size
        self.n_frames = n_frames
        self.n_heads = config.n_heads

        self.n_images = n_p_y * n_p_x
        self.n_video = n_frames * self.n_images

    def apply(self, x):
        orig_dtype = x.dtype

        # x is bhnd
        x_video = x[:,:,:self.n_video]
        x_latent = x[:,:,self.n_video:]


        b,h,n,d = x_video.shape
        x_video = eo.rearrange(
            x_video,
            'b h (n_frames n_p_y n_p_x) d -> b h n_frames n_p_y n_p_x d',
            n_frames = self.n_frames,
            n_p_y = self.n_p_y,
        )
        x_latent = eo.rearrange(
            x_latent,
            'b h (n_frames n_latent_1 n_latent_2) d -> b h n_frames n_latent_1 n_latent_2 d',
            n_frames = self.n_frames,
            n_latent_1 = self.n_latent,
            n_latent_2 = self.n_latent,
        )

        # "big video" where bottom right is latent
        pad_right = torch.zeros((b,h,self.n_frames,self.n_p_y,self.n_latent,d), dtype = x.dtype, device = x.device)
        pad_bottom = torch.zeros((b,h,self.n_frames,self.n_latent,self.n_p_x,d), dtype = x.dtype, device = x.device)
        pad_bottom = torch.cat([pad_bottom, x_latent], dim = 4)
            
        # Now add padding and padding + latent
        x_video = torch.cat([x_video, pad_right], dim = 4)
        x_video = torch.cat([x_video, pad_bottom], dim = 3) # [h,w] -> [h+l,w+l]

        x_video = apply_rotary_emb(self.freqs.detach().float(), x_video.float()).to(x_video.dtype)
        x_latent = x_video[:,:,:,self.n_p_y:,self.n_p_x:].contiguous()
        x_video = x_video[:,:,:,:self.n_p_y,:self.n_p_x].contiguous()

        x_video = eo.rearrange(
            x_video,
            'b h n_frames n_p_y n_p_x d -> b h (n_frames n_p_y n_p_x) d'
        )
        x_latent = eo.rearrange(
            x_latent,
            'b h n_frames n_latent_1 n_latent_2 d -> b h (n_frames n_latent_1 n_latent_2) d',
            n_frames = self.n_frames,
            n_latent_1 = self.n_latent,
            n_latent_2 = self.n_latent,
        )
        x = torch.cat([x_video, x_latent], dim = 2)

        return x.to(orig_dtype)
    
    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k

class VideoRoPE(nn.Module):
    """
    Same as above but without any latents (simplifies a lot)
    """
    def __init__(self, config):
        super().__init__()

        h,w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)
        n_frames = config.n_frames

        n_p_y = h // p_y
        n_p_x = w // p_x

        dim_head = config.d_model // config.n_heads
        rope_emb = RotaryEmbedding(
            dim_head // 8,
            freqs_for = 'pixel',
            max_freq = min(n_p_y, n_p_x) * 0.8
        )
        freqs = rope_emb.get_axial_freqs(
            n_frames,
            n_p_y,
            n_p_x
        )
        self.register_buffer('freqs', freqs, persistent=False)

        self.n_p_y = n_p_y
        self.n_p_x = n_p_x
        self.n_heads = config.n_heads

    def apply(self, x):
        orig_dtype = x.dtype

        b,h,n,d = x.shape
        x = eo.rearrange(
            x,
            'b h (n_frames n_p_y n_p_x) d -> b h n_frames n_p_y n_p_x d',
            n_p_y = self.n_p_y,
            n_p_x = self.n_p_x,
        )
        x = apply_rotary_emb(self.freqs[-x.shape[2]:].detach().float(), x.float()).to(x.dtype)
        x = eo.rearrange(
            x,
            'b h n_frames n_p_y n_p_x d -> b h (n_frames n_p_y n_p_x) d',
        )
        return x.to(orig_dtype)
    
    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k

if __name__ == "__main__":
    # Print shapes of cos and sin

    from ..configs import Config
    cfg = Config.from_yaml("configs/feats_c128/cod_128x_feats_diffdec.yml")
    rope = ImageRoPEWithLatent(cfg.model)

    # Assume 8x8 image, 1x1 patch
    # flattened to b,h,64,d
    n_p_y = cfg.model.sample_size[0] // cfg.model.patch_size
    n_p_x = cfg.model.sample_size[1] // cfg.model.patch_size

    q = torch.randn(1,cfg.model.n_heads,n_p_y * n_p_x + cfg.model.latent_size**2,cfg.model.d_model)
    k = torch.randn(1,cfg.model.n_heads,n_p_y * n_p_x + cfg.model.latent_size**2,cfg.model.d_model)

    q, k = rope(q, k)
    print(q.shape)
    print(k.shape)