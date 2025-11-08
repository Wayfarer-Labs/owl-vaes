import torch
from torch import nn
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

def scatter_heads(full_x: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
    """
    full_x: [b, h, n, d]  (destination buffer; modified in-place)
    x:      [b, h, m, d]  (values to insert)
    mask:   [b, n]        (exactly m True per row)

    Returns: full_x with x scattered into positions where mask==True (per batch row).
    """
    assert full_x.dim() == 4 and x.dim() == 4 and mask.dim() == 2
    b, h, n, d = full_x.shape
    bb, hh, m, dd = x.shape
    bm, nm = mask.shape
    assert b == bb == bm and h == hh and n == nm and d == dd, "shape mismatch"
    # Build per-row source indices (where mask is True) -> [b, m]
    sel_idx = torch.arange(n, device=x.device).expand(b, n)[mask].reshape(b, m)  # [b, m]
    # Expand to match [b, h, m, d] so we can scatter along dim=2 (the n-dimension)
    index_n = sel_idx[:, None, :, None].expand(b, h, m, d)                      # [b, h, m, d]
    # In-place scatter: write x into full_x at those positions
    full_x.scatter_(dim=2, index=index_n, src=x)
    return full_x

def int_to_tuple(x):
    if isinstance(x, int):
        return (x,x)
    elif isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        try:
            return tuple(x)
        except:
            raise ValueError(f"Invalid input: {x}")

def get_rope_impl(impl_name):
    impl_name = impl_name.lower()
    if impl_name == "simple":
        return SimpleRoPE
    elif impl_name == "image":
        return ImageRoPE
    elif impl_name == "image+latent":
        return ImageRoPEWithLatent
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

    def forward(self, q, k, tread_mask = None):
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
            max_freq = 256
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

    def apply(self, x, tread_mask = None):
        b_orig,h_orig,n_orig,d_orig = x.shape
        if tread_mask is not None:
            full_x = torch.zeros(b_orig,h_orig,tread_mask.shape[1],d_orig, device=x.device, dtype=x.dtype)
            x = scatter_heads(full_x, x, tread_mask)

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

        if tread_mask is not None:
            tread_mask_head = tread_mask[:,None,:].repeat(1,self.n_heads,1)
            x = x[tread_mask_head].view(b_orig,h_orig,-1,d_orig)

        return x
    
    def forward(self, q, k, tread_mask = None):
        q = self.apply(q, tread_mask)
        k = self.apply(k, tread_mask)
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