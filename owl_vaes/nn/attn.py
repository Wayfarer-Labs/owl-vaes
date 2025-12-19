import einops as eo
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import einops as eo

from owl_vaes.configs import TransformerConfig

from .mimetic import mimetic_init
from .mlp import MLP
from .normalization import LayerNorm, QKNorm
from .rope import get_rope_impl
from ..utils import int_to_tuple

torch.backends.cuda.enable_flash_sdp(enabled=True)
flex_attention = torch.compile(flex_attention)

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

def get_attn_mask(
    config,
    cache_frame_offset = 0,
    batch_size = None,
    device = None,
    max_q_len = 1024,
    max_kv_len = 1024,
    kernel_size = None,
):
    """
    This will assume combined latents
    """
    h,w = int_to_tuple(config.sample_size)
    p_y, p_x = int_to_tuple(config.patch_size)
    if kernel_size is not None:
        k_y, k_x = int_to_tuple(kernel_size)

    n_frames = getattr(config, "n_frames", 1)
    n_p_y = h // p_y
    n_p_x = w // p_x
    n_p_y_rope = n_p_y + config.latent_size
    n_p_x_rope = n_p_x + config.latent_size
    n_video = n_frames * n_p_y * n_p_x
    n_latent = config.latent_size ** 2 * n_frames

    def is_latent(idx):
        return idx >= n_video
    
    def is_video(idx):
        return idx < n_video

    def frame_idx(idx):
        video_result = idx // (n_p_y * n_p_x)
        shift_idx = idx - n_video
        latent_result = shift_idx // (config.latent_size ** 2)
        return torch.where(is_video(idx), video_result, latent_result)
        
    def row_idx(idx):
        # Latent case
        shift_idx = idx - n_video
        in_frame_idx_latent = shift_idx % (config.latent_size ** 2)
        latent_result = in_frame_idx_latent // config.latent_size

        # Video case
        in_frame_idx_video = idx % (n_p_y * n_p_x)
        video_result = in_frame_idx_video // n_p_x

        return torch.where(is_latent(idx), latent_result, video_result)
    
    def col_idx(idx):
        # Latent case
        shift_idx = idx - n_video
        in_frame_idx_latent = shift_idx % (config.latent_size ** 2)
        latent_result = in_frame_idx_latent % config.latent_size

        # Video case
        in_frame_idx_video = idx % (n_p_y * n_p_x)
        video_result = in_frame_idx_video % n_p_x

        return torch.where(is_latent(idx), latent_result, video_result)
        
    def can_attend_to(b,h,idx_i, idx_j):
        frame_idx_i = frame_idx(idx_i)
        row_idx_i = row_idx(idx_i)
        col_idx_i = col_idx(idx_i)

        frame_idx_j = frame_idx(idx_j)
        row_idx_j = row_idx(idx_j)
        col_idx_j = col_idx(idx_j)

        if kernel_size is None:
            return frame_idx_j <= frame_idx_i
        else:
            causal_mask = (frame_idx_j <= frame_idx_i)

            # If i is image, it can attend to images in neighbourhood
            image_nbr_mask = (torch.abs(row_idx_j - row_idx_i) <= k_y) & (torch.abs(col_idx_j - col_idx_i) <= k_x)
            
            # If i is latent, it can attend to any other latent in its frame
            # It can't attend to any image tokens
            # If i is image, it can attend to any latent tokens or image tokens in neighbourhood
            nbr_mask = torch.where(
                is_latent(idx_i),
                is_latent(idx_j),
                is_latent(idx_j) | image_nbr_mask
            )

            return causal_mask & nbr_mask
    
    return create_block_mask(
        can_attend_to,
        B=batch_size,
        H=config.n_heads,
        Q_LEN=max_q_len,
        KV_LEN=max_kv_len,
        device=device,
    )

class Attn(nn.Module):
    """
    Causal attention for causal diffusion decoder
    """
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias = False)
        self.out = nn.Linear(config.d_model, config.d_model, bias = False)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.rope = get_rope_impl(config.rope_impl)(config)

        self.layer_ind = None
        nn.init.zeros_(self.out.weight)
    
    def forward(self, x, kv_cache = None, attn_mask = None):
        # x: [b, n, d_model]

        # Linear projection and split into q, k, v
        qkv = self.qkv(x)  # [b, n, 3 * d_model]
        q,k,v = eo.rearrange(qkv, 'b n (three h d) -> three b h n d', h = self.n_heads, three = 3)

        q, k = self.qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        x_out = flex_attention(q,k,v,block_mask=attn_mask)

        x_out = eo.rearrange(x_out, 'b h n d -> b n (h d)')
        x_out = self.out(x_out)
        return x_out

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache = None, attn_mask = None):
        res1 = x.clone()
        x = self.norm1(x)
        x = self.attn(x, kv_cache, attn_mask)
        x = res1 + x

        res2 = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = res2 + x

        return x

class StackedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for i in range(config.n_layers):
            blocks.append(Transformer(config))
            blocks[i].attn.layer_ind = i
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, kv_cache = None, attn_mask = None):
        for block in self.blocks:
            x = block(x, kv_cache, attn_mask)

        return x

# === VIT Specific Layers ===

class PatchProjIn(nn.Module):
    def __init__(self, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.proj_in = nn.Conv2d(channels, d_model, patch_size, patch_size, 0, bias=False)

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.proj_in(x)
        x = eo.rearrange(x, 'b c h w -> b (h w) c')
        return x

class PatchProjOut(nn.Module):
    def __init__(self, sample_size, d_model, channels = 3, patch_size=1):
        super().__init__()

        self.norm = LayerNorm(d_model)
        self.act = nn.SiLU()
        self.proj = nn.Linear(d_model, channels*patch_size*patch_size)
        self.sample_size = sample_size
        self.patch_size = patch_size

        self.n_patches = self.sample_size//self.patch_size

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        x = eo.rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)', h = self.n_patches, ph = self.patch_size, pw = self.patch_size)

        return x

def attn_test():
    cfg = TransformerConfig(
        sample_size = 16,
        channels = 32,
        latent_size = 16,
        latent_channels = 128,
        n_layers = 6,
        n_heads = 6,
        d_model = 384,
        patch_size = 1,
        causal = False,
        mimetic_init = False
    )

    # Test Attention layer
    attn = Attn(cfg).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 256, 384).bfloat16().cuda()
        y = attn(x)
        assert y.shape == (1, 256, 384), f"Expected shape (1,256,384), got {y.shape}"

    # Test Transformer layer
    transformer = Transformer(cfg).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 256, 384).bfloat16().cuda()
        y = transformer(x)
        assert y.shape == (1, 256, 384), f"Expected shape (1,256,384), got {y.shape}"

    # Test StackedTransformer
    stacked = StackedTransformer(cfg).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 256, 384).bfloat16().cuda()
        y = stacked(x)
        assert y.shape == (1, 256, 384), f"Expected shape (1,256,384), got {y.shape}"

    # Test PatchProjIn
    patch_in = PatchProjIn(384, 32, 1).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 32, 16, 16).bfloat16().cuda()
        y = patch_in(x)
        assert y.shape == (1, 256, 384), f"Expected shape (1,256,384), got {y.shape}"

    # Test PatchProjOut
    patch_out = PatchProjOut(16, 384, 32, 1).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 256, 384).bfloat16().cuda()
        y = patch_out(x)
        assert y.shape == (1, 32, 16, 16), f"Expected shape (1,32,16,16), got {y.shape}"

    print("All Tests Passed!")
    
if __name__ == "__main__":
    attn_test()
