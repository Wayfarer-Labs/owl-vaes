import einops as eo
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

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
    max_q_len = None,
    max_kv_len = None
):
    """
    This will assume combined latents
    """
    h,w = int_to_tuple(config.sample_size)
    p_y, p_x = int_to_tuple(config.patch_size)
    n_frames = config.n_frames
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
        if is_video(idx):
            return idx // (n_p_y * n_p_x) # Which frame we're on
        else:
            shift_idx = idx - n_video
            return shift_idx // (config.latent_size ** 2)
        
    def row_idx(idx):
        if is_latent(idx):
            shift_idx = idx - n_video
            in_frame_idx = shift_idx % (config.latent_size ** 2)
            return in_frame_idx // config.latent_size
        else:
            in_frame_idx = idx % (n_p_y * n_p_x)
            return in_frame_idx // n_p_x
    
    def col_idx(idx):
        if is_latent(idx):
            shift_idx = idx - n_video
            in_frame_idx = shift_idx % (config.latent_size ** 2)
            return in_frame_idx % config.latent_size
        else:
            in_frame_idx = idx % (n_p_y * n_p_x)
            return in_frame_idx % n_p_x
        
    def can_attend_to(idx_i, idx_j):
        frame_idx_i = frame_idx(idx_i) + cache_frame_offset
        row_idx_i = row_idx(idx_i)
        col_idx_i = col_idx(idx_i)

        frame_idx_j = frame_idx(idx_j)
        row_idx_j = row_idx(idx_j)
        col_idx_j = col_idx(idx_j)

        return frame_idx_j <= frame_idx_i
    
    return create_block_mask(
        can_attend_to,
        B=batch_size,
        H=config.n_heads,
        Q_LEN=max_q_len,
        KV_LEN=max_kv_len,
        device=device,
    )

class Attn(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias = False)
        self.out = nn.Linear(config.d_model, config.d_model, bias = False)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        rope_impl = getattr(config, "rope_impl", None)
        if rope_impl is None:
            self.rope = None
        else:
            self.rope = get_rope_impl(rope_impl)(config)
        self.causal = config.causal

        self.layer_ind = None

        nn.init.zeros_(self.out.weight)

    def forward(self, x, kv_cache = None, attn_mask = None):
        # x: [b, n, d_model]
        b, n, d_model = x.shape
        h = self.n_heads
        d = d_model // h

        # Linear projection and split into q, k, v
        qkv = self.qkv(x)  # [b, n, 3 * d_model]
        qkv = qkv.view(b, n, 3, h, d)  # [b, n, 3, h, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, h, n, d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, h, n, d]

        q, k = self.qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k)
        x_out = F.scaled_dot_product_attention(q, k, v, is_causal = self.causal, attn_mask = attn_mask)
        #x_out = flex_attention(q,k,v)
        x_out = x_out.to(x.dtype)

        # Rearrange from [b, h, n, d] -> [b, n, h * d]
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        x_out = self.out(x_out)
        return x_out

class CausalAttn(nn.Module):
    """
    Causal attention for causal diffusion decoder
    """
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias = False)
        self.out = nn.Linear(config.d_model, config.d_model, bias = False)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.rope = get_rope_impl("video+latent")(config)

        self.layer_ind = None
        self.init.zeros_(self.out.weight)
    
    def forward(self, x, kv_cache = None, attn_mask = None):
        # x: [b, n, d_model]
        b, n, d_model = x.shape
        h = self.n_heads
        d = d_model // h

        # Linear projection and split into q, k, v
        qkv = self.qkv(x)  # [b, n, 3 * d_model]
        qkv = qkv.view(b, n, 3, h, d)  # [b, n, 3, h, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, h, n, d]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [b, h, n, d]

        q, k = self.qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k)

        x_out = flex_attention(q,k,v,block_mask=attn_mask)
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config) if not config.causal else CausalAttn(config)
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

class TemporalAttn(nn.Module):
    def __init__(self, d_model, n_heads, n_frames, layer_ind = None):
        super().__init__()

        self.n_frames = n_frames
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qk_norm = QKNorm(d_model // n_heads)
        self.rope = get_rope_impl("simple")(d_model, n_heads)
        self.layer_ind = layer_ind

    def forward(self, x, kv_cache = None):
        # x is [b*n, c, h, w]
        bn,c,h,w = x.shape
        b = bn//self.n_frames

        x = x.view(b, self.n_frames, c, h, w)
        x = x.permute(0,3,4,1,2).view(b*h*w,self.n_frames,c)
        # bhw,n,c

        qkv = self.qkv(x)
        qkv = qkv.view(b*h*w, self.n_frames, 3, n_heads, c//n_heads)
        qkv = qkv.permute(2,0,3,1,4) # 3,bhw,n_heads,n_frames,d_model//n_heads
        q,k,v = qkv[0], qkv[1], qkv[2]

        q,k = self.qk_norm(q,k)
        q,k = self.rope(q,k)

        x = flex_attention(q,k,v,is_causal=True)
        x = x.view(b,h,w,self.n_frames,c)
        x = x.permute(0,3,4,1,2)
        x = x.view(b*self.n_frames,c,h,w)

        return x

# ===== Conv ATTN =====

# TODO, replace my own deleted code

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
