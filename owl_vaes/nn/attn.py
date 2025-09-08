import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from owl_vaes.configs import TransformerConfig

from .mimetic import mimetic_init
from .mlp import MLP
from .normalization import LayerNorm, QKNorm
from .rope import get_rope_impl

torch.backends.cuda.enable_flash_sdp(enabled=True)

from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

try:
    from flash_cosine_sim_attention import flash_cosine_sim_attention
except:
    print("Warning: Failed to import Flash Cosine Attn. It's only needed for HDiT so you can ignore if not needed.")

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention)

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

    def forward(self, x, kv_cache = None, tread_mask = None):
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
            q, k = self.rope(q, k, tread_mask)
        x_out = F.scaled_dot_product_attention(q, k, v, is_causal = self.causal)
        #x_out = flex_attention(q,k,v)
        x_out = x_out.to(x.dtype)

        # Rearrange from [b, h, n, d] -> [b, n, h * d]
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        x_out = self.out(x_out)
        return x_out

class CosSimAttn(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)
        self.rope = ImageRoPEWithLatent(config) if config.rope_impl == "image+latent" else ImageRoPE(config)
        self.causal = config.causal

        self.layer_ind = None

    def forward(self, x, kv_cache = None):
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
        q, k = self.rope(q, k)
        x_out = flash_cosine_sim_attention(q, k, v, causal = False)
        x_out = x_out.to(x.dtype)

        # Rearrange from [b, h, n, d] -> [b, n, h * d]
        x_out = x_out.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        x_out = self.out(x_out)
        return x_out

class MMAttn(nn.Module):
    """
    MMDiT style attention
    """
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv_1 = nn.Linear(config.d_model, 3 * config.d_model)
        self.qkv_2 = nn.Linear(config.d_model, 3 * config.d_model)

        self.out_1 = nn.Linear(config.d_model, config.d_model)
        self.out_2 = nn.Linear(config.d_model, config.d_model)

        self.qk_norm_1 = QKNorm(config.d_model // config.n_heads)
        self.qk_norm_2 = QKNorm(config.d_model // config.n_heads)

    def split(self, qkv):
        return eo.rearrange(qkv, 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)

    def merge(self, x):
        return eo.rearrange(x, 'b h n d -> b n (h d)')

    def forward(self, x_1, x_2):
        n1 = x_1.shape[1]

        q1,k1,v1 = self.split(self.qkv_1(x_1))
        q2,k2,v2 = self.split(self.qkv_2(x_2))

        q1,k1 = self.qk_norm_1(q1,k1)
        q2,k2 = self.qk_norm_2(q2,k2)

        q = torch.cat([q1,q2],dim=-2)
        k = torch.cat([k1,k2],dim=-2)
        v = torch.cat([v1,v2],dim=-2)

        x = F.scaled_dot_product_attention(q,k,v)
        x = self.merge(x)
        
        x_1, x_2 = x[:,:n1], x[:,n1:]
        x_1 = self.out_1(x_1)
        x_2 = self.out_2(x_2)

        return x_1, x_2
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache = None):
        res1 = x.clone()
        x = self.norm1(x)
        x = self.attn(x, kv_cache)
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

    def forward(self, x, kv_cache = None):
        for block in self.blocks:
            x = block(x, kv_cache)

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
