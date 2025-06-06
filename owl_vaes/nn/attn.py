import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.get_device import DeviceManager
from ..configs import TransformerConfig

from .mimetic import mimetic_init
from .mlp import MLP
from .normalization import LayerNorm, QKNorm

torch.backends.cuda.enable_flash_sdp(enabled=True)

class Attn(nn.Module):
    def __init__(self, config : TransformerConfig):
        super().__init__()

        self.n_heads = config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)

        self.qk_norm = QKNorm(config.d_model // config.n_heads)

        if config.mimetic_init:
            mimetic_init(self.qkv, self.out, config)
        self.causal = config.causal

    def forward(self, x):

        q,k,v = eo.rearrange(self.qkv(x), 'b n (three h d) -> three b h n d', three = 3, h = self.n_heads)
        q,k = self.qk_norm(q,k)
        x = F.scaled_dot_product_attention(q,k,v,is_causal=self.causal)
        x = eo.rearrange(x, 'b h n d -> b n (h d)')
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        dim = config.d_model

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.attn = Attn(config)
        self.mlp = MLP(config)

    def forward(self, x):
        res1 = x.clone()
        x = self.norm1(x)
        x = self.attn(x)
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
        for _ in range(config.n_layers):
            blocks.append(Transformer(config))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

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
