import torch
import torch.nn.functional as F
from torch import nn

from .attn import Attn
from .mlp import MLP
from .modulation import AdaLN, Gate

from copy import deepcopy

class DiTBlock(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(config)
        self.adaln2 = AdaLN(config)
        self.gate1 = Gate(config)
        self.gate2 = Gate(config)

    def forward(self, x, cond, attn_mask = None):
        # x is [b,n,d]
        # cond is [b,d]

        # First block
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x, attn_mask)
        x = self.gate1(x, cond)
        x = res1 + x

        # Second block
        res2 = x.clone()
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = res2 + x

        return x

class FinalLayer(nn.Module):
    def __init__(self, config, skip_proj = False):
        super().__init__()

        channels = config.channels
        d_model = config.d_model
        patch_size = config.patch_size

        config = deepcopy(config)
        config.latent_size = 0

        self.norm = AdaLN(config)
        self.act = nn.SiLU()
        self.proj = nn.Sequential() if skip_proj else nn.Linear(d_model, channels*patch_size*patch_size)

    def forward(self, x, cond):
        x = self.norm(x, cond)
        x = self.act(x)
        x = self.proj(x)

        return x

class DiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(DiTBlock(config))
        self.blocks = nn.ModuleList(blocks)

        self.config = config
        self.n_latents = config.latent_size**2

    def forward(self, x, cond, attn_mask = None):
        for i, block in enumerate(self.blocks):
            x = block(x, cond, attn_mask)

        return x
