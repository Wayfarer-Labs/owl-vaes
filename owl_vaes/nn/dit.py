import torch
import torch.nn.functional as F
from torch import nn

from .attn import MMAttn, Attn
from .mlp import MLP
from .modulation import AdaLN, Gate

class MMDiTBlock(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.config = config
        
        self.attn = MMAttn(config)

        self.mlp1 = MLP(config)
        self.mlp2 = MLP(config)

        self.adaln1_1 = AdaLN(config)
        self.adaln1_2 = AdaLN(config)
        self.gate1_1 = Gate(config)
        self.gate1_2 = Gate(config)

        self.adaln2_1 = AdaLN(config)
        self.adaln2_2 = AdaLN(config)
        self.gate2_1 = Gate(config)
        self.gate2_2 = Gate(config)

        self.n = (config.sample_size // config.patch_size)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]
        x_1 = x[:,:self.n]
        x_2 = x[:,self.n:]

        # First block
        res1_1 = x_1.clone()
        res1_2 = x_2.clone()

        x_1 = self.adaln1_1(x_1, cond)
        x_2 = self.adaln1_2(x_2, cond)
        
        x_1, x_2 = self.attn(x_1, x_2)
        
        x_1 = self.gate1_1(x_1, cond)
        x_2 = self.gate1_2(x_2, cond)
        
        x_1 = res1_1 + x_1
        x_2 = res1_2 + x_2

        # Second block
        res2_1 = x_1.clone()
        res2_2 = x_2.clone()

        x_1 = self.adaln2_1(x_1, cond)
        x_2 = self.adaln2_2(x_2, cond)
        
        x_1 = self.mlp1(x_1)
        x_2 = self.mlp2(x_2)
        
        x_1 = self.gate2_1(x_1, cond)
        x_2 = self.gate2_2(x_2, cond)
        
        x_1 = res2_1 + x_1
        x_2 = res2_2 + x_2

        return torch.cat([x_1, x_2], dim=1)

class DiTBlock(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.attn = Attn(config)
        self.mlp = MLP(config)

        self.adaln1 = AdaLN(config)
        self.adaln2 = AdaLN(config)
        self.gate1 = Gate(config)
        self.gate2 = Gate(config)

    def forward(self, x, cond):
        # x is [b,n,d]
        # cond is [b,d]

        # First block
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x)
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

    def forward(self, x, cond):
        for i, block in enumerate(self.blocks):
            x = block(x, cond)

        return x

class UViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        blocks = []
        for _ in range(config.n_layers):
            blocks.append(MMDiTBlock(config))
        self.blocks = nn.ModuleList(blocks)

        # For odd number of layers, need linear projections for skip connections
        n_skip_connections = config.n_layers // 2
        skip_projs = []
        for _ in range(n_skip_connections):
            skip_projs.append(nn.Linear(config.d_model * 2, config.d_model))
        self.skip_projs = nn.ModuleList(skip_projs)

    def forward(self, x, cond):
        # Cache early block outputs for skip connections
        early_features = []
        n_blocks = len(self.blocks)
        mid_idx = n_blocks // 2

        # Early blocks
        for i in range(mid_idx):
            x = self.blocks[i](x, cond)
            early_features.append(x)

        # Middle block (if odd number of layers)
        x = self.blocks[mid_idx](x, cond)

        # Late blocks with skip connections
        for i in range(mid_idx + 1, n_blocks):
            # Get corresponding early block output
            early_idx = n_blocks - 1 - i
            early_feat = early_features[early_idx]
            
            # Concatenate early and current features
            skip_idx = i - (mid_idx + 1)
            x = torch.cat([x, early_feat], dim=-1)
            x = self.skip_projs[skip_idx](x)
            
            x = self.blocks[i](x, cond)

        return x
