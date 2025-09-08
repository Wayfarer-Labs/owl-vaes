import torch
import torch.nn.functional as F
from torch import nn

from .attn import MMAttn, Attn
from .mlp import MLP
from .modulation import AdaLN, Gate

def scatter(full_x: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
    """
    full_x: [b, n, d]  (destination buffer; modified in-place)
    x:      [b, m, d]  (values to insert)
    mask:   [b, n]     (exactly m True per row)

    Returns: full_x with x scattered into positions where mask==True (per batch row).
    """
    assert full_x.dim() == 3 and x.dim() == 3 and mask.dim() == 2
    b, n, d = full_x.shape
    bb, m, dd = x.shape
    bm, nm = mask.shape
    assert b == bb == bm and n == nm and d == dd, "shape mismatch"

    # Build per-row indices of True positions -> [b, m]
    sel_idx = torch.arange(n, device=x.device).expand(b, n)[mask].reshape(b, m)

    # Expand to match [b, m, d], so we can scatter along dim=1 (the n-dimension)
    index_n = sel_idx[:, :, None].expand(b, m, d)  # [b, m, d]

    # In-place scatter
    full_x.scatter_(dim=1, index=index_n, src=x)
    return full_x

class MMDiTBlock(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.config = config
        
        self.attn = MMAttn(config)

        self.mlp1 = MLP(config)
        self.mlp2 = MLP(config)

        self.adaln1_1 = AdaLN(config.d_model)
        self.adaln1_2 = AdaLN(config.d_model)
        self.gate1_1 = Gate(config.d_model)
        self.gate1_2 = Gate(config.d_model)

        self.adaln2_1 = AdaLN(config.d_model)
        self.adaln2_2 = AdaLN(config.d_model)
        self.gate2_1 = Gate(config.d_model)
        self.gate2_2 = Gate(config.d_model)

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

        self.adaln1 = AdaLN(config.d_model)
        self.adaln2 = AdaLN(config.d_model)
        self.gate1 = Gate(config.d_model)
        self.gate2 = Gate(config.d_model)

    def forward(self, x, cond, tread_mask = None):
        # x is [b,n,d]
        # cond is [b,d]

        # First block
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x, tread_mask = tread_mask)
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

        self.norm = AdaLN(d_model)
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

        self.tread = False
        if getattr(config, "tread", False):
            self.tread = True
            self.tread_i = 2
            self.tread_j = config.n_layers - 4
            self.tread_p = 0.5
        
        self.config = config
        self.n_latents = config.latent_size**2

    def forward(self, x, cond):
        if self.tread:
            # Tread mask is FALSE -> skip those
            # Create a mask with exactly 50% True and 50% False per batch element
            b, n_image = x[:,:-self.n_latents,0].shape
            n_true = n_image // 2
            tread_mask = torch.zeros((b, n_image), dtype=torch.bool, device=x.device)
            for i in range(b):
                perm = torch.randperm(n_image, device=x.device)
                tread_mask[i, perm[:n_true]] = True
            tread_mask = torch.cat([tread_mask, torch.ones_like(x[:,:self.n_latents,0])], dim=1) # [b,n]
            tread_mask = tread_mask.bool() # [b,n]
            tread_mask_input = None

            half_tokens = n_image // 2
            full_x = None
            full_x_heads = None
        else:
            tread_mask = None
            tread_mask_input = None
            full_x = None
            full_x_heads = None

        for i, block in enumerate(self.blocks):
            if self.tread:
                if i == self.tread_i:
                    full_x = x.clone()
                    x = x[tread_mask].view(b, -1, self.config.d_model) # 50% of tokens
                    tread_mask_input = tread_mask

                elif i == self.tread_j:
                    x = scatter(full_x, x, tread_mask)
                    tread_mask_input = None

            x = block(x, cond, tread_mask = tread_mask_input)

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
