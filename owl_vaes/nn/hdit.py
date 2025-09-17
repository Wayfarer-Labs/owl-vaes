import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .attn import Attn
from .mlp import MLPSimple
from .modulation import AdaLN, Gate
from .dit import DiTBlock, DiT

from .normalization import LayerNorm

from natten import NeighborhoodAttention2D
from copy import deepcopy

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

class HDiTAdaLN(nn.Module):
    def __init__(self, dim_in, dim_out = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        self.norm = LayerNorm(dim_out)
        self.proj = weight_norm(nn.Conv2d(dim_in, 2*dim_out, 1, 1, 0, bias = False))

    def forward(self, x, cond):
        # x is [b,n_p_y,n_p_x,d]
        # cond is [b,h,w,d]

        x = self.norm(x)
        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]
        cond = cond.permute(0,3,1,2).contiguous() # [b,d,h,w]

        cond = F.silu(cond)
        cond = F.interpolate(cond, size = x.shape[2:], mode = "nearest")
        cond = self.proj(cond) # [b,2d,n_p_y,n_p_x]

        a,b = cond.chunk(2,dim=1) # each [b,d,n_p_y,n_p_x]
        x = x * (1. + a) + b

        x = x.permute(0,2,3,1).contiguous() # [b,n_p_y,n_p_x,d]
        return x

class HDiTGate(nn.Module):
    def __init__(self, dim_in, dim_out = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        self.proj = weight_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias = False))

    def forward(self, x, cond):
        # x is [b,n_p_y,n_p_x,d]
        # cond is [b,h,w,d]

        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]
        cond = cond.permute(0,3,1,2).contiguous() # [b,d,h,w]

        cond = F.silu(cond)
        cond = F.interpolate(cond, size = x.shape[2:], mode = "nearest")
        cond = self.proj(cond) # [b,d,n_p_y,n_p_x]

        x = x * cond

        x = x.permute(0,2,3,1).contiguous() # [b,n_p_y,n_p_x,d]

        return x

class HDiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, cond_dim):
        super().__init__()

        self.attn = NeighborhoodAttention2D(
            embed_dim=d_model,
            num_heads=n_heads,
            kernel_size=7,  # 7x7 neighborhood window
            stride=1,
            dilation=1,
            is_causal=False
        )
        self.mlp = MLPSimple(d_model)

        self.adaln1 = HDiTAdaLN(cond_dim, d_model)
        self.adaln2 = HDiTAdaLN(cond_dim, d_model)
        self.gate1 = HDiTGate(cond_dim, d_model)
        self.gate2 = HDiTGate(cond_dim, d_model)

    def forward(self, x, cond):
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

class HDiTUpSample(nn.Module):
    def __init__(self, d_model, d_model_out):
        super().__init__()

        self.proj = weight_norm(nn.Conv2d(d_model, d_model_out * 4, 1, 1, 0, bias = False))

    def forward(self, x):
        # x is [b,n_p_y,n_p_x,d]
        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]
        x = self.proj(x)
        x = F.pixel_shuffle(x, 2) # [b,4*d,h,w] -> [b,d,h*2,w*2]
        x = x.permute(0,2,3,1).contiguous() # [b,h,w,d]
        return x

class HDiTDownSample(nn.Module):
    def __init__(self, d_model, d_model_out):
        super().__init__()

        self.proj = weight_norm(nn.Conv2d(4*d_model, d_model_out, 1, 1, 0, bias = False))

    def forward(self, x):
        # x is [b,n_p_y,n_p_x,d]
        x = x.permute(0,3,1,2).contiguous() # [b,d,h,w]
        x = F.pixel_unshuffle(x, 2).contiguous() # [b,d,h,w] -> [b,4*d,h/2,w/2]
        x = self.proj(x)
        x = x.permute(0,2,3,1).contiguous() # [b,h*2,w*2,d])
        return x

class DownStage(nn.Module):
    def __init__(self, d_model, n_heads, d_model_out, cond_dim):
        super().__init__()

        self.block1 = HDiTBlock(d_model, n_heads, cond_dim)
        self.block2 = HDiTBlock(d_model, n_heads, cond_dim)
        self.down = HDiTDownSample(d_model, d_model_out)

    def forward(self, x, cond):
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.down(x)
        return x

class UpStage(nn.Module):
    def __init__(self, d_model, n_heads, d_model_in, cond_dim):
        super().__init__()

        self.up = HDiTUpSample(d_model_in, d_model)
        self.block1 = HDiTBlock(d_model, n_heads, cond_dim)
        self.block2 = HDiTBlock(d_model, n_heads, cond_dim)

    def forward(self, x, cond):
        x = self.up(x)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        return x

class DownStack(nn.Module):
    def __init__(self, d_model_list, n_heads_list, cond_dim):
        super().__init__()

        blocks = []
        for i in range(len(d_model_list) - 1):
            blocks.append(
                DownStage(
                    d_model_list[i],
                    n_heads_list[i],
                    d_model_list[i+1],
                    cond_dim
                )
            )
        self.down = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.down:
            x = block(x, cond)
        return x

class UpStack(nn.Module):
    def __init__(self, d_model_list, n_heads_list, cond_dim):
        super().__init__()

        blocks = []
        for i in range(len(d_model_list) - 1):
            blocks.append(
                UpStage(
                    d_model_list[i+1],
                    n_heads_list[i+1],
                    d_model_list[i],
                    cond_dim
                )
            )
        self.up = nn.ModuleList(blocks)

    def forward(self, x, cond):
        for block in self.up:
            x = block(x, cond)
        return x

class HDiT(nn.Module):
    def __init__(self, config):
        super().__init__()

        mid_config = deepcopy(config)
        mid_config.rope_impl = "image+latent"
        mid_config.sample_size = [
            config.sample_size[0] // 8,
            config.sample_size[1] // 8
        ]

        widths = [
            128,256,384,1024
        ]
        n_heads = [
            2,4,6,16
        ]

        mid_config.d_model = widths[-1]
        mid_config.n_heads = n_heads[-1]

        self.down = DownStack(widths, n_heads, 2*config.d_model)
        self.up = UpStack(widths[::-1], n_heads[::-1], 2*config.d_model)
        self.middle = DiT(mid_config)

        self.proj_z_middle = nn.Linear(
            config.d_model,
            widths[-1]
        )
        self.proj_cond_middle = nn.Linear(
            config.d_model,
            widths[-1]
        )
        
        self.latent_size = config.latent_size
        self.n_latent = self.latent_size ** 2

        self.n_p_x = config.sample_size[1] // config.patch_size[1]
        self.n_p_y = config.sample_size[0] // config.patch_size[0]

    def forward(self, x, cond):
        b,_,d = x.shape

        # We assume the x is [x,z] per diffdec and [b,n,d]
        x, z = x[:,:-self.n_latent], x[:,-self.n_latent:]

        # cond hdit will be image, channel concat for z || t
        cond_hdit = cond.unsqueeze(1).unsqueeze(1) # [b,1,1,d]
        z_img = z.view(b,self.latent_size,self.latent_size,d)
        cond_hdit = cond_hdit.repeat(1,self.latent_size,self.latent_size,1)
        cond_hdit = torch.cat([cond_hdit,z_img], dim = -1)

        # the nattn blocks need x as an image
        x = x.view(b,self.n_p_y,self.n_p_x,d).contiguous()
        x = self.down(x, cond_hdit)

        # Prepare for middle by flattening, and projecting z/cond
        _,n_p_y,n_p_x,d_middle = x.shape
        x = x.view(b,n_p_y * n_p_x, d_middle)
        z = self.proj_z_middle(z.clone())
        cond = self.proj_cond_middle(cond.clone())
        x_mid_in = torch.cat([x,z], dim = 1) # [b,n+d,d]
        x = self.middle(x_mid_in, cond)[:,:-self.n_latent]

        # Reshape to prepare for up
        x = x.view(b, n_p_y, n_p_x, d_middle)
        x = self.up(x, cond_hdit)

        # Flatten for output shape match
        b,n_p_y,n_p_x,d = x.shape
        x = x.view(b, n_p_y * n_p_x, d)

        return x

if __name__ == "__main__":
    from ..configs import Config
    from .dit import DiT 

    cfg = Config.from_yaml("configs/cod_yt_v2/causal_diffdec.yml")
    model = HDiT(cfg.model).cuda().bfloat16()

    # After flattening 
    n_p_y = cfg.model.sample_size[0] // cfg.model.patch_size[0]
    n_p_x = cfg.model.sample_size[1] // cfg.model.patch_size[1]

    xz = torch.randn(1, n_p_y * n_p_x + cfg.model.latent_size ** 2, cfg.model.d_model).cuda().bfloat16()
    cond = torch.randn(1, cfg.model.d_model).cuda().bfloat16()

    print(n_p_y * n_p_x)
    print(xz.shape)
    with torch.no_grad():
        out = model(xz, cond)
    print(out.shape)
    exit()