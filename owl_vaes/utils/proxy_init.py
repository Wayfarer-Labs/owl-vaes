"""
Utility for initializing proxy with ResNet model
Hard-coded for Proxy TiToK + DCAE for now
"""

import torch
from torch import nn

<<<<<<< HEAD
from ..utils.get_device import DeviceManager
=======
>>>>>>> main
from ..configs import Config
from ..models.dcae import Decoder
from ..models.proxy_titok import ProxyTiToKVAE
from . import prefix_filter, versatile_load

class CombinedModule(nn.Module):
    def __init__(self, transformer_cfg, resnet_cfg):
        super().__init__()

        self.transformer = ProxyTiToKVAE(transformer_cfg)
        self.resnet = Decoder(resnet_cfg)

    def load_ckpt(self, t_ckpt_path, r_ckpt_path):
        t_ckpt = versatile_load(t_ckpt_path) # ema checkpoint
        self.transformer.load_state_dict(t_ckpt)

        r_ckpt = versatile_load(r_ckpt_path)
        r_ckpt = prefix_filter(r_ckpt, "decoder.")
        self.resnet.load_state_dict(r_ckpt)

    def encode(self, x):
        return self.transformer.encoder(x)

    def decode(self, z):
        z = self.transformer.decoder(z)
        x = self.resnet(z)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

def load_proxy_model(
    t_cfg_path: str,
    t_ckpt_path: str,
    r_cfg_path: str,
    r_ckpt_path: str
):
    # Load configs
    t_cfg = Config.from_yaml(t_cfg_path).model
    r_cfg = Config.from_yaml(r_cfg_path).model
    
    # Disable mimetic initialization
    t_cfg.mimetic_init = False
    
    # Initialize combined model
    model = CombinedModule(t_cfg, r_cfg).cuda().bfloat16()
    
    # Load checkpoints
    model.load_ckpt(t_ckpt_path, r_ckpt_path)
    
    return model

def proxy_init_test():
    t_cfg_path = "configs/1d_diff_exps/proxy.yml"
    r_cfg_path = "configs/1d_diff_exps/teacher_1.yml"

    t_cfg = Config.from_yaml(t_cfg_path).model
    r_cfg = Config.from_yaml(r_cfg_path).model

    t_cfg.mimetic_init = False
    model = CombinedModule(t_cfg, r_cfg).bfloat16().cuda()

    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).bfloat16().cuda()
        y = model(x)
        assert y.shape == (1, 3, 256, 256), f"Expected shape (1,3,256,256), got {y.shape}"

    print("Test Passed!")
    
if __name__ == "__main__":
    proxy_init_test()
