import torch
from torch import nn
import torch.nn.functional as F

import pytorch_wavelets as pw
import torch_dct as dct
import einops as eo
from copy import deepcopy

from .res import R3GANDiscriminator
from .patch import PatchDiscriminator

torch.autograd.set_detect_anomaly(True)

class WaveletDecomp(nn.Module):
    def __init__(self, levels = 2):
        super().__init__()

        self.levels = levels
        self.wave = 'haar'

    def forward(self, x):
        # x is [b,c,h,w]
        _,_,h,w = x.shape
        # Create a new transform instance for each forward pass
        xfm = pw.DWTForward(J=self.levels, wave=self.wave)
        xfm = xfm.to(x.device)
        
        # Ensure x is contiguous and detached from previous computations
        x = x.contiguous()
        _, yh = xfm(x.float())
        
        feats = []
        for yh_i in yh:
            # Create new tensor with fresh storage
            f = eo.rearrange(yh_i.contiguous(), 'b x y h w -> b (y x) h w').to(x.dtype)
            f = F.interpolate(f, (h,w), mode = 'bilinear')
            feats.append(f)
        return feats

class DCTDecomp(nn.Module):
    def apply(self, x):
        # x is [c,h,w]
        coefs = torch.stack([dct.dct_2d(x_i.float()).to(x_i.dtype) for x_i in x])
        # [c,h,w] still
        magnitude = torch.log(torch.abs(coefs) + 1)
        magnitude = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1.0e-6)).clamp(0,1)
        magnitude = magnitude * 2 - 1
        return magnitude

    def forward(self, x):
        x = torch.stack([self.apply(x_i) for x_i in x])
        # -> [b,c,h,w]
        return x

class FreqDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        wv_cfg = deepcopy(config)
        wv_cfg.channels = 12

        self.core = R3GANDiscriminator(wv_cfg)
        self.wavelet = WaveletDecomp(levels=1)

    def forward(self, x):
        # Get wavelet decompositions
        wavelet_scores = []
        wavelet_features = []

        wv_out = self.wavelet(x)[0]
        x = torch.cat([x, wv_out], dim = 1)
        score, features = self.core(x, output_hidden_states=True)

        return score, features

from dataclasses import dataclass

@dataclass
class DummyConfig:
    ch_0: int = 64
    ch_max: int = 1024
    channels: int = 3
    sample_size: tuple = (360, 640)
    blocks_per_stage: int = 1

def test_freq_discriminator():
    # Create dummy config and model
    config = DummyConfig()
    disc = FreqDiscriminator(config)
    
    # Create dummy input tensor
    batch_size = 2
    x = torch.randn(batch_size, config.channels, *config.sample_size)
    
    # Run forward pass
    scores, features = disc(x)
    
    # Basic assertions
    assert len(scores) == 3  # 2 wavelet + 1 DCT discriminator
    assert len(features) == 3
    
    # Check output shapes
    for score in scores:
        assert score.shape[0] == batch_size
        assert score.shape[1] == 1  # Single channel output
        
    for feature_list in features:
        assert len(feature_list) > 0
        for feature in feature_list:
            assert feature.shape[0] == batch_size

if __name__ == "__main__":
    test_freq_discriminator()
