import torch
from torch import nn
import torch.nn.functional as F

import pytorch_wavelets as pw
import torch_dct as dct
import einops as eo
from copy import deepcopy

from .res import R3GANDiscriminator

torch.autograd.set_detect_anomaly(True)

class WaveletDecomp(nn.Module):
    def __init__(self, levels = 3):
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
        wv_cfg.channels = 9

        dct_cfg = deepcopy(config)
        dct_cfg.channels = 3

        self.wavelet_discs = nn.ModuleList([
            R3GANDiscriminator(wv_cfg),
            R3GANDiscriminator(wv_cfg),
            R3GANDiscriminator(wv_cfg)
        ])

        self.dct_disc = R3GANDiscriminator(dct_cfg)

        self.wavelet = WaveletDecomp()
        self.dct = DCTDecomp()

    def forward(self, x):
        # Get wavelet decompositions
        wavelet_scores = []
        wavelet_features = []

        wv_out = self.wavelet(x)
        for disc, wv in zip(self.wavelet_discs, wv_out):
            score, features = disc(wv, output_hidden_states=True)
            wavelet_scores.append(score)
            wavelet_features.append(features)

        # Get DCT decomposition scores and features
        dct_out = self.dct(x)
        dct_score, dct_features = self.dct_disc(dct_out, output_hidden_states=True)

        # Combine all scores and features
        scores = wavelet_scores + [dct_score]
        features = wavelet_features + [dct_features]

        return scores, features
