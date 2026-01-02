"""
Inspired by https://github.com/madebyollin/seraena/tree/main
"""

import torch
from torch import nn
import torch.nn.functional as F

def ConvLayer(fi, fo, s = 1):
    k = 4 if (s==2) else 3
    return nn.Sequential(
        nn.Conv2d(fi, fo, k, s, 1),
        nn.BatchNorm2d(fo),
        nn.ReLU(inplace=True)
    )

class SeraenaDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch = config.ch_0

        self.blocks = [
            ConvLayer(config.channels + config.latent_channels, ch),
            ConvLayer(ch, ch * 2, 2),
            ConvLayer(ch * 2, ch * 4, 2),
            ConvLayer(ch * 4, ch * 8, 2),
            ConvLayer(ch * 8, ch * 8),
            ConvLayer(ch * 8, ch * 8),
        ]
        self.blocks = nn.ModuleList(self.blocks)

        self.proj = [
            nn.Conv2d(ch, 1, 1, bias=False),
            nn.Conv2d(ch * 2, 1, 1, bias=False),
            nn.Conv2d(ch * 4, 1, 1, bias=False),
            nn.Conv2d(ch * 8, 1, 1, bias=False),
            nn.Conv2d(ch * 8, 1, 1, bias=False),
            nn.Conv2d(ch * 8, 1, 1, bias=False),
        ]
        self.proj = nn.ModuleList(self.proj)

        self.out_hw = [45,80] # 360p / 8

    def forward(self, x, z):
        h = []

        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, z], dim=1)

        for block, proj in zip(self.blocks, self.proj):
            x = block(x)
            
            early = proj(x)
            early = F.adaptive_avg_pool2d(early, self.out_hw)
            h.append(early.clone())

        return torch.cat(h, dim=1)