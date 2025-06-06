"""
R3Gan style resnet based VAE
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..nn.resnet import DownBlock, SameBlock

class R3GANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        size = config.sample_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        blocks_per_stage = config.blocks_per_stage

        self.conv_in = nn.Conv2d(3, ch_0, 1, 1, 0, bias = False)

        # Count number of downsampling stages needed to get from sample_size to 4
        total_blocks = len([size // (2**i) for i in range(100) if size // (2**i) >= 4])
        # Add 1 for final layer
        total_blocks += 1
        blocks = []

        ch = ch_0
        while size > 4:
            next_ch = min(ch*2,ch_max)

            blocks.append(DownBlock(ch, next_ch, blocks_per_stage, total_blocks))
            ch = next_ch
            size = size // 2

        blocks.append(SameBlock(ch, ch_max, blocks_per_stage, total_blocks))
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(ch_max, 1, 4, 1, 0)

    def forward(self, x):
        # Forward on single sample
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.final(x)
        x = x.flatten(0)
        return x

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        sample_size:int= 256
        ch_0:int= 32
        ch_max:int= 256
        blocks_per_stage:int= 2

    model = R3GANDiscriminator(DummyConfig()).bfloat16().to(device)
    with torch.no_grad():
        x = torch.randn(1,3,256,256).bfloat16().to(device)
        y = model.forward(x)
        print(y.shape)
