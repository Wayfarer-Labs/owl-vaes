"""
R3Gan style resnet based VAE
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..nn.resnet import DownBlock, SameBlock, LandscapeToSquare
from torch.nn.utils.parametrizations import weight_norm

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

class R3GANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        size = config.sample_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        blocks_per_stage = config.blocks_per_stage

        self.conv_in = LandscapeToSquare(size, config.channels, ch_0) if is_landscape(size) else weight_norm(nn.Conv2d(config.channels, ch_0, 3, 1, 1))
        size = 512

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
        self.blocks = nn.ModuleList(blocks)

        self.final = weight_norm(nn.Conv2d(ch_max, 1, 4, 1, 0))

    def forward(self, x, output_hidden_states = False):
        # Forward on single sample
        h = []
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
            h.append(x.clone())

        x = self.final(x)
        x = x.flatten(0)
        return (x,h) if output_hidden_states else x
        
def r3gandiscriminator_test():
    from dataclasses import dataclass

    @dataclass
    class DummyConfig:
        sample_size:int= 256
        ch_0:int= 32
        ch_max:int= 256
        blocks_per_stage:int= 2
        
    model = R3GANDiscriminator(DummyConfig()).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1,3,256,256).bfloat16().cuda()
        y = model._forward(x)
        assert y.shape == (1,), f"Expected shape (1,), got {y.shape}"
    print("Test passed!")
    
if __name__ == "__main__":
    r3gandiscriminator_test()

