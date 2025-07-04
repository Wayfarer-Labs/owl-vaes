import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from ..nn.resnet import LandscapeToSquare

def make_conv(ch_in, ch_out, k=3, s=1, p=1):
    return weight_norm(nn.Conv2d(ch_in,ch_out,k,s,p))

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

class PatchDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        ch = config.ch_0
        channels = config.channels
        size = config.sample_size

        self.conv_in = LandscapeToSquare(size, channels, ch) if is_landscape(size) else make_conv(channels, ch)
        self.blocks = nn.ModuleList([
            make_conv(ch, ch*2), # -> 256
            make_conv(ch*2,ch*4), # -> 128
            make_conv(ch*4,ch*8), # -> 64
            make_conv(ch*8,ch*16), # -> 32
            make_conv(ch*16,ch*16) # -> 16
        ])
        self.conv_out = make_conv(ch*16,1)

    def forward(self, x, output_hidden_states = True):
        h = []
        
        x = self.conv_in(x)
        h.append(x.clone())

        for block in self.blocks:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            h.append(x.clone())
            x = F.interpolate(x, scale_factor=.5)

        x = self.conv_out(x)

        if output_hidden_states:
            return x,h
        else:
            return x
        