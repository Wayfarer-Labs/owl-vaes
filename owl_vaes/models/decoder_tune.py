import torch
from torch import nn
import torch.nn.functional as F

from ..nn.resnet import (
    DownBlock, SameBlock, UpBlock,
    LandscapeToSquare, SquareToLandscape
)
from ..nn.sana import ChannelToSpace, SpaceToChannel

from torch.nn.utils.parametrizations import weight_norm

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

class Early(nn.Module):
    def __init__(self, config):
        super().__init__()

        size = config.sample_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.rep_factor = ch_max // config.latent_channels
        self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, ch_max, 1, 1, 0))

        blocks = []
        residuals = []
        ch = ch_0*4

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in reversed(blocks_per_stage[2:]):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))

    @torch.no_grad()
    def forward(self, x):
        res = x.clone()
        rep_res = res.repeat(1, self.rep_factor, 1, 1)

        x = self.conv_in(x)
        x = x + rep_res

        for (block, shortcut) in zip(self.blocks, self.residuals):
            x = block(x) + shortcut(x)

        return x

class Late(nn.Module):
    def __init__(self, config):
        super().__init__()

        size = config.sample_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        blocks = []
        residuals = []
        ch = ch_0


        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in reversed(blocks_per_stage[:2]):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))

        self.conv_out = SquareToLandscape(size, ch_0, config.channels) if is_landscape(size) else weight_norm(nn.Conv2d(ch_0, config.channels, 3, 1, 1, bias = False))
        self.act_out = nn.SiLU()

    def forward(self, x):
        for (block, shortcut) in zip(self.blocks, self.residuals):
            x = block(x) + shortcut(x)

        x = self.act_out(x)
        x = self.conv_out(x)

        return x

def create_split_decoder(decoder):
    """
    Returns Early, Late objects
    """
    config = decoder.config
    early = Early(config)
    late = Late(config)

    # Copy early params
    early.conv_in.load_state_dict(decoder.conv_in.state_dict())
    for i in range(len(early.blocks)):
        early.blocks[i].load_state_dict(decoder.blocks[i].state_dict())
        early.residuals[i].load_state_dict(decoder.residuals[i].state_dict())

    # Copy late params
    for i in range(len(late.blocks)):
        late.blocks[i].load_state_dict(decoder.blocks[i+len(early.blocks)].state_dict())
        late.residuals[i].load_state_dict(decoder.residuals[i+len(early.residuals)].state_dict())
    late.conv_out.load_state_dict(decoder.conv_out.state_dict())

    return early, late

def save_state_dict(early, late):
    """
    Meant for output model, saves a singular checkpoint that's
    ready for inference
    """
    pass