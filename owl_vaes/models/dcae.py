import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..nn.normalization import GroupNorm
from ..nn.resnet import (
    DownBlock, SameBlock, UpBlock,
    LandscapeToSquare, SquareToLandscape
)
from ..nn.sana import ChannelToSpace, SpaceToChannel

from torch.nn.utils.parametrizations import weight_norm
from torch.utils.checkpoint import checkpoint

def is_landscape(sample_size):
    h,w = sample_size
    ratio = w/h
    return abs(ratio - 16/9) < 0.01  # Check if ratio is approximately 16:9

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = LandscapeToSquare(size, config.channels, ch_0) if is_landscape(size) else weight_norm(nn.Conv2d(config.channels, ch_0, 3, 1, 1))

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in blocks_per_stage:
            next_ch = min(ch*2, ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count, total_blocks))
            residuals.append(SpaceToChannel(ch, next_ch))

            ch = next_ch

        self.use_middle_block = config.use_middle_block
        self.middle_block = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks) if self.use_middle_block else None

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)

        self.avg_factor = ch // config.latent_channels
        self.conv_out = weight_norm(nn.Conv2d(ch, config.latent_channels, 1, 1, 0))
        self.conv_out_logvar = weight_norm(nn.Conv2d(ch, config.latent_channels, 1, 1, 0))

    @torch.no_grad()
    def sample(self, x):
        mu, logvar = self.forward(x)
        return mu + (logvar/2).exp() * torch.randn_like(mu)

    def forward(self, x):
        x = self.conv_in(x)
        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res

        res = x.clone()
        # Replace einops reduce with reshape + mean
        b, c, h, w = res.shape
        res = res.reshape(b, self.avg_factor, c//self.avg_factor, h, w)
        res = res.mean(dim=1)
        
        if self.use_middle_block:
            x = self.middle_block(x)
        mu = self.conv_out(x) + res

        if not self.training:
            return mu
        else:
            logvar = self.conv_out_logvar(x)

            return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config : 'ResNetConfig', decoder_only = False):
        super().__init__()

        self.config = config

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.rep_factor = ch_max // config.latent_channels
        self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, ch_max, 1, 1, 0))

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        self.use_middle_block = config.use_middle_block
        self.middle_block = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks) if self.use_middle_block else None

        for block_count in reversed(blocks_per_stage):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))

        self.conv_out = SquareToLandscape(size, ch_0, config.channels) if is_landscape(size) else weight_norm(nn.Conv2d(ch_0, config.channels, 3, 1, 1, bias = False))
        self.act_out = nn.SiLU()

        self.decoder_only = decoder_only

    def forward(self, x):
        res = x.clone()
        rep_res = res.repeat(1, self.rep_factor, 1, 1)

        x = self.conv_in(x)
        x = x + rep_res

        if self.use_middle_block:
            x = self.middle_block(x)

        for (block, shortcut) in zip(self.blocks, self.residuals):
            x = block(x) + shortcut(x)

        x = self.act_out(x)
        x = self.conv_out(x)

        return x

class DCAE(nn.Module):
    """
    DCAE based autoencoder that takes a ResNetConfig to configure.
    """
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu
        rec = self.decoder(z)
        return rec, mu, logvar

def dcae_test():
    from ..configs import ResNetConfig

    cfg = ResNetConfig(
        sample_size=256,
        channels=3,
        latent_size=32,
        latent_channels=4,
        noise_decoder_inputs=0.0,
        ch_0=32,
        ch_max=128,
        encoder_blocks_per_stage = [2,2,2,2],
        decoder_blocks_per_stage = [2,2,2,2]
    )

    model = DCAE(cfg).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).bfloat16().cuda()
        rec, z, down_rec = model(x)
        assert rec.shape == (1, 3, 256, 256), f"Expected shape (1,3,256,256), got {rec.shape}"
        assert z.shape == (1, 4, 32, 32), f"Expected shape (1,4,32,32), got {z.shape}"
        assert down_rec.shape == (1, 3, 128, 128), f"Expected shape (1,3,128,128), got {down_rec.shape}"
    print("Test passed!")
    
if __name__ == "__main__":
    dcae_test()
