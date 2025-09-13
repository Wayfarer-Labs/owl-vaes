import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..nn.normalization import GroupNorm
from ..nn.resnet import (
    DownBlock, SameBlock, UpBlock,
    LandscapeToSquare, SquareToLandscape
)
from ..nn.sana import (
    ChannelToSpace, SpaceToChannel,
    ChannelAverage, ChannelDuplication
)

from ..nn.resnet import WeightNormConv2d
from torch.utils.checkpoint import checkpoint
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        self.skip_logvar = getattr(config, "skip_logvar", False)

        self.conv_in = WeightNormConv2d(config.channels, ch_0, 3, 1, 1)

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
        
        self.conv_out = ChannelAverage(ch, config.latent_channels)
        #self.conv_out = weight_norm(nn.Conv2d(ch, config.latent_channels, 3, 1, 1))
        
        self.conv_out_logvar = WeightNormConv2d(ch, config.latent_channels, 3, 1, 1) if not self.skip_logvar else None

    @torch.no_grad()
    def sample(self, x):
        mu, logvar = self.forward(x)
        return mu + (logvar/2).exp() * torch.randn_like(mu)

    def forward(self, x):
        x = self.conv_in(x)
        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res

        if self.use_middle_block:
            x = self.middle_block(x) + x
        mu = self.conv_out(x)

        if not self.training:
            return mu
        else:
            if self.skip_logvar:
                return mu

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
        self.conv_in = ChannelDuplication(config.latent_channels, ch_max)
        #self.conv_in = weight_norm(nn.Conv2d(config.latent_channels, ch_max, 3, 1, 1))

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

        self.conv_out = WeightNormConv2d(ch_0, config.channels, 3, 1, 1)
        self.act_out = nn.SiLU()

        self.decoder_only = decoder_only

    def forward(self, x):
        x = self.conv_in(x)

        if self.use_middle_block:
            x = self.middle_block(x) + x

        for (block, shortcut) in zip(self.blocks, self.residuals):
            x = block(x) + shortcut(x)

        x = self.act_out(x)
        x = self.conv_out(x)

        return x

@torch.no_grad()
def get_ch_prime(total_ch, min_ch=32, step=8):
    """
    Returns an integer sampled from the sequence [min_ch, min_ch+step, ..., total_ch]
    (inclusive of total_ch if it lands on a step), and ensures the value is the same across all devices.
    """
    # Build the valid channel choices
    ch_choices = list(range(min_ch, total_ch + 1, step))
    ch_choices_tensor = torch.tensor(ch_choices, dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
    num_choices = len(ch_choices)

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            idx = torch.randint(0, num_choices, (1,), device=ch_choices_tensor.device)
        else:
            idx = torch.empty(1, dtype=torch.long, device=ch_choices_tensor.device)
        torch.distributed.broadcast(idx, src=0)
        ch_prime = ch_choices_tensor[idx.item()]
        return int(ch_prime.item())
    else:
        idx = torch.randint(0, num_choices, (1,), device=ch_choices_tensor.device)
        ch_prime = ch_choices_tensor[idx.item()]
        return int(ch_prime.item())

class DCAE(nn.Module):
    """
    DCAE based autoencoder that takes a ResNetConfig to configure.
    """
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config
        self.do_channel_mask = getattr(config, 'do_channel_mask', False)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu

        if self.do_channel_mask and self.training:
            _,chans,_,_ = z.shape
            
            ch_prime = get_ch_prime(chans)
            z[:, ch_prime:, :, :] = 0

        rec = self.decoder(z)
        return rec, mu, logvar

def dcae_test():
    from ..configs import Config

    cfg = Config.from_yaml("configs/waypoint_1/base.yml").model

    model = DCAE(cfg).bfloat16().cuda()
    model = torch.compile(model)
    x = torch.randn(1, 4, 512, 512).bfloat16().cuda()
    x.requires_grad_(True)
    rec, z, _ = model(x)
    assert rec.shape == (1, 4, 512, 512), f"Expected shape (1,4,512,512), got {rec.shape}"
    assert z.shape == (1, 64, 16, 16), f"Expected shape (1,64,16,16), got {z.shape}"
    loss = torch.nn.functional.mse_loss(rec, x)
    loss.backward()
    print("Test passed!")
    
if __name__ == "__main__":
    dcae_test()
