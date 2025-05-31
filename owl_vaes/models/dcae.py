import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..configs import ResNetConfig
from ..utils.get_device import DeviceManager
from ..nn.normalization import GroupNorm
from ..nn.resnet import DownBlock, SameBlock, UpBlock
from ..nn.sana import ChannelToSpace, SpaceToChannel

device = DeviceManager.get_device()

class Encoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = nn.Conv2d(3, ch_0, 1, 1, 0, bias = False)

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in blocks_per_stage[:-1]:
            next_ch = min(ch*2, ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count, total_blocks))
            residuals.append(SpaceToChannel(ch, next_ch))

            size = size // 2
            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)

        self.final = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks)

        self.avg_factor = ch // config.latent_channels
        self.conv_out = nn.Conv2d(ch, config.latent_channels, 1, 1, 0, bias=False)


    def forward(self, x):
        x = self.conv_in(x)

        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res

        x = self.final(x) + x

        res = x.clone()
        res = eo.reduce(res, 'b (rep c) h w -> b c h w', rep = self.avg_factor, reduction = 'mean')
        x = self.conv_out(x) + res

        return x

class Decoder(nn.Module):
    def __init__(self, config : ResNetConfig):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.rep_factor = ch_max // config.latent_channels
        self.conv_in = nn.Conv2d(config.latent_channels, ch_max, 1, 1, 0, bias = False)

        blocks = []
        residuals = []
        ch = ch_0

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        self.starter = SameBlock(ch_max, ch_max, blocks_per_stage[-1], total_blocks)

        for block_count in reversed(blocks_per_stage[:-1]):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))

            size = size // 2
            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))

        self.conv_out = nn.Conv2d(ch_0, 3, 1, 1, 0, bias=False)
        self.norm_out = GroupNorm(ch_0)
        self.act_out = nn.SiLU()

    def forward(self, x):
        res = x.clone()
        res = eo.repeat(res, 'b c h w -> b (rep c) h w', rep = self.rep_factor)

        x = self.conv_in(x) + res
        x = self.starter(x) + x

        for (block, shortcut) in zip(self.blocks, self.residuals):
            res = shortcut(x)
            x = block(x) + res

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        return x

class DCAE(nn.Module):
    """
    DCAE based autoencoder that takes a ResNetConfig to configure.
    """
    def __init__(self, config : ResNetConfig):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x):
        z = self.encoder(x)
        down_z = F.interpolate(z, scale_factor=.5,mode='bilinear')
        if self.config.noise_decoder_inputs > 0.0:
            dec_input = z + torch.randn_like(z) * self.config.noise_decoder_inputs
            down_dec_input = down_z + torch.randn_like(down_z) * self.config.noise_decoder_inputs
        else:
            dec_input = z.clone()
            down_dec_input = down_z.clone()

        rec = self.decoder(dec_input)
        down_rec = self.decoder(down_dec_input)
        return rec, z, down_rec

if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/1d_diff_exps/teacher_1.yml").model
    model = DCAE(cfg).float().to(device)

    with torch.autocast(device, dtype=torch.bfloat16):
        x = torch.randn(1, 3, 256, 256, device=device, dtype=torch.bfloat16)
        rec, z, _ = model(x)

        print(f'Input shape: {x.shape}, dtype: {x.dtype}')
        print(f'Latent shape: {z.shape}, dtype: {z.dtype}')
        print(f'Output shape: {rec.shape}, dtype: {rec.dtype}')
