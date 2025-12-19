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
from ..nn.video_ae import (
    get_frame_causal_attn_mask,
    CausalFrameAttn,
    TemporalDownsample,
    TemporalUpsample,
)

from ..nn.resnet import WeightNormConv2d
from torch.utils.checkpoint import checkpoint
import torch.distributions as dist
from copy import deepcopy

def is_landscape(config : 'ResNetConfig'):
    sample_size = config.sample_size
    if isinstance(sample_size, int):
        return False
    sample_size = (int(sample_size[0]), int(sample_size[1]))
    if sample_size[0] < sample_size[1]: # width > height
        return True
    return False

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        self.skip_logvar = getattr(config, "skip_logvar", False)
        self.skip_residuals = getattr(config, "skip_residuals", False)

        self.latent_channels = config.latent_channels
        self.is_landscape = is_landscape(config)
        self.conv_in = LandscapeToSquare(config.channels, ch_0) if self.is_landscape else WeightNormConv2d(config.channels, ch_0, 3, 1, 1)

        attn_config = deepcopy(config)
        attn_config.sample_size = [int(attn_config.tokens_per_frame**0.5), int(attn_config.tokens_per_frame**0.5)]
        attn_config.patch_size = 1

        blocks = []
        residuals = []
        attn_blocks = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in blocks_per_stage:
            next_ch = min(ch*2, ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count, total_blocks))
            residuals.append(SpaceToChannel(ch, next_ch)) if not self.skip_residuals else None
            attn_blocks.append(CausalFrameAttn(next_ch, attn_config))

            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals) if not self.skip_residuals else None
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.avg_factor = ch // config.latent_channels
        
        self.conv_out = ChannelAverage(ch, config.latent_channels)
        
        self.conv_out_logvar = WeightNormConv2d(ch, 1, 3, 1, 1) if not self.skip_logvar else None

        # TODO, this is sloppy
        self.down = nn.ModuleList([
            nn.Identity(),
            TemporalDownsample(ch_0 * 4),
            TemporalDownsample(ch_0 * 8),
            nn.Identity(),
        ])

    @torch.no_grad()
    def sample(self, x):
        mu, logvar = self.forward(x)
        return mu + (logvar/2).exp() * torch.randn_like(mu)

    def batch(self, x):
        b,t,c,h,w = x.shape
        x = x.reshape(b*t, c, h, w)
        return x
    
    def unbatch(self, x, b):
        x = x.reshape(b, -1, c, h, w)
        return x

    def forward(self, x, kv_cache = None, attn_mask = None):
        if attn_mask is None:
            attn_mask = get_frame_causal_attn_mask(
                self.config,
                batch_size = x.shape[0],
                device = x.device,
                max_q_len = x.shape[1],
                max_kv_len = x.shape[1],
                kernel_size = int_to_tuple(getattr(self.config, "kernel", None))
            )

        b = x.shape[0]
        x = self.batch(x)
        x = self.conv_in(x)
        for i, (block, shortcut, attn, down) in enumerate(zip(self.blocks, self.residuals, self.attn_blocks, self.down)):
            res = shortcut(x)
            x = block(x) + res
            x = self.unbatch(x, b)
            x = attn(x, kv_cache, attn_mask)
            x = down(x)
            x = self.batch(x)

        mu = self.conv_out(x)
        mu = self.unbatch(mu, b)

        if not self.training:
            return mu
        else:
            if self.skip_logvar:
                return mu
            logvar = self.conv_out_logvar(x)
            logvar = logvar.repeat(1, self.latent_channels, 1, 1)
            logvar = self.unbatch(logvar, b)
            return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.config = config
        self.skip_residuals = getattr(config, "skip_residuals", False)
        self.is_landscape = is_landscape(config)

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.rep_factor = ch_max // config.latent_channels
        self.conv_in = ChannelDuplication(config.latent_channels, ch_max)

        attn_config = deepcopy(config)
        attn_config.sample_size = [int(attn_config.tokens_per_frame**0.5), int(attn_config.tokens_per_frame**0.5)]
        attn_config.patch_size = 1

        blocks = []
        residuals = []
        attn_blocks = []
        ch = ch_0

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = len(blocks_per_stage)

        for block_count in reversed(blocks_per_stage):
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch)) if not self.skip_residuals else None
            attn_blocks.append(CausalFrameAttn(next_ch, attn_config))

            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        self.conv_out = SquareToLandscape(ch_0, config.channels) if self.is_landscape else WeightNormConv2d(ch_0, config.channels, 3, 1, 1)
        self.act_out = nn.SiLU()

        self.up = nn.ModuleList([
            nn.Identity(),
            TemporalUpsample(ch_0 * 8),
            TemporalUpsample(ch_0 * 4),
            nn.Identity(),
        ])

    def batch(self, x):
        b,t,c,h,w = x.shape
        x = x.reshape(b*t, c, h, w)
        return x
    
    def unbatch(self, x, b):
        x = x.reshape(b, -1, c, h, w)
        return x

    def forward(self, x):
        b = x.shape[0]
        x = self.batch(x)
        x = self.conv_in(x)

        for i, (block, shortcut, attn, up) in enumerate(zip(self.blocks, self.residuals, self.attn_blocks, self.up)):
            res = shortcut(x)
            x = block(x) + res
            x = self.unbatch(x, b)
            x = attn(x, kv_cache, attn_mask)
            x = up(x)
            x = self.batch(x)

        x = self.act_out(x)
        x = self.conv_out(x)
        x = self.unbatch(x, b)
        return x

class VideoDCAE(nn.Module):
    """
    Video DCAE based autoencoder
    """
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu

        rec = self.decoder(z)
        return rec

def test_video_dcae():
    config = ResNetConfig(
        sample_size = [360, 640],
        channels = 3,
        latent_size = 16,
        latent_channels = 16,
        ch_0 = 256,
        ch_max = 2048,
        d_model = 768,
        n_heads = 12,
        encoder_blocks_per_stage = [4, 4, 4, 8],
        decoder_blocks_per_stage = [4, 4, 4, 8],
        tokens_per_frame = 256,
        rope_impl = "video"
    )
    model = VideoDCAE(config)
    model = model.cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 8, 3, 360, 640).cuda().bfloat16()
        mu, logvar = model.encoder(x)
        print(mu.shape)
        print(logvar.shape)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu
        rec = model.decoder(z)
        print(rec.shape)
    

