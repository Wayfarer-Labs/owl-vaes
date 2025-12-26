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

def is_landscape(config):
    sample_size = config.sample_size
    if isinstance(sample_size, int):
        return False
    sample_size = (int(sample_size[0]), int(sample_size[1]))
    if sample_size[0] < sample_size[1]: # width > height
        return True
    return False

def batch(x):
    b,t,c,h,w = x.shape
    x = x.reshape(b*t, c, h, w)
    return x

def unbatch(x, b):
    _, c, h, w = x.shape
    x = x.reshape(b, -1, c, h, w)
    return x

def latent_ln(z, eps=1e-6):
    # z: [b, c, h, w]
    mean = z.mean(dim=(1, 2, 3), keepdim=True)
    var  = z.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (z - mean) / torch.sqrt(var + eps)

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.config = config
        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max
        self.skip_logvar = getattr(config, "skip_logvar", False)

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
            residuals.append(SpaceToChannel(ch, next_ch))
            attn_blocks.append(CausalFrameAttn(next_ch, attn_config))

            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.avg_factor = ch // config.latent_channels
        
        self.conv_out = ChannelAverage(ch, config.latent_channels)
        
        self.conv_out_logvar = WeightNormConv2d(ch, 1, 3, 1, 1) if not self.skip_logvar else None

        # TODO, this is sloppy
        self.down = nn.ModuleList([
            nn.Identity(),
            TemporalDownsample(min(ch_0 * 4, ch_max)),
            TemporalDownsample(min(ch_0 * 8, ch_max)),
            nn.Identity(),
        ])
        self.normalize_mu = getattr(config, 'normalize_mu', False)

    def forward(self, x, kv_cache = None, attn_mask = None):
        if attn_mask is None:
            attn_mask = get_frame_causal_attn_mask(
                self.config,
                batch_size = x.shape[0],
                device = x.device,
                max_q_len = x.shape[1],
                max_kv_len = x.shape[1],
                kernel_size = getattr(self.config, "kernel", None)
            )

        b = x.shape[0]
        x = batch(x)
        x = self.conv_in(x)
        for i, (block, shortcut, attn, down) in enumerate(zip(self.blocks, self.residuals, self.attn_blocks, self.down)):
            res = shortcut(x)
            x = block(x) + res
            x = unbatch(x, b)
            x = attn(x, kv_cache, attn_mask)
            x = down(x)
            x = batch(x)

        mu = self.conv_out(x)
        if self.normalize_mu:
            mu = latent_ln(mu)
        mu = unbatch(mu, b)

        if not self.training:
            return mu
        else:
            if self.skip_logvar:
                return mu
            logvar = self.conv_out_logvar(x)
            logvar = logvar.repeat(1, self.latent_channels, 1, 1)
            logvar = unbatch(logvar, b)
            return mu, logvar

class Decoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.config = config
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

        for block_count in blocks_per_stage:
            next_ch = min(ch*2, ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace(next_ch, ch))
            attn_blocks.append(CausalFrameAttn(next_ch, attn_config))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))
        self.attn_blocks = nn.ModuleList(list(reversed(attn_blocks)))

        self.conv_out = SquareToLandscape(ch_0, config.channels) if self.is_landscape else WeightNormConv2d(ch_0, config.channels, 3, 1, 1)
        self.act_out = nn.SiLU()

        self.up = nn.ModuleList([
            nn.Identity(),
            TemporalUpsample(min(ch_0 * 8, ch_max)),
            TemporalUpsample(min(ch_0 * 4, ch_max)),
            nn.Identity(),
        ])

    def forward(self, x, kv_cache = None, attn_mask = None):
        if attn_mask is None:
            attn_mask = get_frame_causal_attn_mask(
                self.config,
                batch_size = x.shape[0],
                device = x.device,
                max_q_len = x.shape[1],
                max_kv_len = x.shape[1],
                kernel_size = getattr(self.config, "kernel", None)
            )

        b = x.shape[0]
        x = batch(x)
        x = self.conv_in(x)

        for i, (block, shortcut, attn, up) in enumerate(zip(self.blocks, self.residuals, self.attn_blocks, self.up)):
            x = unbatch(x, b)
            x = up(x)
            x = attn(x, kv_cache, attn_mask)
            x = batch(x)
            res = shortcut(x)
            x = block(x) + res

        x = self.act_out(x)
        x = self.conv_out(x)
        x = unbatch(x, b)
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
        return rec, mu, logvar

def test_video_dcae():
    from dataclasses import dataclass
    @dataclass
    class VideoDCAEConfig:
        sample_size = (360, 640)
        channels = 3
        latent_size = 16
        latent_channels = 16
        ch_0 = 256
        ch_max = 2048
        d_model = 768
        n_heads = 12
        encoder_blocks_per_stage = [4, 4, 4, 8]
        decoder_blocks_per_stage = [4, 4, 4, 8]
        tokens_per_frame = 256
        rope_impl = "video"
        n_frames = 16
    
    config = VideoDCAEConfig()
    model = VideoDCAE(config)
    model = model.cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, config.n_frames, config.channels, config.sample_size[0], config.sample_size[1]).cuda().bfloat16()
        mu, logvar = model.encoder(x)
        print(mu.shape)
        print(logvar.shape)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu
        rec = model.decoder(z)
        print(rec.shape)

if __name__ == "__main__":
    test_video_dcae()
    

