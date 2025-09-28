import torch
from torch import nn
import torch.nn.functional as F

from .dcae import Encoder, Decoder, DCAE
from ..nn.attn import TemporalAttn
from ..nn.modulation import FiLM
from ..nn.causdec import CausalTransformerEncoder

class CausalEncoder(Encoder):
    pass # TODO

class CausalDecoder(Decoder):
    def __init__(self, config):
        super().__init__(config)

        f_dim = 384

        self.crt = CausalTransformerEncoder(config.latent_channels, f_dim)
        # conv_to_f applied to [b, c, t, h, w]
        # to force causal, add padding in the temporal dimension on the front
        # need manual pad to do this
        self.conv_to_f = nn.Conv3d(
            config.latent_channels,
            config.latent_channels,
            (3, config.latent_size[0], config.latent_size[1]),
            (1, 1, 1),
            0
        ) # -> [b,c,t,1,1]

        blocks_per_stage = config.decoder_blocks_per_stage
        ch_list = [] # List of channel counts going into each layer

        ch_0 = config.ch_0
        ch_max = config.ch_max

        ch = ch_0

        for block_count in blocks_per_stage:
            ch = min(ch*2, ch_max)
            ch_list.append(ch)
            
        ch_list = list(reversed(ch_list))

        self.film_modules = nn.ModuleList([
            FiLM(f_dim, ch_list[i]) for i in range(len(ch_list))
        ])
    
    def pack(self, x):
        # b c t h w -> (b*t) c h w
        b,c,t,h,w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        return x.contiguous().view(b*t, c, h, w)
    
    def unpack(self, x, b):
        # (b*t) c h w -> b c t h w
        bt,c,h,w = x.shape
        t = bt // b
        return x.contiguous().view(b, c, t, h, w)

    def forward(self, x, ignore_nonterminal_frames = False):
        b = x.shape[0]

        # x is [b,n,c,h,w]
        x = x.permute(0, 2, 1, 3, 4) # -> [b,c,n,h,w]

        x_pad = F.pad(x,(0, 0, 0, 0, 2, 0))
        f = self.conv_to_f(x_pad)# -> [b,c,t, 1, 1]
        f = f.mean((-1,-2)) # -> [b,t,c]
        f = f.permute(0, 2, 1) # -> [b,t,c]
        f = self.crt(f) # -> [b,t,c]

        if ignore_nonterminal_frames:
            f = f[:,-1:]
            x = x[:,:,-1:] 

        x = self.pack(x)
        x = self.conv_in(x)

        if self.use_middle_block:
            x = self.middle_block(x) + x

        if not self.skip_residuals:
            for (block, shortcut, film) in zip(self.blocks, self.residuals, self.film_modules):
                x = self.pack(film(self.unpack(x, b), f))
                x = block(x) + shortcut(x)
        else:
            for (block, film) in zip(self.blocks, self.film_modules):
                x = self.pack(film(self.unpack(x, b), f))
                x = block(x)

        x = self.act_out(x)
        x = self.conv_out(x)

        x = self.unpack(x, b)
        x = x.permute(0, 2, 1, 3, 4) # -> [b,t,c,h,w]

        return x

class CausalDCAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = CausalEncoder(config)
        self.decoder = CausalDecoder(config)

        self.config = config
        self.do_channel_mask = getattr(config, 'do_channel_mask', False)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu

        if self.do_channel_mask and self.training:
            _,chans,_,_ = z.shape
            ch_prime = get_ch_prime(chans)
            z[:, ch_prime:, :, :] = 0

        return self.decoder(z)

if __name__ == "__main__":
    from ..configs import Config

    cfg = Config.from_yaml("configs/waypoint_1/gan_v3.yaml").model
    model = CausalDCAE(cfg)
    model = model.cuda().bfloat16()
    decoder = model.decoder

    with torch.no_grad():
        x = torch.randn(1, 4, 64, 16, 16).cuda().bfloat16()
        rec = decoder(x, ignore_nonterminal_frames = True)
        print(rec.shape)