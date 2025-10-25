import torch
from torch import nn

import einops as eo

from ..nn.attn import StackedTransformer

def create_attn_mask(b, n_audio_tokens, n_latent_tokens):
    # Assume that in the video
    attn_mask = torch.ones(n_audio_tokens+n_latent_tokens, n_audio_tokens+n_latent_tokens) * -float('inf')
    pass

class CausalAudioEncoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.p = config.patch_size
        self.n_latents = config.latent_samples
        self.latent_starter = nn.Parameter(0.02 * torch.randn(config.d_model))

        self.proj_in = nn.Linear(2 * self.p, config.d_model)

        self.proj_out_longvar = None
        self.skip_logvar = getattr(config, "skip_logvar", False)
        if not self.skip_logvar:
            self.proj_out_logvar = nn.Linear(config.d_model, config.latent_channels)

        self.proj_out_mu = nn.Linear(config.d_model, config.latent_channels)

        self.transformer = StackedTransformer(config)

    def forward(self, x):
        # x is [b,n,2]
        b,n,d = x.shape
        x = eo.rearrange(x, 'b (n p) d -> b n (p d)', p = self.p)
        x = self.proj_in(x)
        b,n,d = x.shape

        z = eo.repeat(self.latent_starter, 'd -> b n d', b = b, n = self.n_latents)

        x = torch.cat([x, z], dim=1)
        x = self.transformer(x)
        z = x[:,n:]

        mu = self.proj_out_mu(z)
        if not self.skip_logvar:
            logvar = self.proj_out_logvar(z)
            return mu, logvar
        else:
            return mu

class CausalAudioDecoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.p = config.patch_size
        self.n_samples = config.n_samples
        self.n_patches = self.n_samples // self.p

        self.proj_in = nn.Linear(config.latent_channels, config.d_model)
        self.transformer = StackedTransformer(config)   
        self.proj_out = nn.Linear(config.d_model, 2 * self.p)

        self.audio = nn.Parameter(0.02 * torch.randn(config.d_model))

    def forward(self, z):
        # z is [b,n,d]
        x = eo.repeat(self.audio, 'd -> b n d', b = z.shape[0], n = self.n_patches)
        z = self.proj_in(z)

        x = torch.cat([x, z], dim=1)
        x = self.transformer(x)
        x = x[:,:self.n_patches]
        x = self.proj_out(x)
        x = eo.rearrange(x, 'b n (p d) -> b (n p) d', p = self.p)
        return x
        

class CausalAudioVAE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.encoder = CausalAudioEncoder(config)
        self.decoder = CausalAudioDecoder(config)

    def forward(self, x):
        # x is [b,n,2]
        mu, logvar = self.encoder(x)
        z = torch.randn_like(mu) * (logvar/2).exp() + mu
        return self.decoder(z), mu, logvar

if __name__ == "__main__":
    from ..configs import Config

    cfg_path = "configs/waypoint_1_audio/basic.yml"
    cfg = Config.from_yaml(cfg_path).model
    model = CausalAudioVAE(cfg)
    model = model.cuda().bfloat16()
    encoder = model.encoder
    decoder = model.decoder

    with torch.no_grad():
        x = torch.randn(1, 88200, 2).cuda().bfloat16()
        rec, mu, logvar = model(x)
        print(rec.shape)
        print(mu.shape)
        print(logvar.shape)