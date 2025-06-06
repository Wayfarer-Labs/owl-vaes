from typing import Tuple

import einops as eo
import torch
from torch import nn
from torch import Tensor

from ..nn.attn import PatchProjIn, PatchProjOut, StackedTransformer
from ..nn.embeddings import LearnedPosEnc

class Encoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        n_tokens = (config.sample_size//config.patch_size)**2
        n_latents = config.latent_size

        self.proj_in = PatchProjIn(config.d_model, config.channels, config.patch_size)
        self.pos_enc = LearnedPosEnc(n_tokens+n_latents, config.d_model)
        self.latent_tokens = nn.Parameter(torch.randn(n_latents,config.d_model)*0.02)

        self.transformer = StackedTransformer(config)
        self.proj_out = nn.Linear(config.d_model, config.latent_channels, bias=False)

    def forward(self, x):
        # x is [b,c,h,w]
        x = self.proj_in(x) # -> [b,p^2,d]

        b,n,d = x.shape
        z = eo.repeat(self.latent_tokens, 'n d -> b n d', b = b)
        n_latents = z.shape[1]
        x = torch.cat([z,x], dim = 1)
        x = self.pos_enc(x)

        x = self.transformer(x)
        z = x[:,:n_latents]
        z = self.proj_out(z)

        return z

class Decoder(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        n_tokens = (config.sample_size//config.patch_size)**2
        n_latents = config.latent_size

        self.proj_out = PatchProjOut(config.sample_size, config.d_model, config.channels, config.patch_size)
        self.pos_enc = LearnedPosEnc(n_tokens+n_latents, config.d_model)
        self.image_tokens = nn.Parameter(torch.randn(n_tokens,config.d_model)*0.02)

        self.transformer = StackedTransformer(config)
        self.proj_in = nn.Linear(config.latent_channels, config.d_model, bias=False)

    def forward(self, z : Tensor):
        # z is [b,l,d_latent]
        z = self.proj_in(z) # [b,l,d]

        b,n_latents,d = z.shape
        x = eo.repeat(self.image_tokens, 'n d -> b n d', b = b)
        x = torch.cat([z,x], dim = 1)
        x = self.pos_enc(x)

        x = self.transformer(x)
        x = x[:,n_latents:]
        x = self.proj_out(x)

        return x

class TiToKVAE(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x is [b,c,h,w]
        z = self.encoder(x) # -> [b,l,d]

        if self.config.noise_decoder_inputs > 0.0:
            dec_input = z + torch.randn_like(z) * self.config.noise_decoder_inputs
        else:
            dec_input = z.clone()

        rec = self.decoder(dec_input)

        return rec, z

def titok_test():
    from ..configs import TransformerConfig

    cfg = TransformerConfig(
        sample_size = 16,
        channels = 32,
        latent_size = 16,
        latent_channels = 128,
        n_layers = 6,
        n_heads = 6,
        d_model = 384,
        patch_size = 1
    )

    model = TiToKVAE(cfg).bfloat16().cuda()
    with torch.no_grad():
        x = torch.randn(1, 32, 16, 16).bfloat16().cuda()
        rec, z = model(x)
        assert rec.shape == (1, 32, 16, 16), f"Expected shape (1,32,16,16), got {rec.shape}"
        assert z.shape == (1, 16, 128), f"Expected shape (1,16,128), got {z.shape}"
    
    print("Test passed!")
    
if __name__ == "__main__":
    titok_test()
