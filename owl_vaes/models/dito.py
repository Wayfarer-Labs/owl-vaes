import torch
import torch.nn.functional as F
from torch import nn

from copy import deepcopy
import einops as eo

from .dcae import Encoder
from ..nn.dit import DiT, FinalLayer
from ..nn.embeddings import TimestepEmbedding
from ..nn.attn import get_attn_mask
from ..utils import int_to_tuple

class DiTODecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        config.rope_impl = 'image'
        size = config.sample_size
        patch_size = config.patch_size
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        n_tokens = size[0] // patch_size[0] * size[1] // patch_size[1]

        self.proj_in = nn.Linear(patch_size[0] * patch_size[1] * config.channels, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, patch_size[0] * patch_size[1] * config.channels, bias = False)
        self.ts_embed = TimestepEmbedding(config.d_model)
        self.final = FinalLayer(config, skip_proj = True)

        self.blocks = DiT(config)
        self.config = config

        self.noise_scale = getattr(config, "noise_scale", 1.0)
        self.p_y, self.p_x = patch_size
        self.n_p_y = config.sample_size[0] // self.p_y
        self.n_p_x = config.sample_size[1] // self.p_x

    def forward(self, x, z, ts, attn_mask = None):
        # x is [b,c,h,w]
        # ts is [b,] in [0,1]
    
        if attn_mask is None:
            # z is interpolated and concatenated channel-wise, so no extra tokens
            max_q = self.n_p_y * self.n_p_x
            max_kv = max_q

            attn_mask = get_attn_mask(
                self.config,
                batch_size = x.shape[0],
                device = x.device,
                max_q_len = max_q,
                max_kv_len = max_kv,
                kernel_size = int_to_tuple(getattr(self.config, "kernel", None))
            )

        cond = self.ts_embed(ts)

        # Convert from image format [b,c,h,w] to patches [b,n_patches,patch_size*patch_size*c]
        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        orig_c = x.shape[1]
        x = torch.cat([x, z], dim=1)

        x = eo.rearrange(
            x,
            'b c (n_p_h p_h) (n_p_w p_w) -> b (n_p_h n_p_w) (p_h p_w c)',
            p_h = self.p_y,
            p_w = self.p_x
        )

        x = self.proj_in(x) # -> [b,n,d]
        x = self.blocks(x, cond, attn_mask = attn_mask)
        x = self.final(x, cond)
        x = self.proj_out(x)

        x = eo.rearrange(
            x,
            'b (n_p_h n_p_w) (p_h p_w c) -> b c (n_p_h p_h) (n_p_w p_w)',
            n_p_h = self.n_p_y,
            n_p_w = self.n_p_x,
            p_h = self.p_y,
            p_w = self.p_x
        )

        return x[:,:orig_c]

class DiTo(nn.Module):
    """
    Diffusion Tokenizer with CFG and X0 Prediction
    """
    def __init__(self, config):
        super().__init__()

        config.skip_logvar = True
        config.normalize_mu = True
        config.clamp_mu = False
        self.encoder = Encoder(config)

        proxy_channels = getattr(config, "proxy_channels", config.channels)
        proxy_sample_size = getattr(config, "proxy_sample_size", config.sample_size)

        decoder_config = deepcopy(config)
        decoder_config.channels = proxy_channels + config.latent_channels
        decoder_config.sample_size = proxy_sample_size

        self.decoder = DiTODecoder(decoder_config)

        self.x0_mode = getattr(config, "x0_mode", True)
        self.noise_scale = getattr(config, "noise_scale", 1.0)

    @torch.no_grad()
    def sample_timesteps(self, b, device, dtype):
        # Sample ts in [0, 1] from a U(0,1)
        ts = torch.rand(b, device=device, dtype=dtype)
        return ts

    def forward(self, x, proxy = None):
        z = self.encoder(x)
        z_original = z.clone()

        if proxy is not None:
            x = proxy

        # Noise sync and then timesteps
        with torch.no_grad():
            sync_mask = torch.rand(len(z), device=z.device) < 0.1

            tau = self.sample_timesteps(len(z), z.device, z.dtype)
            tau_signal = torch.ones_like(tau)

            tau = torch.where(
                sync_mask,
                tau,
                tau_signal
            )
            eps_z = torch.randn_like(z)
            tau_exp = tau.view(-1, 1, 1, 1).expand_as(z)

            ts = self.sample_timesteps(len(x), x.device, x.dtype)

            # Sync mask => x should be just as noisy as z
            # In x0 mode, x(0) is noise, x(1) is image
            # In flow mode, x(0) is image, x(1) is noise
            ts = torch.where(
                sync_mask,
                ts,
                ts * tau if self.x0_mode else (1 - ts * tau)
            )
            eps_x = torch.randn_like(x)
            ts_exp = ts.view(-1, 1, 1, 1).expand_as(x)

            if self.x0_mode:
                noisy_x = ts_exp * x + (1. - ts_exp) * eps_x
                den = (1. - ts_exp).clamp(min=0.05)
                target = (x - noisy_x) / den
            else:
                noisy_x = x * (1. - ts_exp) + ts_exp * eps_x
                target = eps_x - x

        if self.x0_mode:
            noisy_z = tau_exp * z + (1. - tau_exp) * eps_z
            pred = self.decoder(noisy_x, noisy_z, ts)
            pred = (pred - noisy_x) / den
        else:
            noisy_z = z * (1. - tau_exp) + tau_exp * eps_z
            pred = self.decoder(noisy_x, noisy_z, ts)
        loss = F.mse_loss(pred, target)

        return loss, z_original
