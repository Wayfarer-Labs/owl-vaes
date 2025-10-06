from tqdm import tqdm
import torch

class CausDecSampler:
    """
    Sampler for Causal DCAE decoder
    """
    @torch.no_grad()
    def __call__(self, decoder, latents, window_size: int):
        # latents: [B, N, C, H, W]
        B, N, C, H, W = latents.shape
        assert N >= window_size, f"{N=} < {window_size=}"

        # Optional but helps if any layers are mode-sensitive
        was_training = decoder.training
        decoder.eval()

        # First window: full decode
        rec0 = decoder(latents[:, :window_size], ignore_nonterminal_frames=False)  # [B, W, C, H, W]
        out = latents.new_empty((B, N, rec0.shape[2], rec0.shape[3], rec0.shape[4]))
        out[:, :window_size] = rec0

        # Subsequent windows: terminal-only
        for i in range(1, N - window_size + 1):
            rec = decoder(latents[:, i:i+window_size], ignore_nonterminal_frames=True)  # [B, 1, C, H, W]
            out[:, window_size + i - 1] = rec[:, -1]  # or rec[:, 0]

        if was_training:
            decoder.train()
        return out


if __name__ == "__main__":
    from ..configs import Config
    from ..models.causal_dcae import CausalDCAE

    cfg_path = "configs/waypoint_1/gan_v3.yaml"
    cfg = Config.from_yaml(cfg_path).model
    model = CausalDCAE(cfg)
    model = model.cuda().bfloat16()
    decoder = model.decoder
    decoder = torch.compile(decoder)#, mode = 'max-autotune', fullgraph = True, dynamic = False)

    sampler = CausDecSampler()
    with torch.no_grad():
        rec = sampler(decoder, torch.randn(1, 1000, 64, 16, 16).cuda().bfloat16(), window_size = 4)
        print(rec.shape)