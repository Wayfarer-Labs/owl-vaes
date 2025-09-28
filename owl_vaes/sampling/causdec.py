from tqdm import tqdm
import torch

class CausDecSampler:
    """
    Sampler for Causal DCAE decoder
    """
    @torch.no_grad()
    def __call__(self, decoder, latents, window_size : int):
        # Decoder is latents -> videos 
        # [b,n,c,h,w]

        assert latents.shape[1] >= window_size

        recs = []
        
        # First window, generate all frames
        rec = decoder(latents[:,:window_size], ignore_nonterminal_frames = False)
        recs.append(rec)

        for i in tqdm(range(1, latents.shape[1] - window_size + 1), desc = "Generating video..."):
            rec = decoder(latents[:,i:i+window_size], ignore_nonterminal_frames = True)
            recs.append(rec)

        return torch.cat(recs, dim=1)


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