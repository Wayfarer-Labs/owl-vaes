from .schedulers import get_sd3_euler
import torch
from tqdm import tqdm
import einops as eo

def zlerp(x, t):
    eps = torch.randn_like(x)
    return x * (1. - t) + t * eps

@torch.no_grad()
def causal_diffdec_sample(
    model, 
    gt_frames, 
    z, 
    steps, 
    decoder=None, 
    scaling_factor = 1.0, 
    progress_bar = True, 
    cfg_scale = 1.0,
    window_size = None,
    prev_noise = 0.15
):
    window_size = model.n_frames

    generated_frames = []
    context = gt_frames[:,:window_size]
    ts_start = torch.ones_like(context[:,:,0,0,0]) * prev_noise
    ts_start[:,-1] = 1.0 # Full noise

    dt_list = get_sd3_euler(steps).to(z.device)
    null_emb = eo.repeat(model.null_emb, 'c h w -> b n c h w', b = len(z), n = window_size)

    frames_to_generate = z.shape[1] - window_size + 1
    for frame_idx in tqdm(range(frames_to_generate), disable = not progress_bar):
        # Context video with noise as last frame
        noisy_video = context.clone()
        noisy_video[:,:-1] = zlerp(noisy_video[:,:-1], prev_noise)
        noisy_video[:,-1] = torch.randn_like(noisy_video[:,-1])
        ts = ts_start.clone()

        context_z = z[:,frame_idx:frame_idx+window_size].clone()

        for i in range(steps):
            pred = model(noisy_video, context_z, ts)
            pred_uncond = model(noisy_video, null_emb, ts)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)
            noisy_video[:,-1] = noisy_video[:,-1] - dt_list[i] * pred[:,:-1]
            ts[:,-1] = ts[:,-1] - dt_list[i]
        
        generated_frames.append(noisy_video[:,-1])
        context = torch.cat([context[:,1:], noisy_video[:,-1]], dim = 1)
    
    x = torch.stack(generated_frames, dim = 1)
    if decoder is not None:
        x = x.bfloat16() * scaling_factor
        b,n = x.shape[:2]
        x = eo.rearrange(x, 'b n c h w -> (b n) c h w')
        x = decoder(x)
        x = eo.rearrange(x, '(b n) c h w -> b n c h w', b = b, n = n)
        x = x.clamp(-1,1)
    return x