from .schedulers import get_sd3_euler
import torch
from tqdm import tqdm
import einops as eo
from ..nn.attn import get_attn_mask
from ..utils import int_to_tuple

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

    max_q = model.config.latent_size ** 2 * model.config.n_frames
    max_q += model.n_p_y * model.n_p_x * model.config.n_frames
    max_kv = max_q

    attn_mask = get_attn_mask(
        model.config,
        batch_size = 1,
        device = z.device,
        max_q_len = max_q,
        max_kv_len = max_kv,
        kernel_size = int_to_tuple(getattr(model.config, "kernel", None)) 
    )

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

        context_z = z[:,frame_idx:frame_idx+window_size].clone().contiguous()
        noisy_video = noisy_video.contiguous()
        ts = ts.contiguous()
        null_emb = null_emb.contiguous()

        for i in range(steps):

            pred = model(noisy_video, context_z, ts, attn_mask = attn_mask)
            pred_uncond = model(noisy_video, null_emb, ts, attn_mask = attn_mask)
            pred = pred_uncond + cfg_scale * (pred - pred_uncond)
            noisy_video[:,-1] = noisy_video[:,-1] - dt_list[i] * pred[:,-1]
            ts[:,-1] = ts[:,-1] - dt_list[i]
        
        generated_frames.append(noisy_video[:,-1])
        context = torch.cat([context[:,1:], noisy_video[:,-1:]], dim = 1)
    
    x = torch.stack(generated_frames, dim = 1)
    if decoder is not None:
        x = x.bfloat16() * scaling_factor
        b,n = x.shape[:2]
        # For loop over n, passing b c h w each time and stacking results
        frames = []
        for i in range(n):
            xi = x[:, i]  # [b, c, h, w]
            xi_dec = decoder(xi)
            frames.append(xi_dec)
        x = torch.stack(frames, dim=1)  # [b, n, c, h, w]
        x = x.clamp(-1,1)
    return x

if __name__ == "__main__":
    from ..models.causal_diffdec import CausalDiffusionDecoder
    from ..configs import Config
    import torch

    cfg_path = "configs/waypoint_1/wp1_caus_diffdec_360p.yml"
    cfg = Config.from_yaml(cfg_path).model
    model = CausalDiffusionDecoder(cfg).core
    model = model.cuda().bfloat16()
    #model = torch.compile(model)

    N_FRAMES = 128
    BATCH_SIZE = 1

    with torch.inference_mode():
        gt_frames = torch.randn(BATCH_SIZE,N_FRAMES,3,720,1280).cuda().bfloat16()
        z = torch.randn(BATCH_SIZE,N_FRAMES,128,8,8).cuda().bfloat16()

        samples = causal_diffdec_sample(model, gt_frames, z, 4, decoder = None)
    print(samples.shape)