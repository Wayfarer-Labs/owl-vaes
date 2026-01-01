from .schedulers import get_sd3_euler
import torch
from tqdm import tqdm
import einops as eo
from ..nn.attn import get_attn_mask
from ..utils import int_to_tuple

def zlerp(x, t, reverse = True):
    eps = torch.randn_like(x)
    if reverse:
        return x * t + (1. - t) * eps
    else:
        return x * (1. - t) + t * eps

@torch.no_grad()
def causal_x0_sample(
    model,
    dummy, 
    z, 
    steps,
    progress_bar = True,
    prev_signal = 0.85
):
    """
    Sample video from latents given diffusion decoder.
    Note this specific sampler can't take controls, but that's fine.

    :param dummy: Dummy video tensor that matches shape of what we want to generate. We will assume this maxes models max frames
    :param z: Latent tensor of shape [b,n,c,h,w] where is n is length of dummy as well but divided by some factor
    """
    generated_frames = []
    n_chunks = dummy.shape[1] // model.temporal_factor
    chunk_size = model.temporal_factor

    dt = 1. / steps

    # Step 1: Generate the first chunk of frames

    max_q = model.n_p_y * model.n_p_x  * model.n_frames
    max_kv = max_q

    attn_mask = get_attn_mask(
        model.config,
        batch_size = z.shape[0],
        device = z.device,
        max_q_len = max_q,
        max_kv_len = max_kv,
        kernel_size = int_to_tuple(getattr(model.config, "kernel", None)) 
    )

    context = torch.empty(
        dummy.shape[0],
        0,
        dummy.shape[2],
        dummy.shape[3],
        dummy.shape[4],
        device = dummy.device,
        dtype = dummy.dtype,
    )
    ts = torch.empty(
        dummy.shape[0],
        0,
        device = dummy.device,
        dtype = dummy.dtype,
    )

    for chunk_idx in tqdm(range(n_chunks), disable = not progress_bar):
        # Make noise and ts for next chunk, get all relevant z up to now
        noisy = torch.randn_like(dummy[:,:chunk_size])
        noisy_ts = torch.zeros(dummy.shape[0], chunk_size, device = dummy.device, dtype = dummy.dtype)
        z_slice = z[:,:chunk_idx+1].contiguous()

        # concat to get ready
        context = torch.cat([context, noisy], dim = 1)
        ts = torch.cat([ts, noisy_ts], dim = 1)

        # Denoising loop
        for step_idx in range(steps):
            # V Prediction for just the last chunk
            den = (1. - ts).clamp(min=0.05)[:,-chunk_size:] # [b,chunk_size]
            pred_x0 = model(context, z_slice, ts)[:,-chunk_size:]
            v_pred = (pred_x0 - context[:,-chunk_size:]) / den.view(den.shape[0],den.shape[1],1,1,1).expand_as(pred_x0)
            
            context[:,-chunk_size:] = context[:,-chunk_size:] + dt * v_pred
            ts[:,-chunk_size:] = ts[:,-chunk_size:] + dt
        
        # Once denoised, add new frame to generated frames
        # And update context to be partially noised
        generated_frames.append(context[:,-chunk_size:].clone())
        context[:,-chunk_size:] = zlerp(context[:,-chunk_size:], prev_signal)
        ts[:,-chunk_size:] = ts[:,-chunk_size:] * prev_signal

    generated_frames = torch.cat(generated_frames, dim = 1).clamp(-1,1)
    return generated_frames

@torch.no_grad()
def causal_v_sample(
    model,
    dummy, 
    z, 
    steps,
    progress_bar = True,
    prev_noise = 0.15
):
    """
    Sample video from latents given diffusion decoder. Assumes outputs are v predictions.
    """
    """
    Sample video from latents given diffusion decoder.
    Note this specific sampler can't take controls, but that's fine.

    :param dummy: Dummy video tensor that matches shape of what we want to generate. We will assume this maxes models max frames
    :param z: Latent tensor of shape [b,n,c,h,w] where is n is length of dummy as well but divided by some factor
    """
    generated_frames = []
    n_chunks = dummy.shape[1] // model.temporal_factor
    chunk_size = model.temporal_factor

    dt = 1. / steps

    # Step 1: Generate the first chunk of frames

    max_q = model.n_p_y * model.n_p_x  * model.n_frames
    max_kv = max_q

    attn_mask = get_attn_mask(
        model.config,
        batch_size = z.shape[0],
        device = z.device,
        max_q_len = max_q,
        max_kv_len = max_kv,
        kernel_size = int_to_tuple(getattr(model.config, "kernel", None)) 
    )

    context = torch.empty(
        dummy.shape[0],
        0,
        dummy.shape[2],
        dummy.shape[3],
        dummy.shape[4],
        device = dummy.device,
        dtype = dummy.dtype,
    )
    ts = torch.empty(
        dummy.shape[0],
        0,
        device = dummy.device,
        dtype = dummy.dtype,
    )

    for chunk_idx in tqdm(range(n_chunks), disable = not progress_bar):
        # Make noise and ts for next chunk, get all relevant z up to now
        noisy = torch.randn_like(dummy[:,:chunk_size])
        noisy_ts = torch.ones(dummy.shape[0], chunk_size, device = dummy.device, dtype = dummy.dtype)
        z_slice = z[:,:chunk_idx+1].contiguous()

        # concat to get ready
        context = torch.cat([context, noisy], dim = 1)
        ts = torch.cat([ts, noisy_ts], dim = 1)

        # Denoising loop
        for step_idx in range(steps):
            # V Prediction for just the last chunk
            v_pred = model(context, z_slice, ts)[:,-chunk_size:]
            
            context[:,-chunk_size:] = context[:,-chunk_size:] - dt * v_pred
            ts[:,-chunk_size:] = ts[:,-chunk_size:] - dt
        
        # Once denoised, add new frame to generated frames
        # And update context to be partially noised
        generated_frames.append(context[:,-chunk_size:].clone())
        context[:,-chunk_size:] = zlerp(context[:,-chunk_size:], prev_signal)
        ts[:,-chunk_size:] = ts[:,-chunk_size:] * prev_noise

    generated_frames = torch.cat(generated_frames, dim = 1).clamp(-1,1)
    return generated_frames

if __name__ == "__main__":
    from ..models.video_dito import VideoDiTo
    from ..configs import Config
    import torch

    cfg_path = "configs/dito/wp1_video_dito.yml"
    cfg = Config.from_yaml(cfg_path).model
    model = VideoDiTo(cfg)
    encoder = model.encoder
    model = model.decoder
    model = model.cuda().bfloat16()
    encoder = encoder.cuda().bfloat16()
    #model = torch.compile(model)

    rgb = torch.randn(1, 16, 3, 360, 640).cuda().bfloat16()
    with torch.no_grad():
        z = encoder(rgb)
    print(z.shape)
    exit()

    dummy = torch.randn(1, 16, 3, 360, 640).cuda().bfloat16()
    z = torch.randn(1, 4, 16, 32, 32).cuda().bfloat16()

    with torch.inference_mode():
        samples = all_at_once_x0_sample(model, dummy, z, 25)
    print(samples.shape)