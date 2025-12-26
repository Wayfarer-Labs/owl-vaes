from .schedulers import get_sd3_euler
import torch
from tqdm import tqdm

def get_sample_fn(name="simple"):
    if name == "simple":
        return flow_sample
    elif name == "causal":
        from .causdiffdec import causal_diffdec_sample
        return causal_diffdec_sample

@torch.no_grad()
def flow_sample(model, dummy, z, steps, decoder=None, scaling_factor = 1.0, progress_bar = True, cfg_scale = 1.0):
    x = torch.randn_like(dummy) * model.noise_scale
    ts = torch.ones(len(z), device = z.device, dtype = z.dtype)

    # Get null embedding from model if CFG is enabled
    use_cfg = cfg_scale != 1.0
    if use_cfg:
        # Get null embedding - expand to batch size
        null_emb = model.null_emb.unsqueeze(0).expand(len(z), -1, -1, -1)

    if steps > 1:
        dt = get_sd3_euler(steps).to(z.device)
        for i in tqdm(range(steps), disable = not progress_bar):
            if use_cfg:
                # Conditional prediction
                pred_cond = model(x, z, ts)
                # Unconditional prediction
                pred_uncond = model(x, null_emb, ts)
                # Apply CFG
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = model(x, z, ts)

            x = x - dt[i] * pred
            ts = ts - dt[i]
    else:
        if use_cfg:
            pred_cond = model(x, z, ts)
            pred_uncond = model(x, null_emb, ts)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = model(x, z, ts)
        x = x - pred

    if decoder is not None:
        x = x.bfloat16() * scaling_factor
        x = decoder(x)
    x = x.clamp(-1,1)

    return x

@torch.no_grad()
def x0_sample(model, dummy, z, steps, decoder=None, scaling_factor = 1.0, progress_bar = True, cfg_scale = 1.0):
    x = torch.randn_like(dummy) * model.noise_scale
    ts = torch.zeros(len(z), device = z.device, dtype = z.dtype)

    def expand(t):
        return t.view(-1, 1, 1, 1).expand_as(x)

    # Get null embedding from model if CFG is enabled
    use_cfg = cfg_scale != 1.0
    if use_cfg:
        # Get null embedding - expand to batch size
        null_emb = model.null_emb.unsqueeze(0).expand(len(z), -1, -1, -1)


    dt = 1. / steps
    for i in tqdm(range(steps), disable = not progress_bar):
        den = expand(1. - ts).clamp(min=0.05)
        if use_cfg:
            pred_x0_cond = model(x, z, ts)
            pred_x0_uncond = model(x, null_emb, ts)

            pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0_cond - pred_x0_uncond)
        else:
            pred_x0 = model(x, z, ts)

        v_pred = (pred_x0 - x) / den
        x = x + dt * v_pred
        ts = ts + dt
    
    if decoder is not None:
        x = x.bfloat16() * scaling_factor
        x = decoder(x)
    x = x.clamp(-1,1)
    return x