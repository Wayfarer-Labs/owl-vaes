import torch.distributed as dist
import wandb
import torch
from torch import Tensor

import numpy as np

class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out).
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.data = {}

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}

        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)

            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced

        self.data = {}
        return final
# ==== IMAGES ====

def to_wandb(x1, x2, gather = False):
    # x1, x2 both is [b,c,h,w]
    x = torch.cat([x1,x2], dim = -1) # side to side
    x = x[:,:3] # Limit to RGB when theres extra channels
    x = x.clamp(-1, 1)

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    x = (x.detach().float().cpu() + 1) * 127.5 # [-1,1] -> [0,255]
    x = x.permute(0,2,3,1).numpy().astype(np.uint8) # [b,c,h,w] -> [b,h,w,c]
    return [wandb.Image(img) for img in x]

def to_wandb_depth(x1, x2, gather = False):
    # Extract depth channel (channel 3) from 4 or 7 channel images
    # x1, x2 both is [b,c,h,w] where c >= 4
    if x1.shape[1] < 4 or x2.shape[1] < 4:
        return []
    
    depth1 = x1[:,3:4] # Keep as single channel
    depth2 = x2[:,3:4]
    
    x = torch.cat([depth1, depth2], dim = -1) # side to side
    x = x.clamp(-1, 1)

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    x = (x.detach().float().cpu() + 1) * 127.5 # [-1,1] -> [0,255]
    x = x.permute(0,2,3,1).numpy().astype(np.uint8) # [b,c,h,w] -> [b,h,w,c]
    # Convert single channel to grayscale images
    x = x.squeeze(-1) if x.shape[-1] == 1 else x
    return [wandb.Image(img, mode='L') for img in x]

def to_wandb_flow(x1, x2, gather = False):
    # Extract optical flow channels (channels 4-6) from 7 channel images
    # x1, x2 both is [b,c,h,w] where c >= 7
    if x1.shape[1] < 7 or x2.shape[1] < 7:
        return []
    
    flow1 = x1[:,4:7] # RGB optical flow
    flow2 = x2[:,4:7]
    
    x = torch.cat([flow1, flow2], dim = -1) # side to side
    x = x.clamp(-1, 1)

    if dist.is_initialized() and gather:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        x = torch.cat(gathered, dim=0)

    x = (x.detach().float().cpu() + 1) * 127.5 # [-1,1] -> [0,255]
    x = x.permute(0,2,3,1).numpy().astype(np.uint8) # [b,c,h,w] -> [b,h,w,c]
    return [wandb.Image(img) for img in x]

# ==== AUDIO ====

def log_audio_to_wandb(
    original: Tensor,
    reconstructed: Tensor,
    sample_rate: int = 44100,
    max_samples: int = 4,
) -> dict[str, wandb.Audio]:
    """
    Log audio samples to Weights & Biases.

    Args:
        original: Original audio tensor (B, C, T)
        reconstructed: Reconstructed audio tensor (B, C, T)
        sample_rate: Audio sample rate
        max_samples: Maximum number of samples to log

    Returns:
        Dictionary for wandb logging
    """
    batch_size = min(original.size(0), max_samples)
    audio_logs = {}

    for i in range(batch_size):
        # Convert to numpy and ensure correct shape for wandb
        orig_audio = original[i].detach().cpu().numpy()  # (C, T)
        rec_audio = reconstructed[i].detach().cpu().numpy()  # (C, T)

        # For stereo audio, mix down to mono for logging
        if orig_audio.shape[0] == 2:
            orig_mono = np.mean(orig_audio, axis=0)
            rec_mono = np.mean(rec_audio, axis=0)
        else:
            orig_mono = orig_audio[0]
            rec_mono = rec_audio[0]

        # Ensure audio is in correct range [-1, 1]
        orig_mono = np.clip(orig_mono, -1.0, 1.0)
        rec_mono = np.clip(rec_mono, -1.0, 1.0)

        audio_logs[f"audio_original_{i}"] = wandb.Audio(
            orig_mono, sample_rate=sample_rate
        )
        audio_logs[f"audio_reconstructed_{i}"] = wandb.Audio(
            rec_mono, sample_rate=sample_rate
        )

    return audio_logs

def to_wandb_video(x):
    # x is [b,n,c,h,w] [-1,1]
    x = x.clamp(-1,1) * 127.5 + 127.5
    x = x.to(torch.uint8).cpu()
    x = x.numpy()
    return [wandb.Video(x_i, fps=10, format="gif") for x_i in x]

def to_wandb_video_sidebyside(original, reconstructed):
    """
    Create side-by-side video comparisons of original and reconstructed videos.
    
    Args:
        original: Original video tensor [b,n,c,h,w] [-1,1]  
        reconstructed: Reconstructed video tensor [b,n,c,h,w] [-1,1]
    
    Returns:
        List of wandb.Video objects showing side-by-side comparisons
    """

    def to_vid(t):
        t = t.clamp(-1,1) * 127.5 + 127.5
        t = t.to(torch.uint8).cpu()
        t = t.numpy()
        return t

    # Concatenate horizontally (side by side)
    combined = torch.cat([original, reconstructed], dim=-1)  # [b,n,c,h,2*w]
    
    # Convert to uint8 range [0,255]
    combined = combined.clamp(-1,1) * 127.5 + 127.5
    combined = combined.to(torch.uint8).cpu()
    combined = combined.numpy()

    rgb = to_vid(original)
    rec = to_vid(reconstructed)

    wandb_dict = {
        'original' : [wandb.Video(rgb_i, fps=10, format="mp4") for rgb_i in rgb],
        'reconstructed' : [wandb.Video(rec_i, fps=10, format="mp4") for rec_i in rec],
    }
    
    return wandb_dict