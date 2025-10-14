"""
Similar to local_cod_features.py but for OWLC data.
Assumes a very specific directory structure in order to minimize ls calls.
"""

import os
import glob
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from math import floor
import json

def safe_normalize(tensor):
    # tensor could be [-1,1], [0,1], [0,255]
    if tensor.float().max() > 1.1: # [0,255]
        return tensor.float().div_(127.5).sub_(1.0)
    elif tensor.float().min() < -0.01: # [-1,1]
        return tensor.float()
    else: # [0,1]
        return tensor.float().mul_(2.0).sub_(1.0)

def augment_data(samples, target_size=(256,256)):
    # both are [c,h,w]
    # augments will be: horizontal flip and 
    # random resize crop to target_size

    if random.random() < 0.5:
        samples = torch.flip(samples, dims=[2])  # flip width

    if random.random() < 0.5:
        # Get current dimensions
        _, h, w = samples.shape

        # Random crop to a smaller size (randomly choose between 0.7 and 1.0 of original size)
        crop_scale = random.uniform(0.7, 1.0)
        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)

        # Random crop coordinates
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        # Apply crop
        samples = samples[:, top:top+crop_h, left:left+crop_w]

        # Resize to target size
        samples = F.interpolate(samples.unsqueeze(0), target_size, mode='bilinear').squeeze(0)
    else:
        samples = F.interpolate(samples.unsqueeze(0), target_size, mode='bilinear').squeeze(0)

    return samples

    
class OwlControlDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        target_size=(360, 640),
        assumed_chunk_size=2000
    ):
        super().__init__()
        # Allow root_dir to be a list of strings or a single string
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = list(root_dir)

        self.all_vid_dirs = []

        """
        We assume a directory structure so that every dir in root_dir (if it's a list)
        itself contains many dirs that have 1. a vid_info.json file and 2. a splits folder
        that contains _rgb.pt files

        In order to minimize ls calls, we are going to collect all vid_dirs and extrapolate num chunks from vid_info.json
        """

        for root in self.root_dirs:
            vids = [os.path.join(root, vid) for vid in os.listdir(root)]
            self.all_vid_dirs.extend(vids)
        
        self.assumed_chunk_size = assumed_chunk_size
        self.target_size = list(target_size)

    def __iter__(self):
        while True:
            vid_dir = None
            try:
                vid_dir = random.choice(self.all_vid_dirs)
                vid_info = json.load(open(os.path.join(vid_dir, "vid_info.json")))

                dur = vid_info["duration"]
                fps = vid_info["fps"]
                n_chunks = floor(dur * fps / self.assumed_chunk_size)
                chunk_idx = random.randint(0, n_chunks - 1) if n_chunks > 1 else 0
                rgb_path = os.path.join(vid_dir, "splits", f"{chunk_idx:08d}_rgb.pt")
                rgb = torch.load(rgb_path, map_location='cpu', mmap=True, weights_only=False)
                idx = random.randint(0, rgb.shape[0] - 1)
                rgb = safe_normalize(rgb[idx].clone().float().contiguous())
                samples = augment_data(rgb, self.target_size)

                yield samples
            except Exception as e:
                print(f"Error loading {vid_dir}: {e}")
                continue

def get_loader(batch_size, **data_kwargs):
    ds = OwlControlDataset(**data_kwargs)
    return DataLoader(ds, batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True)

if __name__ == "__main__":
    import time
    from owl_vaes.configs import Config

    config_path = "configs/waypoint_1/owlc_rgb.yml"
    data_kwargs = Config.from_yaml(config_path).train.data_kwargs

    batch_size = 32

    loader = get_loader(batch_size, **data_kwargs)
    loader_iter = iter(loader)

    # Preload first 2 batches (not timed)
    for _ in range(2):
        _ = next(loader_iter)

    n_batches = 10
    times = []
    for i in range(n_batches):
        start = time.time()
        batch = next(loader_iter)[:,-1]
        print(batch.max(), batch.min(), batch.mean())
        end = time.time()
        times.append(end - start)
        if isinstance(batch, tuple):
            print("Batch shapes:", [b.shape for b in batch])
        else:
            print("Batch shape:", batch.shape)
        print(f"Batch {i+1} loaded in {times[-1]:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"Average batch load time: {avg_time:.4f} seconds over {n_batches} batches")
