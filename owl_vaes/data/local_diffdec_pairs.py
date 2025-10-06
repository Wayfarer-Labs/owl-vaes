"""
Dataset with local pairs of latents and images

Assumes a dataset structure where we have folders upon folders, with the lowest level folders containing files like:
00000000_rgb.pt
00000000_{suffix}.pt (default suffix is 'depthlatent')

Example structure:
- dataset/
  - 00000000/
    - 00000000_rgb.pt
    - 00000000_depthlatent.pt
  - 00000001/
    - 00000001_rgb.pt
    - 00000001_depthlatent.pt

*_rgb.pt is video slice ala [n,c,h,w] RGB uint8
*_{suffix}.pt is a latent [n,c,h,w] bf16, that corresponds to the RGB frames
"""

import os
import glob
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F

class LocalLatentDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        window_size=1,
        target_size=None,
        mix_ratios=None,
    ):
        super().__init__()
        # Allow root_dir to be a list of strings or a single string
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = list(root_dir)
        self.window_size = window_size
        self.target_size = target_size

        # Handle mix_ratios
        self.mix_ratios = mix_ratios
        if self.mix_ratios is not None:
            self.mix_ratios = [float(x) for x in self.mix_ratios]
            if not isinstance(self.mix_ratios, (list, tuple)):
                raise ValueError("mix_ratios must be a list or tuple of floats")
            if len(self.mix_ratios) != len(self.root_dirs):
                raise ValueError("mix_ratios must be the same length as root_dirs")
            if not abs(sum(self.mix_ratios) - 1.0) < 1e-6:
                raise ValueError("mix_ratios must sum to 1.0")

        # For each root_dir, collect rgb files
        self.rgb_files_per_dir = []
        self.all_rgb_files = []
        for root in self.root_dirs:
            pattern = os.path.join(root, "**", "*_rgb.pt")
            rgb_files = glob.glob(pattern, recursive=True)
            if not rgb_files:
                print(f"Warning: No rgb files found in {root}")
                self.rgb_files_per_dir.append([])
                continue
            self.rgb_files_per_dir.append(rgb_files)
            self.all_rgb_files.extend(rgb_files)

        if self.mix_ratios is not None:
            total_valid = sum(len(rf) for rf in self.rgb_files_per_dir)
            if total_valid == 0:
                raise ValueError(f"No rgb files found in any root_dir")
            for i, rf in enumerate(self.rgb_files_per_dir):
                print(f"Found {len(rf)} rgb files in {self.root_dirs[i]}")
        else:
            if not self.all_rgb_files:
                raise ValueError(f"No rgb files found in {self.root_dirs}")
            print(f"Found {len(self.all_rgb_files)} rgb files in {self.root_dirs}")

    def __iter__(self):
        while True:
            try:
                # Select rgb_path according to mix_ratios if provided
                if self.mix_ratios is not None:
                    dir_idx = random.choices(range(len(self.rgb_files_per_dir)), weights=self.mix_ratios, k=1)[0]
                    rgb_files = self.rgb_files_per_dir[dir_idx]
                    if not rgb_files:
                        continue  # skip empty dataset
                    rgb_path = random.choice(rgb_files)
                else:
                    rgb_path = random.choice(self.all_rgb_files)

                rgb_tensor = torch.load(rgb_path, map_location='cpu', mmap=True, weights_only=False)
                n = rgb_tensor.shape[0]
                w = self.window_size

                if w == 1:
                    idx = random.randint(0, n - 1)
                    rgb_sample = rgb_tensor[idx]
                else:
                    if n < w:
                        raise ValueError(f"Not enough frames in {rgb_path} to sample a window of size {w}")
                    start_idx = random.randint(0, n - w)
                    rgb_sample = rgb_tensor[start_idx:start_idx + w]

                yield rgb_sample
            except Exception as e:
                print(f"Error loading {rgb_path}: {e}")
                continue

def collate_fn_rgb(batch):
    # batch is a list of rgb tensors, each of shape [c,h,w] or [w,c,h,w]
    # If window_size==1: [b, c, h, w]
    # If window_size>1: [b, w, c, h, w]
    if isinstance(batch[0], torch.Tensor) and batch[0].ndim == 3:
        # [b, c, h, w]
        batch_rgb = torch.stack(batch).to(torch.bfloat16)
        batch_rgb = (batch_rgb / 127.5) - 1.0  # [0,255] -> [-1,1]
    else:
        # [b, w, c, h, w]
        batch_rgb = torch.stack(batch).to(torch.bfloat16)
        batch_rgb = (batch_rgb / 127.5) - 1.0  # [0,255] -> [-1,1]
    return batch_rgb

def get_loader(batch_size, **data_kwargs):
    ds = LocalLatentDataset(**data_kwargs)
    return DataLoader(ds, batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True, collate_fn=collate_fn_rgb)

def draw_tensor(tensor, filename="sample.png"):
    """
    Draws a 2x2 grid from a [4, c, h, w] tensor normalized to [-1, 1] and saves as an image.
    """
    import torch
    import numpy as np
    from PIL import Image

    if tensor.ndim != 4 or tensor.shape[0] != 4:
        raise ValueError("Input tensor must have shape [4, c, h, w]")

    # Clamp and denormalize to [0, 255]
    tensor = tensor.detach().cpu()
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor + 1) / 2  # [0,1]
    tensor = tensor * 255
    tensor = tensor.to(torch.uint8)

    # Only use first 3 channels if more (for RGB)
    if tensor.shape[1] > 3:
        tensor = tensor[:, :3, :, :]

    # Convert to numpy and [h, w, c]
    imgs = [tensor[i].permute(1, 2, 0).numpy() for i in range(4)]

    h, w, c = imgs[0].shape
    grid = np.zeros((2 * h, 2 * w, c), dtype=np.uint8)
    grid[0:h, 0:w] = imgs[0]
    grid[0:h, w:2*w] = imgs[1]
    grid[h:2*h, 0:w] = imgs[2]
    grid[h:2*h, w:2*w] = imgs[3]

    img = Image.fromarray(grid)
    img.save(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    root_dir = "/mnt/data/waypoint_1/data_pt/MKIF_360P"
    batch_size = 32

    loader = get_loader(batch_size, root_dir=root_dir, window_size=4)
    batch_rgb = next(iter(loader))

    draw_tensor(batch_rgb[0])
    exit()

    import time

    n_batches = 22
    times = []
    loader_iter = iter(loader)

    # Preload first 2 batches (not timed)
    for _ in range(2):
        _ = next(loader_iter)

    for i in range(n_batches - 2):
        start = time.time()
        batch_rgb = next(loader_iter)
        end = time.time()
        times.append(end - start)
        print(f"Batch {i+1} loaded in {times[-1]:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"Average batch load time (excluding first 2): {avg_time:.4f} seconds over {len(times)} batches")