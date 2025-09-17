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

class LocalLatentDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        window_size=1,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.window_size = window_size

        # Use depth files as anchor
        pattern = os.path.join(root_dir, "**", "*_depth.pt")
        self.anchor_files = glob.glob(pattern, recursive=True)
        if not self.anchor_files:
            raise ValueError(f"No depth files found in {root_dir}")

        # Build list of (rgb, depth) file pairs
        self.pair_files = []
        for depth_path in self.anchor_files:
            rgb_path = depth_path.replace("_depth.pt", "_rgb.pt")
            self.pair_files.append((rgb_path, depth_path))

        print(f"Found {len(self.pair_files)} pairs of RGB and depth files")

    def __iter__(self):
        while True:
            try:
                rgb_path, depth_path = random.choice(self.pair_files)

                rgb_tensor = torch.load(rgb_path, map_location='cpu', mmap=True)
                depth_tensor = torch.load(depth_path, map_location='cpu', mmap=True)
                n = rgb_tensor.shape[0]
                w = self.window_size

                if w == 1:
                    idx = random.randint(0, n - 1)
                    rgb_sample = rgb_tensor[idx]
                    depth_sample = depth_tensor[idx]
                else:
                    if n < w:
                        raise ValueError(f"Not enough frames in {rgb_path} to sample a window of size {w}")
                    start_idx = random.randint(0, n - w)
                    rgb_sample = rgb_tensor[start_idx:start_idx + w]
                    depth_sample = depth_tensor[start_idx:start_idx + w]

                yield (rgb_sample, depth_sample)
            except Exception as e:
                print(f"Error loading {rgb_path} or {depth_path}: {e}")
                continue

def collate_fn_rgb_depth(batch):
    # batch is a list of (rgb, depth) tuples
    rgbs, depths = zip(*batch)
    batch_rgb = torch.stack(rgbs).to(torch.bfloat16)
    batch_rgb = (batch_rgb / 127.5) - 1.0  # [0,255] -> [-1,1]
    batch_depth = torch.stack(depths).to(torch.bfloat16)
    batch_depth = (batch_depth / 127.5) - 1.0  # [0,255] -> [-1,1]
    return batch_rgb, batch_depth

def get_loader(batch_size, **data_kwargs):
    ds = LocalLatentDataset(**data_kwargs)
    return DataLoader(ds, batch_size=batch_size, num_workers = 8, prefetch_factor = 2, pin_memory=True, collate_fn=collate_fn_rgb_depth)

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
    root_dir = "/mnt/data/datasets/cod_yt_latents"
    batch_size = 32

    loader = get_loader(batch_size, root_dir=root_dir, window_size=4)
    batch_rgb, batch_depth = next(iter(loader))

    import time

    n_batches = 22
    times = []
    loader_iter = iter(loader)

    # Preload first 2 batches (not timed)
    for _ in range(2):
        _ = next(loader_iter)

    for i in range(n_batches - 2):
        start = time.time()
        batch_rgb, batch_depth = next(loader_iter)
        end = time.time()
        times.append(end - start)
        print(f"Batch {i+1} loaded in {times[-1]:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"Average batch load time (excluding first 2): {avg_time:.4f} seconds over {len(times)} batches")