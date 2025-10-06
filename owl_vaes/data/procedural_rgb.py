"""
Local dataset with too many files.
Give an input dir that has mp4s
Assuming owl-data-2 data processing,
there is a specific directory structure we shall assume
for the output rgb.pt files. This will make it easier to find them.
"""

import os
import glob
import json
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from math import floor

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

def augment_video(video, target_size=(256,256)):
    # video: [b, c, h, w]
    # augments: horizontal flip and random resize crop to target_size

    if random.random() < 0.5:
        video = torch.flip(video, dims=[3])  # flip width

    if random.random() < 0.5:
        # Get current dimensions
        _, _, h, w = video.shape

        # Random crop to a smaller size (randomly choose between 0.7 and 1.0 of original size)
        crop_scale = random.uniform(0.7, 1.0)
        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)

        # Random crop coordinates
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        # Apply crop
        video = video[:, :, top:top+crop_h, left:left+crop_w]

        # Resize to target size
        video = F.interpolate(video, target_size, mode='bilinear')
    else:
        video = F.interpolate(video, target_size, mode='bilinear')

    return video

def sanitize_filename(filename: str) -> str:
    """Replace leading hyphens and other problematic characters with underscores."""
    # Replace leading hyphens with underscores
    while filename.startswith('-'):
        filename = '_' + filename[1:]
    return filename

def get_vid_info_path_with_output_dirs(root_dir, output_dir, file_type='mp4'):
    """
    Get a list of tuples of (path_to_vid_info, path_to_corresponding_output_dir)
    """
    vid_info_paths_with_output_dirs = []
    # Handle both single path and list of paths
    root_dirs = [root_dir] if isinstance(root_dir, str) else root_dir
    
    from tqdm import tqdm

    pbar = tqdm(desc="Collecting video paths", unit="video", total=None)
    for dir in root_dirs:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(file_type):
                    full_path = os.path.join(root, file)

                    # first get raw fp sanitized
                    fp = os.path.basename(full_path)
                    fp = os.path.splitext(fp)[0]
                    fp = sanitize_filename(fp)

                    # full path starts with root_dir, so for output, swap it
                    output_path = os.path.dirname(full_path.replace(root_dir, output_dir))
                    output_path = os.path.join(output_path, fp)

                    vid_info_path = os.path.join(output_path, "vid_info.json")

                    vid_info_paths_with_output_dirs.append((vid_info_path, output_path))
                    pbar.update(1)
    pbar.close()
    return vid_info_paths_with_output_dirs

class ProceduralRGBDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        output_dir,
        target_size=None,
        mix_ratios=None,
        assumed_chunk_size = 2000,
        window_size = 4,
        stride_options = [1, 2, 3, 4],
    ):
        super().__init__()
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.target_size = [int(x) for x in target_size]
        self.mix_ratios = mix_ratios
        self.assumed_chunk_size = assumed_chunk_size
        self.window_size = window_size
        self.stride_options = stride_options

        self.tuples = get_vid_info_path_with_output_dirs(self.root_dir, self.output_dir)
        self.n_tuples = len(self.tuples)
        random.shuffle(self.tuples)

    def get_item(self):
        idx = random.randint(0, self.n_tuples - 1)
        vid_info_path, output_path = self.tuples[idx]
        vid_info = json.load(open(vid_info_path))

        n_chunks = floor(vid_info["duration"] * vid_info["fps"] / self.assumed_chunk_size)
        chunk_idx = random.randint(0, n_chunks - 1) if n_chunks > 1 else 0

        rgb_path = os.path.join(output_path, "splits", f"{chunk_idx:08d}_rgb.pt")
        rgb = torch.load(rgb_path, map_location='cpu', mmap=True, weights_only=False)

        stride = random.choice(self.stride_options)
        window_size = self.window_size * stride

        start_idx = random.randint(0, len(rgb) - window_size)
        rgb = rgb[start_idx:start_idx + window_size]
        rgb = rgb[::stride].contiguous() # nchw
        rgb = safe_normalize(rgb)
        rgb = augment_video(rgb, self.target_size)

        return rgb

    def __iter__(self):
        while True:
            try:
                yield self.get_item()
            except Exception as e:
                print(f"Error loading: {e}")
                continue
        
def get_loader(batch_size, **data_kwargs):
    ds = ProceduralRGBDataset(**data_kwargs)
    return DataLoader(ds, batch_size=batch_size, num_workers=1, prefetch_factor=1, pin_memory=True)

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    ds = ProceduralRGBDataset(
        root_dir="/mnt/data/waypoint_1/data/MKIF_360P",
        output_dir="/mnt/data/waypoint_1/data_pt/MKIF_360P",
        target_size=(256,256),
        window_size=4
    )
    for i in tqdm(range(10)):
        rgb = ds.get_item()
        print(rgb.shape)