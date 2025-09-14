"""
Local version of S3CoDDataset for loading CoD features from local tensor files.

Assumes a directory structure with folders containing files like:
00000000_rgb.pt
00000000_depth.pt
00000000_flow.pt

Each *_rgb.pt is a [n,3,h,w] uint8 tensor.
Each *_depth.pt is a [n,1,h,w] or [n,h,w] uint8 tensor.
Each *_flow.pt is a [n,3,h,w] uint8 tensor (optional).

Arguments are similar to S3CoDDataset, but with root_dir instead of bucket/prefix.
"""

import os
import glob
import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

def valid_data_test(tensor):
    # tensor is [n,c,h,w]
    return True

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

    
class LocalCoDDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        rank=0,
        world_size=1,
        include_suffixes=['rgb', 'depth'],
        target_size=(360, 640),
        mix_ratios=None,
    ):
        super().__init__()
        # Allow root_dir to be a list of strings or a single string
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = list(root_dir)
        self.rank = rank
        self.world_size = world_size
        self.include_suffixes = include_suffixes
        self.target_size = tuple(int(x) for x in target_size)

        if mix_ratios is not None:
            mix_ratios = [float(x) for x in mix_ratios]
        if isinstance(root_dir, list) or isinstance(root_dir, tuple):
            self.root_dirs = [str(s) for s in root_dir]

        # Handle mix_ratios
        self.mix_ratios = mix_ratios
        if self.mix_ratios is not None:
            if not isinstance(self.mix_ratios, (list, tuple)):
                raise ValueError("mix_ratios must be a list or tuple of floats")
            if len(self.mix_ratios) != len(self.root_dirs):
                raise ValueError("mix_ratios must be the same length as root_dirs")
            if not abs(sum(self.mix_ratios) - 1.0) < 1e-6:
                raise ValueError("mix_ratios must sum to 1.0")

        # For each root_dir, collect rgb files and build file_tuples
        self.all_file_tuples = []
        self.file_tuples_per_dir = []
        for root in self.root_dirs:
            pattern = os.path.join(root, "**", "*_rgb.pt")
            rgb_files = glob.glob(pattern, recursive=True)
            if not rgb_files:
                print(f"Warning: No rgb files found in {root}")
                self.file_tuples_per_dir.append([])
                continue

            file_tuples = []
            for rgb_path in rgb_files:
                base = rgb_path[:-7]  # remove "_rgb.pt"
                tuple_ = [rgb_path]
                for suffix in self.include_suffixes:
                    if suffix == 'rgb':
                        continue
                    path = base + f"_{suffix}.pt"
                    if not os.path.exists(path):
                        continue
                    tuple_.append(path)
                file_tuples.append(tuple_)  
            self.file_tuples_per_dir.append(file_tuples)
            self.all_file_tuples.extend(file_tuples)

        if self.mix_ratios is not None:
            total_valid = sum(len(ft) for ft in self.file_tuples_per_dir)
            if total_valid == 0:
                raise ValueError(f"No valid file tuples found in any root_dir with include_suffixes={self.include_suffixes}")
            for i, ft in enumerate(self.file_tuples_per_dir):
                print(f"Found {len(ft)} valid file tuples in {self.root_dirs[i]}")
        else:
            if not self.all_file_tuples:
                raise ValueError(f"No valid file tuples found in {self.root_dirs} with include_depth={self.include_depth}, include_flow={self.include_flow}")
            print(f"Found {len(self.all_file_tuples)} valid file tuples in {self.root_dirs}")

    def __iter__(self):
        while True:
            try:
                # Select file_tuple according to mix_ratios if provided
                if self.mix_ratios is not None:
                    # Choose which dataset to sample from
                    dir_idx = random.choices(range(len(self.file_tuples_per_dir)), weights=self.mix_ratios, k=1)[0]
                    file_tuples = self.file_tuples_per_dir[dir_idx]
                    if not file_tuples:
                        continue  # skip empty dataset
                    file_tuple = random.choice(file_tuples)
                else:
                    file_tuple = random.choice(self.all_file_tuples)

                tensors = [torch.load(f, map_location='cpu', mmap=True, weights_only=False) for f in file_tuple]
                idx = random.randint(0, tensors[0].shape[0] - 1)
                samples = [t[idx] for t in tensors]
                samples = torch.cat(samples, dim = 0) # Concatenate along channel dimension assumes rgb || depth || etc.
                samples = augment_data(samples, self.target_size)

                yield samples.float() / 127.5 - 1.0 # Concatenate along channel dimension assumes rgb || depth || etc.
            except Exception as e:
                print(f"Error loading {file_tuple}: {e}")
                continue

def get_loader(batch_size, **data_kwargs):
    ds = LocalCoDDataset(**data_kwargs)
    return DataLoader(ds, batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True)

if __name__ == "__main__":
    import time
    from owl_vaes.configs import Config

    config_path = "configs/waypoint_1/base.yml"
    data_kwargs = Config.from_yaml(config_path).train.data_kwargs
    data_kwargs['root_dir'] = '/mnt/data/datasets/cod_yt_latents'
    data_kwargs['mix_ratios'] = None

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
        batch = next(loader_iter)
        end = time.time()
        times.append(end - start)
        if isinstance(batch, tuple):
            print("Batch shapes:", [b.shape for b in batch])
        else:
            print("Batch shape:", batch.shape)
        print(f"Batch {i+1} loaded in {times[-1]:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"Average batch load time: {avg_time:.4f} seconds over {n_batches} batches")
