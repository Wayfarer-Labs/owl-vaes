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

def augment_data(rgb_tensor, depth_tensor, target_size):
    # both are [b, c,h,w]
    # augments will be: horizontal flip and 
    # random resize crop to target_size

    if random.random() < 0.5:
        rgb_tensor = rgb_tensor[:, :, :, ::-1]
        depth_tensor = depth_tensor[:, :, :, ::-1]

    if random.random() < 0.5:
        # Get current dimensions
        _, _, h, w = rgb_tensor.shape
        
        # Random crop to a smaller size (randomly choose between 0.7 and 1.0 of original size)
        crop_scale = random.uniform(0.7, 1.0)
        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)
        
        # Random crop coordinates
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))
        
        # Apply crop
        rgb_tensor = rgb_tensor[:, :, top:top+crop_h, left:left+crop_w]
        depth_tensor = depth_tensor[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize to target size
        rgb_tensor = F.interpolate(rgb_tensor, target_size, mode='bilinear')
        depth_tensor = F.interpolate(depth_tensor, target_size, mode='bilinear')
    else:
        rgb_tensor = F.interpolate(rgb_tensor, target_size, mode='bilinear')
        depth_tensor = F.interpolate(depth_tensor, target_size, mode='bilinear')

    return rgb_tensor, depth_tensor

    
class LocalCoDDataset(IterableDataset):
    def __init__(
        self,
        root_dir,
        rank=0,
        world_size=1,
        include_flow=False,
        include_depth=False,
        target_size=(360, 640),
        mix_ratios=None,
        window_size=1,
    ):
        super().__init__()
        # Allow root_dir to be a list of strings or a single string
        if isinstance(root_dir, str):
            self.root_dirs = [root_dir]
        else:
            self.root_dirs = list(root_dir)
        self.rank = rank
        self.world_size = world_size
        self.include_flow = include_flow
        self.include_depth = include_depth
        self.target_size = tuple(int(x) for x in target_size)
        self.window_size = window_size

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
                if self.include_depth:
                    depth_path = base + "_depth.pt"
                    if not os.path.exists(depth_path):
                        continue
                    tuple_.append(depth_path)
                if self.include_flow:
                    flow_path = base + "_flow.pt"
                    if not os.path.exists(flow_path):
                        continue
                    tuple_.append(flow_path)
                file_tuples.append(tuple_)
            if not file_tuples:
                print(f"Warning: No valid file tuples found in {root} with include_depth={self.include_depth}, include_flow={self.include_flow}")
            self.file_tuples_per_dir.append(file_tuples)
            self.all_file_tuples.extend(file_tuples)

        if self.mix_ratios is not None:
            total_valid = sum(len(ft) for ft in self.file_tuples_per_dir)
            if total_valid == 0:
                raise ValueError(f"No valid file tuples found in any root_dir with include_depth={self.include_depth}, include_flow={self.include_flow}")
            for i, ft in enumerate(self.file_tuples_per_dir):
                print(f"Found {len(ft)} valid file tuples in {self.root_dirs[i]}")
        else:
            if not self.all_file_tuples:
                raise ValueError(f"No valid file tuples found in {self.root_dirs} with include_depth={self.include_depth}, include_flow={self.include_flow}")
            print(f"Found {len(self.all_file_tuples)} valid file tuples in {self.root_dirs}")

    def __iter__(self):
        def is_valid_tensor(t):
            return True # This is being excessive for some reason
            # Check for NaNs, Infs, and all zeros
            if not torch.is_tensor(t):
                return False
            if torch.isnan(t).any() or torch.isinf(t).any():
                return False
            if t.numel() == 0:
                return False
            if (t == 0).all():
                return False
            return True

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

                rgb_tensor = torch.load(file_tuple[0], map_location='cpu', mmap=True, weights_only=False)
                n = rgb_tensor.shape[0]
                w = self.window_size

                # Load other tensors if needed
                tensors = [rgb_tensor]
                if self.include_depth:
                    depth_tensor = torch.load(file_tuple[1], map_location='cpu', mmap=True, weights_only=False)
                    tensors.append(depth_tensor)
                if self.include_flow:
                    flow_tensor = torch.load(file_tuple[-1], map_location='cpu', mmap=True, weights_only=False) if self.include_flow else None
                    if self.include_flow:
                        tensors.append(flow_tensor)

                # Sample window
                if w == 1:
                    idx = random.randint(0, n - 1)
                    samples = [t[idx] for t in tensors]
                else:
                    if n < w:
                        raise ValueError(f"Not enough frames in {file_tuple[0]} to sample a window of size {w}")
                    start_idx = random.randint(0, n - w)
                    samples = [t[start_idx:start_idx + w] for t in tensors]

                # Resize if needed
                samples = augment_data(samples[0], samples[1], self.target_size)

                # Validate rgb and depth tensors
                rgb_valid = is_valid_tensor(samples[0])
                depth_valid = True
                if self.include_depth:
                    depth_valid = is_valid_tensor(samples[1])
                if not (rgb_valid and depth_valid):
                    print(f"Invalid data detected in {file_tuple}: rgb_valid={rgb_valid}, depth_valid={depth_valid}")
                    continue

                yield tuple(samples) if len(samples) > 1 else samples[0]
            except Exception as e:
                print(f"Error loading {file_tuple}: {e}")
                continue

def collate_fn(batch):
    # Each item is a tensor (rgb only)
    tensors = []
    for item in batch:
        tensor = item
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.max() > 1.1:  # Assume uint8 [0,255]
            tensor = (tensor / 127.5) - 1.0
        tensors.append(tensor)
    return torch.stack(tensors)

def collate_fn_depth(batch):
    # Each item is (rgb, depth)
    rgbs, depths = zip(*batch)
    batch_rgb = torch.stack(rgbs)
    batch_depth = torch.stack(depths)
    if batch_rgb.dtype != torch.float32:
        batch_rgb = batch_rgb.float()
    if batch_depth.dtype != torch.float32:
        batch_depth = batch_depth.float()
    if batch_rgb.max() > 1.1:
        batch_rgb = (batch_rgb / 127.5) - 1.0
    if batch_depth.max() > 1.1:
        batch_depth = (batch_depth / 127.5) - 1.0
    return batch_rgb, batch_depth

def collate_fn_flow(batch):
    # Each item is (rgb, flow)
    rgbs, flows = zip(*batch)
    batch_rgb = torch.stack(rgbs)
    batch_flow = torch.stack(flows)
    if batch_rgb.dtype != torch.float32:
        batch_rgb = batch_rgb.float()
    if batch_flow.dtype != torch.float32:
        batch_flow = batch_flow.float()
    if batch_rgb.max() > 1.1:
        batch_rgb = (batch_rgb / 127.5) - 1.0
    if batch_flow.max() > 1.1:
        batch_flow = (batch_flow / 127.5) - 1.0
    return batch_rgb, batch_flow

def collate_fn_depth_and_flow(batch):
    # Each item is (rgb, depth, flow)
    rgbs, depths, flows = zip(*batch)
    batch_rgb = torch.stack(rgbs)
    batch_depth = torch.stack(depths)
    batch_flow = torch.stack(flows)
    for t in (batch_rgb, batch_depth, batch_flow):
        if t.dtype != torch.float32:
            t = t.float()
    if batch_rgb.max() > 1.1:
        batch_rgb = (batch_rgb / 127.5) - 1.0
    if batch_depth.max() > 1.1:
        batch_depth = (batch_depth / 127.5) - 1.0
    if batch_flow.max() > 1.1:
        batch_flow = (batch_flow / 127.5) - 1.0
    return batch_rgb, batch_depth, batch_flow

def get_loader(batch_size, **data_kwargs):
    include_flow = data_kwargs.get('include_flow', False)
    include_depth = data_kwargs.get('include_depth', False)
    ds = LocalCoDDataset(**data_kwargs)
    if include_flow and include_depth:
        collate = collate_fn_depth_and_flow
    elif include_flow:
        collate = collate_fn_flow
    elif include_depth:
        collate = collate_fn_depth
    else:
        collate = collate_fn
    return DataLoader(ds, batch_size=batch_size, num_workers=8, prefetch_factor=2, pin_memory=True, collate_fn=collate)

if __name__ == "__main__":
    import time

    root_dir = "/mnt/data/datasets/cod_yt_v2"
    batch_size = 32

    loader = get_loader(batch_size, root_dir=root_dir, include_depth=True, include_flow=False, target_size=(256,256))
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
