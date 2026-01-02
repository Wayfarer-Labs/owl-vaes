from __future__ import annotations

from collections import defaultdict
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Generator
from torchcodec.decoders import VideoDecoder
from typing import Literal
import numpy as np
import torch
import ray
import time
from huggingface_hub import snapshot_download
from tqdm import tqdm
from torchvision.io import write_jpeg, decode_image
from torch.utils.data import DataLoader, IterableDataset

HF_URL       = "https://huggingface.co/datasets/1x-technologies/world_model_raw_data"
DATASET_ROOT = Path("/mnt") / "data" / "sami" / "1x_dataset" / "original"
IMAGE_PATH   = Path("/mnt") / "data" / "sami" / "1x_dataset" / "data"


def _repo_id_from_url(hf_url: str) -> str:
    # e.g. https://huggingface.co/datasets/1x-technologies/world_model_raw_data
    # should return 1x-technologies/world_model_raw_data

    parts = hf_url.rstrip("/").split("/")

    if len(parts) < 2: raise ValueError(f"Unrecognized HF URL: {hf_url}")

    # last two path components after '/datasets/'
    if "datasets" in parts:
        i = parts.index("datasets")
        return "/".join(parts[i + 1 : i + 3])

    # fallback: assume already repo_id
    return parts[-2] + "/" + parts[-1]

def _load_segment(segment_path: Path) -> np.memmap:
    return np.memmap(str(segment_path.absolute()), dtype=np.uint32)


# Alternative version that's even more memory efficient - processes in chunks
@ray.remote(num_cpus=1, memory=8_000_000_000, retry_exceptions=False)
def _process_video(shard_idx: int, video_path: Path, segment_path: Path, save_dir: Path) -> dict:
    """Ray remote function for processing a single video shard."""
    segment_n = _load_segment(segment_path)
    video_nchw = VideoDecoder(str(video_path))

    assert segment_n.shape[0] == video_nchw._num_frames, f'{segment_n.shape=} {video_nchw._num_frames=}'
    
    imgs_dir = save_dir.absolute() / 'images'
    imgs_dir.mkdir(exist_ok=True, parents=True)

    # Initialize with the first episode from the data
    current_episode = int(segment_n[0]) if len(segment_n) > 0 else 0
    episode_start_idx = 0
    episode_to_num_frames: dict[int, int] = {}
    episodes_seen = []

    # Process all frames
    for i, frame_chw in enumerate(video_nchw):
        episode = int(segment_n[i])

        # Detect episode boundary
        if episode != current_episode:
            # Record the previous episode's frame count
            frames_in_prev_episode = i - episode_start_idx
            episode_to_num_frames[current_episode] = frames_in_prev_episode
            episodes_seen.append(current_episode)
            
            print(f'Shard {shard_idx}: Episode {current_episode} completed with {frames_in_prev_episode} frames')
            
            # Start tracking the new episode
            episode_start_idx = i
            current_episode = episode

        # Calculate frame index within the current episode
        frame_idx = i - episode_start_idx
        filename = str(imgs_dir / f'rgb_sh{shard_idx}_ep{episode}_fr{frame_idx}.jpeg')
        write_jpeg(frame_chw, filename, quality=95)

    # Don't forget the final episode!
    final_episode_frames = len(segment_n) - episode_start_idx
    episode_to_num_frames[current_episode] = final_episode_frames
    episodes_seen.append(current_episode)
    
    print(f'Shard {shard_idx}: Final episode {current_episode} completed with {final_episode_frames} frames')
    print(f'Shard {shard_idx}: Finished processing {len(episodes_seen)} episodes, {len(segment_n)} total frames')

    video_info = {
        'shard_idx': shard_idx,
        'episodes': episodes_seen,  # List of episode IDs, not just the last one
        'num_frames_per_episode': episode_to_num_frames,
        'total_frames': len(segment_n),
        'total_episodes': len(episodes_seen)
    }

    metadata_dir = save_dir.absolute() / 'metadata' 
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / f'metadata_{shard_idx}.json'

    with open(metadata_path, 'w') as f:
        json.dump(video_info, f, indent=2)  # Use json.dump instead of json.dumps
    
    return video_info
    

def train_iter() -> Generator[tuple[int, Path, Path]]:
    video_dir       = DATASET_ROOT  / 'train_v2.0_raw' / 'videos'
    segment_dir     = DATASET_ROOT  / 'train_v2.0_raw' / 'segment_indices'

    for video_path in video_dir.glob('*.mp4'):
        shard_idx    = int(str(video_path.stem).split('_')[-1])
        segment_path = segment_dir / f'segment_idx_{shard_idx}.bin'
        yield (shard_idx, video_path, segment_path)


def setup_dataset(hf_url: str = HF_URL) -> None:
    # Initialize Ray with high-performance configuration
    # Using ~half of your 1.5TB RAM (750GB) and half of 128 cores (64 cores)
    if not ray.is_initialized():
        ray.init(
            num_cpus=64,                    # Use 64 of your 128 cores
            object_store_memory=600_000_000_000,  # 600GB for object store (~80% of target 750GB)
            _memory=150_000_000_000,       # 150GB for regular Ray memory
            ignore_reinit_error=True
        )

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_PATH  .mkdir(parents=True, exist_ok=True)

    # 1) Download from Hugging Face into DATASET_ROOT / 'original'
    repo_id     = _repo_id_from_url(hf_url)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(DATASET_ROOT),
        local_dir_use_symlinks=False,
    )

    train_dir = IMAGE_PATH / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    # Process videos in parallel using Ray with progress tracking
    tasks = []
    task_ids = []


    for shard_idx, video_path, segment_path in train_iter():
        task = _process_video.remote(shard_idx, video_path, segment_path, train_dir)
        tasks.append(task)
        task_ids.append(shard_idx)
    
    print(f'Submitted {len(tasks)} tasks to Ray cluster (64 cores, 750GB RAM)')
    
    # Process results as they complete for better monitoring
    completed = 0
    remaining_tasks = tasks.copy()
    results = []
    
    while remaining_tasks:
        # Wait for at least one task to complete
        ready_tasks, remaining_tasks = ray.wait(remaining_tasks, num_returns=1)
        
        # Get results from completed tasks
        batch_results = ray.get(ready_tasks)
        results.extend(batch_results)
        completed += len(batch_results)
        
        print(f'Progress: {completed}/{len(tasks)} video shards completed')
    
    print(f'All {len(results)} video shards processed successfully!')
    
    # Calculate and print processing statistics
    total_frames = sum(sum(result['num_frames_per_episode'].values()) for result in results)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Total shards: {len(results)}")
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Optionally shutdown Ray when done
    ray.shutdown()



class Robotics_1X_Dataset(IterableDataset):
    def __init__(self, root: Path = IMAGE_PATH, split: Literal['train', 'val'] = 'train'):
        super().__init__()

        self.image_root     = root / split / 'images'
        self.metadata_root  = root / split / 'metadata'

        self.all_metadata: list[dict]   = [
            json.load(open(str(path)))
            for path in self.metadata_root.glob('metadata_*.json')
        ]

        self.metadata: dict = {
            meta['shard_idx']: meta for meta in self.all_metadata
        } 
        # NOTE turns out each metadata has a global episode, instead of a local one.
        # e.g. metadata_0.json ends at episode 519, metadata_1.jsson starts at 520.
        # This means we can randomly sample an episode, and then a frame number, and load.
        self.episode_to_shard = {
            e: meta['shard_idx']
            for meta in self.all_metadata
            for e in meta['episodes']
        }
        self.episode_to_nframes = {
            int(e): meta['num_frames_per_episode'][e]
            for meta in self.all_metadata
            for e in meta['num_frames_per_episode']
        }
        assert set(self.episode_to_shard) == set(self.episode_to_nframes)
        print(f'Loaded metadata for {len(self.episode_to_shard)} episodes totaling {sum(self.episode_to_nframes.values())} frames')

    def random_filename(self) -> Path:
        import random
        episode      = random.randint(0, max(self.episode_to_shard.keys()))
        frame        = random.randint(0, self.episode_to_nframes[episode]-1)
        shard        = self.episode_to_shard[episode]
        return self.image_root / f'rgb_sh{shard}_ep{episode}_fr{frame}.jpeg'
    
    def get_random_image(self) -> torch.Tensor:
        filename    = self.random_filename()
        image_chw   = decode_image(filename)
        return image_chw

    def __iter__(self):
        while True: yield self.get_random_image()

def collate_fn(x):
    # x is list of frames
    res = torch.stack(x)
    return res  # [b,c,h,w]

def get_loader(batch_size, **data_kwargs):
    dataset = Robotics_1X_Dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **data_kwargs
    )
    return loader

if __name__ == "__main__":
    # setup_dataset()
    ds = Robotics_1X_Dataset()
    dl = get_loader(64, num_workers=16)
    idl = iter(dl)
    i = 0
    n = 256
    ttkn = 0
    while i < n:
        start_time = time.time()
        batch = next(idl)
        end_time = time.time()
        print(f'Batch load took {end_time - start_time}')
        i += 1
        ttkn += end_time - start_time

    print(f'time taken avg: {ttkn / n}')