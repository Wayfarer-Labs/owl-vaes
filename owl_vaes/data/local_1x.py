from __future__ import annotations

from collections import defaultdict
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Generator
from torchcodec.decoders import VideoDecoder

import numpy as np
import torch
import ray
import time
from huggingface_hub import snapshot_download
from tqdm import tqdm
from torchvision.io import write_jpeg

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

@ray.remote(num_cpus=1, memory=8_000_000_000)  # 8GB per task - allows ~64 concurrent tasks
def _process_video(shard_idx: int, video_path: Path, segment_path: Path, save_dir: Path) -> dict:
    """Ray remote function for processing a single video shard."""
    segment_n   = _load_segment(segment_path)
    video_nchw  = VideoDecoder(str(video_path))

    assert segment_n.shape[0] == video_nchw._num_frames, f'{segment_n.shape=} {video_nchw._num_frames=}'
    
    episode_to_frames: dict[int, list[torch.Tensor]] = defaultdict(list)

    for i, frame in enumerate(video_nchw):
        episode = int(segment_n[i])
        episode_to_frames[episode].append(frame)

    for episode, frames in episode_to_frames.items():
        for frame_idx, frame_chw in enumerate(frames):
            imgs_dir = save_dir.absolute() / 'images'
            imgs_dir.mkdir(exist_ok=True, parents=True)
            filename = str(imgs_dir / f'rgb_sh{shard_idx}_ep{episode}_fr{frame_idx}.jpeg')
            write_jpeg(frame_chw, filename, quality=95)

        print(f'Finished writing {episode=} in shard={shard_idx} with n-frames={len(frames)}')

    print(f'Finished writing entire shard {shard_idx} with num-episodes={len(episode_to_frames)}')
    video_info = {
        'shard_idx': shard_idx,
        'episodes': list(episode_to_frames.keys()),
        'num_frames_per_episode': {episode_idx: len(frames) for episode_idx, frames in episode_to_frames.items()}
    }

    metadata_dir  = save_dir.absolute() / 'metadata' 
    metadata_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = metadata_dir / f'metadata_{shard_idx}.json'

    with open(metadata_path, 'w') as f:
        f.write(json.dumps(video_info))
    
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
    total_frames = sum(result['total_frames'] for result in results)
    total_time = max(result['processing_time_seconds'] for result in results)  # Max since parallel
    avg_time_per_shard = sum(result['processing_time_seconds'] for result in results) / len(results)
    
    print(f"\n=== Processing Summary ===")
    print(f"Total frames processed: {total_frames:,}")
    print(f"Total shards: {len(results)}")
    print(f"Wall clock time: {total_time:.2f} seconds")
    print(f"Average time per shard: {avg_time_per_shard:.2f} seconds")
    print(f"Effective throughput: {total_frames/total_time:.1f} frames/second")
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Optionally shutdown Ray when done
    ray.shutdown()


if __name__ == "__main__":
    setup_dataset()