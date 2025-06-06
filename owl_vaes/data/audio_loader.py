import os
import random
import time
from dataclasses import dataclass
from os import path
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.types import Tensor
from torch.utils.data import DataLoader
from torchaudio import load
from torchaudio import transforms as T
from torchtyping import TensorType

from owl_vaes.utils.audio_utils import (
    Mono,
    PadCrop_Normalized_T,
    PhaseFlipper,
    Stereo,
    get_audio_filenames,
    is_silence,
)

@dataclass
class LocalDatasetConfig:
    def __init__(
        self, id: str, path: str, custom_metadata_fn: Callable[[str], str] | None = None
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: list[LocalDatasetConfig],
        sample_size: int = 65536,
        sample_rate: int = 48000,
        keywords: list[str] | None = None,
        random_crop: bool = True,
        force_channels: str = "stereo",
    ):
        super().__init__()

        self.filenames = []
        self.root_paths = []
        self.sample_size = sample_size
        self.random_crop = random_crop

        self.augs = torch.nn.Sequential(PhaseFlipper())

        self.pad_crop = PadCrop_Normalized_T(
            sample_size, sample_rate, randomize=random_crop
        )

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate
        self.custom_metadata_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            self.filenames.extend(get_audio_filenames(config.path, keywords))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        self._calculate_chunks()

    def _calculate_chunks(self):
        """Calculate total number of chunks across all files."""
        self.file_chunks = []
        self.total_chunks = 0

        for filename in self.filenames:
            try:
                audio, in_sr = load(filename)

                if in_sr != self.sr:
                    audio_length = int(audio.shape[-1] * self.sr / in_sr)
                else:
                    audio_length = audio.shape[-1]

                if self.random_crop:
                    chunks_per_file = max(
                        1, audio_length // self.sample_size * 10
                    )  # 10x oversampling
                else:
                    chunks_per_file = max(1, audio_length // self.sample_size)

                self.file_chunks.append((filename, chunks_per_file, audio_length))
                self.total_chunks += chunks_per_file

            except Exception as e:
                print(f"Warning: Could not process {filename}: {e}")
                continue

        print(
            f"== Found {len(self.filenames)} files with {self.total_chunks} total chunks =="
        )

    def load_file(self, filename: str) -> Tensor:
        ext = filename.split(".")[-1]
        audio, in_sr = load(filename, format=ext)
        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)
        return audio

    def __len__(self):
        return self.total_chunks

    def __getitem__(
        self, chunk_idx: int
    ) -> tuple[TensorType["batch", "channels", "n_samples"], Dict[str, str | Any]]:  # noqa: F821
        for filename, chunks_per_file, audio_length in self.file_chunks:
            if chunk_idx < chunks_per_file:
                break
            chunk_idx -= chunks_per_file

        try:
            start_time = time.time()
            audio = self.load_file(filename)

            # If random cropping, the PadCrop_Normalized_T transform will randomize the start;
            # otherwise, compute deterministic offset
            if not self.random_crop:
                offset = chunk_idx * self.sample_size
                # Slice the audio to that chunk (if shorter near the end, the PadCrop will handle it)
                audio = audio[:, offset: offset + self.sample_size]

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = (
                self.pad_crop(audio)
            )

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info: Dict[str, Any] = {}
            info["path"] = filename

            for root_path in self.root_paths:
                if root_path in filename:
                    info["relpath"] = path.relpath(filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            info["sample_rate"] = self.sr
            end_time = time.time()
            info["load_time"] = end_time - start_time

            # Any custom metadata injection
            for custom_md_path, custom_metadata_fn in self.custom_metadata_fns.items():
                if custom_md_path in filename:
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                    del info["__audio__"]

            return (audio, info)  # type: ignore

        except Exception as e:
            print(f"Couldn't load file {filename}: {e}")
            return self[random.randrange(len(self))]

def collate_fn(samples):
    batched = list(zip(*samples))
    result = []

    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        else:
            b = b
        result.append(b)

    return result
    
def get_audio_loader(batch_size: int, paths: list[str] | str):
    """Get data loader for Sample Audio Model pipeline."""
    world_size = 1
    global_rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

    paths = [paths] if isinstance(paths, str) else paths
    assert all([os.path.exists(i) for i in paths]), (
        "Some provided paths do not exist/are detectable."
    )

    train_dataset = SampleDataset(
        configs=[LocalDatasetConfig(str(i), path) for i, path in enumerate(paths)],
        random_crop=True,
        sample_size=81408,  # 2 * 44100, rounded
        sample_rate=44100,
    )

    if world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator("cpu"),
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True
    )

    return train_loader


if __name__ == "__main__":
    ll = get_audio_loader(32, "my_data/")
