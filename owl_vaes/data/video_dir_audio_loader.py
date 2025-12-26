import os, glob, random
from pathlib import Path
from fractions import Fraction
import numpy as np
import av
import random

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torchvision.transforms.functional as TF


class RandomAudioFromMP4s:
    """
    Continuous iterator yielding random audio chunks (2, n_samples) from MP4s.
    - Uniform over files; uniform over time within each file.
    - No persistent handles: each yield opens the chosen file once and closes it.
    - Duration/sample_rate are computed lazily on first use and cached in memory.
    - Ensures stereo output by converting mono to stereo or taking first 2 channels.
    """
    def __init__(self, source, seed=None, target_sampling_rate=44100, target_window_length=88200):
        # Ensure source is a list
        if isinstance(source, str):
            source = [source]
        self.target_sampling_rate = target_sampling_rate
        self.target_window_length = target_window_length  # number of samples
        # 1. Collect all .mp4 files (can be glob, dir, list, …)
        self.paths = self._find_mp4s(source)

        if not self.paths:
            raise RuntimeError("No videos found in the supplied source.")
        self.rng = random.Random(seed)
        self.meta = {}  # path -> (duration_s, sample_rate)

    @staticmethod
    def _find_mp4s(spec):
        """Return sorted unique .mp4 Paths from globs/dirs/files (abs/rel OK)."""
        specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
        out = []
        for s in specs:
            s = os.path.expanduser(str(s))
            p = Path(s)
            if p.exists() and p.is_dir():
                out += glob.glob(str(p / "**/*.mp4"), recursive=True)
            elif p.exists() and p.is_file() and p.suffix.lower() == ".mp4":
                out.append(str(p))
            else:
                # treat as (possibly absolute) glob pattern
                out += glob.glob(s, recursive=True)
        return [Path(x) for x in sorted({x for x in out if x.lower().endswith(".mp4")})]

    def __iter__(self):
        return self

    def __next__(self):
        max_attempts = 10  # Try up to 10 different videos before giving up
        for attempt in range(max_attempts):
            try:
                p = self.paths[self.rng.randrange(len(self.paths))]
                # If we already know (dur, sr), use it; otherwise compute inside the same open.
                if p in self.meta:
                    dur, sr = self.meta[p]
                    # Calculate window duration in seconds
                    window_dur = self.target_window_length / self.target_sampling_rate
                    # Pick random start time ensuring we can get full window
                    t = self.rng.random() * max(0.0, dur - window_dur)
                    audio = self._decode_at_time(p, t, window_dur)
                else:
                    # First time for this file: open once, read metadata, pick t, decode, close.
                    with av.open(str(p)) as c:
                        a = next((s for s in c.streams if s.type == "audio"), None)
                        if a is None:
                            raise RuntimeError(f"No audio stream found in {p}")

                        sr = a.sample_rate if a.sample_rate else 44100
                        dur = (c.duration / 1e6) if c.duration is not None else 600.0
                        self.meta[p] = (float(dur), int(sr))

                        window_dur = self.target_window_length / self.target_sampling_rate
                        t = self.rng.random() * max(0.0, dur - window_dur)
                        audio = self._decode_from_open(c, a, t, window_dur)  # decode using this same open

                return self._ensure_stereo_and_length(audio)

            except Exception as e:
                print(f"Error decoding audio from {p}: {e}. Trying another video...")
                # Remove failed video from meta cache if it exists
                if p in self.meta:
                    del self.meta[p]
                continue

        raise RuntimeError(f"Failed to decode audio after {max_attempts} attempts")

    @staticmethod
    def _decode_from_open(
            container: av.container.input.InputContainer,
            astream: av.audio.stream.AudioStream,
            t_sec: float,
            window_dur: float
    ) -> np.ndarray:
        """Seek+decode within an already-open container; returns audio (channels, samples)."""
        tb: Fraction = astream.time_base

        # Clamp t to known duration if available
        if container.duration is not None:
            max_t = max(0.0, container.duration / 1e6 - window_dur)
            t_sec = min(max(0.0, t_sec), max_t)

        # Seek to the target time
        container.seek(int(max(0.0, t_sec) / float(tb)), stream=astream, backward=True, any_frame=False)

        # Collect audio frames
        audio_frames = []
        end_time = t_sec + window_dur

        for pkt in container.demux(astream):
            for frame in pkt.decode():
                if frame.time is None or frame.time >= t_sec:
                    # Convert to numpy array (channels, samples)
                    audio_data = frame.to_ndarray()
                    # PyAV returns (samples, channels) or (channels, samples) depending on layout
                    # Ensure it's (channels, samples)
                    if audio_data.ndim == 1:
                        audio_data = audio_data.reshape(1, -1)
                    elif audio_data.shape[0] > audio_data.shape[1]:
                        audio_data = audio_data.T

                    audio_frames.append(audio_data)

                    if frame.time is not None and frame.time >= end_time:
                        break
            if frame.time is not None and frame.time >= end_time:
                break

        if not audio_frames:
            raise RuntimeError("No audio frames decoded")

        # Concatenate all frames
        audio = np.concatenate(audio_frames, axis=1)
        return audio

    def _ensure_stereo_and_length(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure audio is stereo (2 channels) and exactly target_window_length samples.
        - If mono, duplicate to stereo
        - If > 2 channels, take first 2
        - If too long, crop
        - If too short, pad with zeros
        """
        # Ensure stereo
        if audio.shape[0] == 1:
            # Mono -> Stereo: duplicate channel
            audio = np.repeat(audio, 2, axis=0)
        elif audio.shape[0] > 2:
            # More than stereo: take first 2 channels
            audio = audio[:2, :]

        # Ensure correct length
        current_length = audio.shape[1]
        if current_length > self.target_window_length:
            # Crop to target length
            audio = audio[:, :self.target_window_length]
        elif current_length < self.target_window_length:
            # Pad with zeros
            padding = self.target_window_length - current_length
            audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')

        return audio

    def _decode_at_time(self, path: Path, t_sec: float, window_dur: float) -> np.ndarray:
        """Open the file, decode audio at t, close; returns (channels, samples)."""
        with av.open(str(path), options={"fflags": "fastseek+nobuffer"}) as c:
            a = next((s for s in c.streams if s.type == "audio"), None)
            if a is None:
                raise RuntimeError(f"No audio stream in {path}")
            return self._decode_from_open(c, a, t_sec, window_dur)


class RandomWaveformDataset(IterableDataset):
    """
    Infinite stream of waveform tensors. One independent generator per worker.
    Returns audio in shape (2, n_samples) as float32 normalized to [-1, 1].
    Audio is always stereo.
    """
    def __init__(
        self,
        source,
        seed: int = 0,
        target_sampling_rate = 44100,
        target_window_length = 88200,  # 2 seconds of audio at 44.1kHz
        loudness_range = (0.5, 1.5),
    ):
        super().__init__()
        self.source = source
        self.seed = int(seed)
        self.target_sampling_rate = target_sampling_rate
        self.target_window_length = target_window_length
        self.loudness_range = loudness_range

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        # Derive a per-worker seed (works with persistent workers)
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        rng = RandomAudioFromMP4s(
            self.source,
            seed=int(wseed),
            target_sampling_rate=self.target_sampling_rate,
            target_window_length=self.target_window_length
        )
        for audio in rng:
            # Convert to torch tensor (already in shape (2, n_samples))
            # Normalize to [-1, 1] if needed (audio from PyAV is typically float32 already normalized)
            audio_tensor = torch.from_numpy(audio).float()

            # Pick a random scaling factor for loudness
            loudness_scale = random.uniform(self.loudness_range[0], self.loudness_range[1])
            audio_tensor = audio_tensor * loudness_scale

            # Ensure audio is normalized to [-1, 1]
            #max_val = audio_tensor.abs().max()
            #if max_val > 1.0:
            #    audio_tensor = audio_tensor / max_val

            yield audio_tensor.contiguous().clone().permute(1,0)

def get_loader(batch_size, num_workers=4, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = RandomWaveformDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        multiprocessing_context="spawn",
    )


if __name__ == "__main__":
    import time
    # Test with 2 seconds of audio at 44.1kHz = 88200 samples
    loader = get_loader(
        32,
        num_workers=8,
        source="/mnt/data/datasets/extracted_tars/kbm/fps/*/*.mp4",
        target_sampling_rate=44100,
        target_window_length=88200
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = iter(loader)
    for i in range(1000):
        t0 = time.time()
        batch_audio = next(loader)
        t1 = time.time()
        print(f"Batch {i}: shape {tuple(batch_audio.shape)}, dtype {batch_audio.dtype}, "
              f"min {batch_audio.min():.3f}, max {batch_audio.max():.3f}, load_time {t1-t0:.4f}s")
        _ = batch_audio.to(dev, non_blocking=True)  # example move to GPU