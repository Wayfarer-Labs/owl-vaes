import os, glob, random
from pathlib import Path
from fractions import Fraction
import numpy as np
import av

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torchvision.transforms.functional as TF


class RandomRGBFromMP4s:
    """
    Continuous iterator yielding random RGB frames (H,W,3 uint8) from MP4s.
    - Uniform over files; uniform over time within each file.
    - No persistent handles: each yield opens the chosen file once and closes it.
    - Duration/fps are computed lazily on first use and cached in memory.
    """
    def __init__(self, source, seed=None, target_size=(360,640)):
        # Ensure source is a list
        if isinstance(source, str):
            source = [source]
        self.target_size = target_size  # (H, W)
        # 1. Collect all .mp4 files (can be glob, dir, list, …)
        self.paths = self._find_mp4s(source)

        if not self.paths:
            raise RuntimeError("No videos found in the supplied source.")
        self.rng = random.Random(seed)
        self.meta = {}  # path -> (duration_s, eps_end_s)

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
                # If we already know (dur, eps), use it; otherwise compute inside the same open.
                if p in self.meta:
                    dur, eps = self.meta[p]
                    t = self.rng.random() * max(0.0, dur - eps)
                    frame = self._decode_at_time(p, t)
                else:
                    # First time for this file: open once, read metadata, pick t, decode, close.
                    with av.open(str(p)) as c:
                        v = next(s for s in c.streams if s.type == "video")
                        fps = float(v.average_rate) if v.average_rate else 30.0
                        dur = (c.duration / 1e6) if c.duration is not None else (
                            (float(v.frames) / fps) if (v.frames and fps) else 600.0
                        )
                        eps = 1.0 / max(1.0, fps)
                        self.meta[p] = (float(dur), float(eps))
                        t = self.rng.random() * max(0.0, dur - eps)
                        frame = self._decode_from_open(c, v, t)  # decode using this same open

                # Resize frame if needed
                return self._resize_if_needed(frame)

            except Exception as e:
                print(f"Error decoding {p}: {e}. Trying another video...")
                # Remove failed video from meta cache if it exists
                if p in self.meta:
                    del self.meta[p]
                continue

        raise RuntimeError(f"Failed to decode a video after {max_attempts} attempts")

    @staticmethod
    def _decode_from_open(
            container: av.container.input.InputContainer,
            vstream: av.video.stream.VideoStream,
            t_sec: float
    ) -> np.ndarray:
        """Seek+decode within an already-open container; returns RGB24 (H,W,3)."""
        tb: Fraction = vstream.time_base
        # Light speed knobs (don’t persist beyond this call anyway)
        try:
            vstream.thread_type = "FRAME"
            vstream.codec_context.thread_count = 1
            vstream.codec_context.skip_loop_filter = "ALL"
        except Exception:
            pass
        # Clamp t to known duration if available
        if container.duration is not None:
            t_sec = min(max(0.0, t_sec), max(0.0, container.duration / 1e6 - 1e-3))
        # Keyframe seek then decode forward to >= t
        try:
            vstream.codec_context.skip_frame = "NONKEY"
            container.seek(int(max(0.0, t_sec) / float(tb)), stream=vstream, backward=True, any_frame=False)
        finally:
            vstream.codec_context.skip_frame = "DEFAULT"

        last = None
        for pkt in container.demux(vstream):
            for fr in pkt.decode():
                last = fr
                if fr.time is not None and fr.time + 1e-6 >= t_sec:
                    return fr.to_ndarray(format="rgb24")
        # Fallbacks
        if last is not None:
            return last.to_ndarray(format="rgb24")
        container.seek(0, stream=vstream)
        for pkt in container.demux(vstream):
            for fr in pkt.decode():
                return fr.to_ndarray(format="rgb24")
        raise RuntimeError("Decode failed")

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target_size if needed. Uses AREA interpolation for downsampling."""
        h, w = frame.shape[:2]
        target_h, target_w = self.target_size

        # Skip if already at target size
        if h == target_h and w == target_w:
            return frame

        # Convert to torch tensor (HWC -> CHW), resize, then convert back
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # CHW
        # Use AREA interpolation (antialias=True) for best quality downsampling
        # This is the fastest high-quality method for downsampling
        resized = TF.resize(frame_tensor, [target_h, target_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        return resized.permute(1, 2, 0).numpy()  # CHW -> HWC

    def _decode_at_time(self, path: Path, t_sec: float) -> np.ndarray:
        """Open the file, decode at t, close; returns RGB24 (H,W,3)."""
        with av.open(str(path), options={"fflags": "fastseek+nobuffer"}) as c:
            v = next(s for s in c.streams if s.type == "video")
            return self._decode_from_open(c, v, t_sec)


class RandomRGBDataset(IterableDataset):
    """
    Infinite stream of CHW uint8 frames. One independent generator per worker.
    Assumes frames share a common resolution so default collate can stack.
    """
    def __init__(self, source, seed: int = 0, target_size = (360, 640)):
        super().__init__()
        self.source = source
        self.seed = int(seed)
        self.target_size = target_size

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        # Derive a per-worker seed (works with persistent workers)
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        rng = RandomRGBFromMP4s(self.source, seed=int(wseed), target_size = self.target_size)
        for rgb in rng:
            # HWC uint8 -> CHW uint8; clone() gives the tensor its own
            # resizable storage, preventing rare ‘resize_ not allowed’ errors.
            yield torch.from_numpy(rgb).permute(2, 0, 1).contiguous().clone().bfloat16() / 127.5 - 1.0

def get_loader(batch_size, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = RandomRGBDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        multiprocessing_context="spawn",
    )


if __name__ == "__main__":
    import time
    #loader = get_loader(32, source="/mnt/data/datasets/extracted_tars/kbm/fps/*/*.mp4", target_size = (360, 640))
    loader = get_loader(
        4,
        source="/mnt/data/waypoint_1/data/MKIF/*/*.mp4",
        target_size = (720,1280)
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = iter(loader)
    for i in range(1000):
        t0 = time.time()
        batch_u8 = next(loader)
        t1 = time.time()
        print(f"Batch {i}: shape {tuple(batch_u8.shape)}, dtype {batch_u8.dtype}, load_time {t1-t0:.4f}s")
        _ = (batch_u8.to(dev, non_blocking=True).float() / 255.0)  # example move/normalize