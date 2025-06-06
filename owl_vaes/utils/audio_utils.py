##
# Audio utils
# Stolen from stable-audio repo
# credits: https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/data/utils.py
###
import math
import os
import random
from typing import Tuple

import torch
from torch import nn
from torch.types import Tensor
from torchaudio import transforms as T


class PadCrop(nn.Module):
    def __init__(self, n_samples: int, randomize: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal: Tensor):
        n, s = signal.shape

        start = (
            0
            if (not self.randomize)
            else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        )

        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, : min(s, self.n_samples)] = signal[:, start:end]  # type: ignore[misc]
        return output


class PadCrop_Normalized_T(nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        super().__init__()

        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: Tensor) -> Tuple[Tensor, float, float, int, int, Tensor]:
        n_channels, n_samples = source.shape

        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)

        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if self.randomize and n_samples > self.n_samples:
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, : min(n_samples, self.n_samples)] = source[
            :, offset : offset + self.n_samples
        ]

        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[: min(n_samples, self.n_samples)] = 1

        return (chunk, t_start, t_end, seconds_start, seconds_total, padding_mask)


class PhaseFlipper(nn.Module):
    "Randomly invert the phase of a signal"

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, signal):
        return -signal if (random.random() < self.p) else signal


class Mono(nn.Module):
    def __call__(self, signal: Tensor):
        if len(signal.shape) > 1:
            return torch.mean(input=signal, dim=0, keepdims=True)  # type: ignore[call-overload]
        else:
            return signal


class Stereo(nn.Module):
    def __call__(self, signal: Tensor):
        signal_shape = signal.shape
        # Check if it's mono
        if len(signal_shape) == 1:  # s -> 2, s
            signal = signal.unsqueeze(0).repeat(2, 1)
        elif len(signal_shape) == 2:
            if signal_shape[0] == 1:  # 1, s -> 2, s
                signal = signal.repeat(2, 1)
            elif signal_shape[0] > 2:  # ?, s -> 2,s
                signal = signal[:2, :]

        return signal


class VolumeNorm(nn.Module):
    "Volume normalization and augmentation of a signal [LUFS standard]"

    def __init__(
        self,
        params: list[int] = [-16, 2],
        sample_rate: int = 16000,
        energy_threshold: float = 1e-6,
    ):
        super().__init__()
        self.loudness = T.Loudness(sample_rate)
        self.value = params[0]
        self.gain_range = [-params[1], params[1]]
        self.energy_threshold = energy_threshold

    def __call__(self, signal: Tensor):
        """
        signal: torch.Tensor [channels, time]
        """
        # avoid do normalisation for silence
        energy = torch.mean(signal**2)
        if energy < self.energy_threshold:
            return signal

        input_loudness = self.loudness(signal)
        # Generate a random target loudness within the specified range
        target_loudness = self.value + (
            torch.rand(1).item() * (self.gain_range[1] - self.gain_range[0])
            + self.gain_range[0]
        )
        delta_loudness = target_loudness - input_loudness
        gain = torch.pow(10.0, delta_loudness / 20.0)
        output = gain * signal

        # Check for potentially clipped samples
        if torch.max(torch.abs(output)) >= 1.0:
            output = self.declip(output)

        return output

    def declip(self, signal: Tensor):
        """
        Declip the signal by scaling down if any samples are clipped
        """
        max_val = torch.max(torch.abs(signal))

        if max_val > 1.0:
            signal = signal / max_val
            signal *= 0.95

        return signal

# === === === ===
# Helper Functions
# === === === ===

def fast_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions,
    # max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
):
    """Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"""
    subfolders, files = [], []

    ext = [
        "." + x if x[0] != "." else x for x in ext
    ]  # add starting period to extensions if needed

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)  # type: ignore[arg-type]

    return subfolders, files


def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list[str],  # list of allowed file extensions
    keywords: list[str],  # list of keywords to search for in the file name
):
    """Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"""
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ["." + x if x[0] != "." else x for x in ext]

    banned_words = ["paxheader", "__macosx"]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == "."
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any([keyword in name_lower for keyword in keywords])
                    has_banned = any([
                        banned_word in name_lower for banned_word in banned_words
                    ])
                    if (
                        has_ext
                        and has_keyword
                        and not has_banned
                        and not is_hidden
                        and not os.path.basename(f.path).startswith("._")
                    ):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)  # type: ignore[arg-type]
    return subfolders, files

def get_audio_filenames(
    paths: str | list[str],  # directories in which to search
    keywords: list[str] | None = None,
    exts=[".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus"],
):
    "recursively get a list of audio filenames"
    filenames = []

    if isinstance(paths, str):
        paths = [paths]

    for _path in paths:  # get a list of relevant filenames
        if keywords is not None:
            subfolders, files = keyword_scandir(_path, exts, keywords)
        else:
            subfolders, files = fast_scandir(_path, exts)
        filenames.extend(files)
    return filenames


# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio: Tensor,  # torch tensor of (multichannel) audio
):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20 * torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()


def is_silence(
    audio,  # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh
