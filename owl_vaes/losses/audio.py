import torch
from torch import Tensor
import torch.nn.functional as F

def stft_loss(
    x_rec: Tensor,
    x_target: Tensor,
    n_fft_list: list[int] = [1024, 2048, 512],
    hop_length_ratio: float = 0.25,
) -> Tensor:
    """
    Multi-scale STFT loss for audio reconstruction
    Using view_as_real to properly handle complex tensors
    """
    total_loss = 0.0

    x_rec = x_rec.to(torch.float32)
    x_target = x_target.to(torch.float32)

    for n_fft in n_fft_list:
        hop_length = int(n_fft * hop_length_ratio)

        stft_rec = torch.stft(
            x_rec.view(-1, x_rec.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )
        stft_target = torch.stft(
            x_target.view(-1, x_target.size(-1)),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
        )

        mag_rec = torch.abs(stft_rec)
        mag_target = torch.abs(stft_target)
        mag_loss = F.l1_loss(mag_rec, mag_target)

        # Convert Complex tensors to real so torch doesn't panic
        stft_rec_real = torch.view_as_real(stft_rec)        # Shape: (..., 2) where last dim is [real, imag]
        stft_target_real = torch.view_as_real(stft_target)

        # Compute MSE on the real-valued tensor (contains both real and imaginary parts)
        phase_loss = F.mse_loss(stft_rec_real, stft_target_real)

        total_loss = total_loss + mag_loss + phase_loss

    return total_loss / len(n_fft_list)

@torch.compile(mode="max-autotune")
def lr_to_ms(audio: Tensor) -> Tensor:
    """
    Convert Left-Right (L/R) stereo to Mid-Side (M/S)

    Args:
        audio: Input audio tensor (B, 2, T) where channel 0=L, channel 1=R

    Returns:
        Audio in M/S format (B, 2, T) where channel 0=M, channel 1=S
    """
    if audio.size(1) != 2:
        return audio

    left = audio[:, 0:1, :].clone()  # (B, 1, T)
    right = audio[:, 1:2, :].clone()  # (B, 1, T)

    mid = (
        left + right
    ) * 0.5  # Mid channel - use multiplication for better compilation
    side = (left - right) * 0.5  # Side channel

    return torch.cat([mid, side], dim=1)

@torch.compile(mode="max-autotune")
def compute_ms_loss(batch_rec: Tensor, batch: Tensor) -> Tensor:
    """
    Compute M/S component loss - Compiled for performance.
    """
    batch_ms = lr_to_ms(batch)
    batch_rec_ms = lr_to_ms(batch_rec)
    return F.mse_loss(batch_rec_ms, batch_ms)

