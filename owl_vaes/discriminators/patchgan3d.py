import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def make_conv(ch_in, ch_out, k=4, s=2, p=1, bias=True):
    return weight_norm(nn.Conv3d(ch_in, ch_out, k, s, p, bias=bias))

class PatchGAN3D(nn.Module):
    """
    PatchGAN discriminator that outputs a grid of predictions.
    Each output value corresponds to a patch of the input image.

    Based on the original PatchGAN from pix2pix paper:
    "Image-to-Image Translation with Conditional Adversarial Networks"
    """

    def __init__(self, config):
        super().__init__()

        ch = getattr(config, 'ch_0', 64)
        channels = config.channels
        n_layers = getattr(config, 'n_layers', 3)

        self.n_layers = n_layers

        layers = []

        # First layer: no normalization
        layers.append(nn.Conv3d(channels, ch, kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1), bias=False))
        layers.append(nn.LeakyReLU(0.2))

        # Intermediate layers
        ch_mult = 1
        for i in range(1, n_layers):
            ch_mult_prev = ch_mult
            ch_mult = min(2 ** i, 8)  # Cap at 8x to prevent explosion
            layers.append(make_conv(ch * ch_mult_prev, ch * ch_mult, k=(3,4,4), s=(1,2,2), p=(1,1,1), bias=False))
            layers.append(nn.LeakyReLU(0.2))

        # Final layer before output
        ch_mult_prev = ch_mult
        ch_mult = min(2 ** n_layers, 8)
        layers.append(make_conv(ch * ch_mult_prev, ch * ch_mult, k=(3,4,4), s=(1,1,1), p=(1,1,1), bias=False))
        layers.append(nn.LeakyReLU(0.2))

        # Output layer: single channel, no normalization
        layers.append(nn.Conv3d(ch * ch_mult, 1, kernel_size=(3,4,4), stride=(1,1,1), padding=(1,1,1), bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x, output_hidden_states=False):
        """
        Forward pass through PatchGAN discriminator.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            If output_hidden_states=False: Tensor of shape (B, 1, T, H_out, W_out)
                where H_out and W_out depend on input size and number of layers
        """
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

if __name__ == "__main__":
    # Make a dataclass with required fields
    from dataclasses import dataclass

    from .patchgan import PatchGAN

    @dataclass
    class DummyConfig:
        sample_size:int= 256
        ch_0:int= 128
        n_layers:int= 3
        channels:int= 3

    model = PatchGAN3D(DummyConfig()).cuda().bfloat16()
    model_2d = PatchGAN(DummyConfig()).cuda().bfloat16()
    import torch

    # Compile models (basic torch.compile if available)
    try:
        model = torch.compile(model)
        model_2d = torch.compile(model_2d)
    except Exception:
        pass  # torch.compile may not be available in all torch versions

    with torch.no_grad():
        x = torch.randn(1,3,4,256,256).bfloat16().cuda()
        x_2d = x[:,:,0,:,:]  # (1,3,256,256)

        # Warmup
        for _ in range(3):
            _ = model(x)
            _ = model_2d(x_2d)

        # CUDA timing for 3D model
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(10):
            y = model(x)
        ender.record()
        torch.cuda.synchronize()
        time_3d = starter.elapsed_time(ender) / 1000.0  # seconds
        fps_3d = 10 / time_3d
        print(f"3D PatchGAN output shape: {y.shape}")
        print(f"3D PatchGAN FPS: {fps_3d:.2f}")

        # CUDA timing for 2D model
        starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter2.record()
        for _ in range(10):
            y_2d = model_2d(x_2d)
        ender2.record()
        torch.cuda.synchronize()
        time_2d = starter2.elapsed_time(ender2) / 1000.0  # seconds
        fps_2d = 10 / time_2d
        print(f"2D PatchGAN output shape: {y_2d.shape}")
        print(f"2D PatchGAN FPS: {fps_2d:.2f}")

        print(f"FPS comparison: 3D={fps_3d:.2f}, 2D={fps_2d:.2f}")