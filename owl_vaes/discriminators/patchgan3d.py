import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from .patchgan import PatchGAN

class PatchGAN3D(PatchGAN):
    """
    PatchGAN discriminator that outputs a grid of predictions.
    Each output value corresponds to a patch of the input image.

    Based on the original PatchGAN from pix2pix paper:
    "Image-to-Image Translation with Conditional Adversarial Networks"
    """

    def __init__(self, config):
        n_frames = getattr(config, 'n_frames', 4)
        config = deepcopy(config)
        config.channels = config.channels * n_frames
        super().__init__(config)

    def forward(self, x, output_hidden_states=False):
        """
        Assume x is [b,n,c,h,w]
        """
        b,n,c,h,w = x.shape
        x = x.contiguous().view(b,n*c,h,w).contiguous()
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