import torch
import torch.nn.functional as F
from .patchgan import PatchGAN

class ReconstructionGAN(PatchGAN):
    def __init__(self, config):
        config.channels = config.channels * 2
        super().__init__(config)

    def forward(self, x, output_hidden_states=False):
        return super().forward(x, output_hidden_states)