import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def make_conv(ch_in, ch_out, k=4, s=2, p=1, bias=True):
    return weight_norm(nn.Conv2d(ch_in, ch_out, k, s, p, bias=bias))


class PatchGAN(nn.Module):
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
        layers.append(nn.Conv2d(channels, ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        # Intermediate layers
        ch_mult = 1
        for i in range(1, n_layers):
            ch_mult_prev = ch_mult
            ch_mult = min(2 ** i, 8)  # Cap at 8x to prevent explosion
            layers.append(make_conv(ch * ch_mult_prev, ch * ch_mult, k=4, s=2, p=1, bias=False))
            layers.append(nn.LeakyReLU(0.2))

        # Final layer before output
        ch_mult_prev = ch_mult
        ch_mult = min(2 ** n_layers, 8)
        layers.append(make_conv(ch * ch_mult_prev, ch * ch_mult, k=4, s=1, p=1, bias=False))
        layers.append(nn.LeakyReLU(0.2))

        # Output layer: single channel, no normalization
        layers.append(nn.Conv2d(ch * ch_mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x, output_hidden_states=False):
        """
        Forward pass through PatchGAN discriminator.

        Args:
            x: Input tensor of shape (B, C, H, W)
            output_hidden_states: If True, return intermediate features

        Returns:
            If output_hidden_states=False: Tensor of shape (B, 1, H_out, W_out)
                where H_out and W_out depend on input size and number of layers
            If output_hidden_states=True: Tuple of (output, hidden_states_list)
        """
        if output_hidden_states:
            hidden_states = []
            h = x
            for layer in self.model:
                h = layer(h)
                if isinstance(layer, (nn.LeakyReLU)):
                    hidden_states.append(h.clone())
            return h, hidden_states
        else:
            return self.model(x)