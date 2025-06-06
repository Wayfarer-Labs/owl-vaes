import math
from typing import Literal

import torch
from alias_free_torch import Activation1d
from torch import exp, nn, pow, sin
from torch.types import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):
    def __init__(
        self, in_features: int, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super(SnakeBeta, self).__init__()

        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def snake_beta(self, x, alpha, beta):
        return x + (1.0 / (beta + 0.000000001)) * pow(sin(x * alpha), 2)

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)

        if self.alpha_logscale:
            alpha = exp(alpha)
            beta = exp(beta)

        x = self.snake_beta(x, alpha, beta)

        return x


def get_activation(
    activation: Literal["elu", "snake", "none"],
    antialias: bool = False,
    channels: int | None = None,
):
    act: Activation1d | SnakeBeta | nn.Module = nn.Identity()

    match activation:
        case "elu":
            act = nn.ELU()
        case "snake":
            if channels is not None:
                act = SnakeBeta(channels)
        case "none":
            act = nn.Identity()
        case _:
            raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


def WNConv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(function, *args, **kwargs)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = x

        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)

        return x + res


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=1,
                use_snake=use_snake,
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=3,
                use_snake=use_snake,
            ),
            ResidualUnit(
                in_channels=in_channels,
                out_channels=in_channels,
                dilation=9,
                use_snake=use_snake,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
    ):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = WNConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            upsample_layer,
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1,
                use_snake=use_snake,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3,
                use_snake=use_snake,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9,
                use_snake=use_snake,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
