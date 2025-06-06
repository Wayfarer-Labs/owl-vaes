import torch
from torch import nn
from torchtyping import TensorType

from owl_vaes.data.audio_loader import get_audio_loader
from owl_vaes.nn.audio_blocks import (
    DecoderBlock,
    EncoderBlock,
    WNConv1d,
    get_activation,
)

class OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] = [1, 2, 4, 8],
        strides: list[int] = [2, 4, 8, 8],
        use_snake: bool = False,
        antialias_activation: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=in_channels,
                out_channels=c_mults[0] * channels,
                kernel_size=7,
                padding=3,
            )
        ]

        for i in range(self.depth - 1):
            layers += [
                EncoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i + 1] * channels,
                    stride=strides[i],
                    use_snake=use_snake,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[-1] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[-1] * channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=1,
            ),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        self.out_channels = out_channels

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(
                in_channels=latent_dim,
                out_channels=c_mults[-1] * channels,
                kernel_size=7,
                padding=3,
            ),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [
                DecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            ]

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            WNConv1d(
                in_channels=c_mults[0] * channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self.layers(x)


class AudioAutoEncoder(nn.Module):
    """
    Complete Audio AutoEncoder combining OobleckEncoder and OobleckDecoder.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()

        # Extract parameters from config with defaults
        in_channels = getattr(config, "in_channels", 2)
        out_channels = getattr(config, "out_channels", 2)
        channels = getattr(config, "channels", 128)
        latent_dim = getattr(config, "latent_dim", 32)
        c_mults = getattr(config, "c_mults", [1, 2, 4, 8])
        strides = getattr(config, "strides", [2, 4, 8, 8])
        use_snake = getattr(config, "use_snake", False)
        antialias_activation = getattr(config, "antialias_activation", False)
        use_nearest_upsample = getattr(config, "use_nearest_upsample", False)
        final_tanh = getattr(config, "final_tanh", True)

        self.encoder = OobleckEncoder(
            in_channels=in_channels,
            channels=channels,
            latent_dim=latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
        )

        self.decoder = OobleckDecoder(
            out_channels=out_channels,
            channels=channels,
            latent_dim=latent_dim,
            c_mults=c_mults,
            strides=strides,
            use_snake=use_snake,
            antialias_activation=antialias_activation,
            use_nearest_upsample=use_nearest_upsample,
            final_tanh=final_tanh,
        )

        # Calculate total stride for shape validation
        self.total_stride = 1

        for stride in strides:
            self.total_stride *= stride

    def encode(self, x: TensorType) -> TensorType:
        return self.encoder(x)

    def decode(self, z: TensorType) -> TensorType:
        return self.decoder(z)

    def forward(self, x: TensorType) -> tuple[TensorType, TensorType]:
        """
        Forward pass through encoder and decoder.

        Args:
            x: Input audio tensor of shape (batch, channels, time)

        Returns:
            Tuple: (reconstructed_audio, latent_representation)
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def get_compression_ratio(self) -> int:
        return self.total_stride


if __name__ == "__main__":
    loader = get_audio_loader(1, "my_data/")
    sample = next(iter(loader))[0]

    myEncoder = OobleckEncoder(2, 128, use_snake=True, antialias_activation=True)
    myDecoder = OobleckDecoder(
        2, 128, antialias_activation=True, use_nearest_upsample=True
    )

    model_config = {
        "in_channels": 2,
        "out_channels": 2,
        "use_snake": True,
        "antialias_activation": True,
        "use_nearest_upsample": True,
    }

    model = AudioAutoEncoder(model_config)

    latent = myEncoder(sample)
    decoded = myDecoder(latent)

    print("Latent: ", latent)
    print("Latent: ", latent.shape)
    print("Latent: ", latent.dtype)

    print("Decoded: ", decoded)
    print("Decoded: ", decoded.shape)
    print("Decoded: ", decoded.dtype)

    print("MSE:", torch.mean((decoded - sample).pow(2)))

    full_fwd_pass = model(sample)

    print("MSE for full pass:", torch.mean((full_fwd_pass[0] - sample).pow(2)))
