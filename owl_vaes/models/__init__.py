from typing import Any

from owl_vaes.nn.audio import AudioAutoEncoder

from .dcae import DCAE
from .dcae_vq import DCVQVAE
from .proxy_titok import ProxyTiToKVAE
from .titok import TiToKVAE
from .titok_vq import TiToKVQVAE


def get_model_cls(model_id: str) -> Any:
    return {
        "dcae": DCAE,
        "titok": TiToKVAE,
        "titok_vq": TiToKVQVAE,
        "dcae_vq": DCVQVAE,
        "proxy_titok": ProxyTiToKVAE,
        "audio_ae": AudioAutoEncoder,
    }.get(model_id)
