from typing import Any

def get_model_cls(model_id: str) -> Any:
    if model_id == "dcae":
        from .dcae import DCAE
        return DCAE
    if model_id == "titok":
        from .titok import TiToKVAE
        return TiToKVAE
    if model_id == "titok_vq":
        from .titok_vq import TiToKVQVAE
        return TiToKVQVAE
    if model_id == "dcae_vq":
        from .dcae_vq import DCVQVAE
        return DCVQVAE
    if model_id == "proxy_titok":
        from .proxy_titok import ProxyTiToKVAE
        return ProxyTiToKVAE
    if model_id == "diff_dec":
        from .diffdec import DiffusionDecoder
        return DiffusionDecoder
    if model_id == "audio_ae":
        from .oobleck import OobleckVAE
        return OobleckVAE
    if model_id == "tdec":
        from .audio_tdec import AudioTransformerDecoder
        return AudioTransformerDecoder
    return None
