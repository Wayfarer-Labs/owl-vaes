from typing import Any

def get_model_cls(model_id: str) -> Any:
    if model_id == "dcae":
        from .dcae import DCAE
        return DCAE
    if model_id == "dcae_transformer":
        from .dcae_transformer import DCAETransformer
        return DCAETransformer
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
    if model_id == "causal_diffdec":
        from .causal_diffdec import CausalDiffusionDecoder
        return CausalDiffusionDecoder
    if model_id == "meanflow_dec":
        from .meanflow import MeanFlowImage
        return MeanFlowImage
    if model_id == "audio_ae":
        from .oobleck import OobleckVAE
        return OobleckVAE
    if model_id == "tdec":
        from .audio_tdec import AudioTransformerDecoder
        return AudioTransformerDecoder
    if model_id == "distill_vae":
        from .distill_vae import DistillVAE
        return DistillVAE
    if model_id == "causal_dcae":
        from .causal_dcae import CausalDCAE
        return CausalDCAE
    if model_id == "causal_audio":
        from .causal_audio import CausalAudioVAE
        return CausalAudioVAE
    if model_id == "video_dcae":
        from .video_dcae import VideoDCAE
        return VideoDCAE
    if model_id == "dito":
        from .dito import DiTo
        return DiTo
    return None
