from typing import Any

def get_model_cls(model_id: str) -> Any:
    if model_id == "dcae":
        from .dcae import DCAE
        return DCAE
    if model_id == "diff_dec":
        from .diffdec import DiffusionDecoder
        return DiffusionDecoder
    if model_id == "causal_diffdec":
        from .causal_diffdec import CausalDiffusionDecoder
        return CausalDiffusionDecoder
    if model_id == "distill_vae":
        from .distill_vae import DistillVAE
        return DistillVAE
    #if model_id == "causal_audio":
    #    from .causal_audio import CausalAudioVAE
    #    return CausalAudioVAE
    if model_id == "video_dcae":
        from .video_dcae import VideoDCAE
        return VideoDCAE
    if model_id == "dito":
        from .dito import DiTo
        return DiTo
    return None
