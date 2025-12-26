from typing import Literal

from .audio_rec import AudioRecTrainer
from .proxy import ProxyTrainer
from .rec import RecTrainer
from .decoder_tune import DecTuneTrainer

def get_trainer_cls(trainer_id: Literal["rec", "proxy", "audio_rec"]):
    match trainer_id:
        case "rec":
            return RecTrainer
        #case "audio_rec":
        #    return AudioRecTrainer
        #case "audio_dec_tune":
        #    from .audio_decoder_tune import AudDecTuneTrainer
        #    return AudDecTuneTrainer
        case "diff_dec":
            from .diffdec_trainer import DiffusionDecoderTrainer
            return DiffusionDecoderTrainer
        case "distill_dec":
            from .distill_dec import DistillDecTrainer
            return DistillDecTrainer
        case "distill_enc":
            from .distill_enc import DistillEncTrainer
            return DistillEncTrainer
        #case "diffdec_ode_tune":
        #    from .diffdec_ode_tune import DiffDecODETrainer
        #    return DiffDecODETrainer
        #case "diffdec_dmd":
        #    from .diffdec_dmd_trainer import DiffDMDTrainer
        #    return DiffDMDTrainer
        case "caus_diffdec":
            from .caus_diffdec_trainer import CausalDiffusionDecoderTrainer
            return CausalDiffusionDecoderTrainer
        case "distill_pretrained_enc":
            from .distill_pretrained_enc import DistillPretrainedEncTrainer
            return DistillPretrainedEncTrainer
        case "proxy_diffdec":
            from .proxy_diffdec import DiffDecLiveDepthTrainer
            return DiffDecLiveDepthTrainer
        case "video_rec":
            from .video_rec import VideoRecTrainer
            return VideoRecTrainer
        case "dito":
            from .dito import DiToTrainer
            return DiToTrainer
        case _:
            raise NotImplementedError
