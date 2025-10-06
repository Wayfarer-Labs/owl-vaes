from typing import Literal

from .audio_rec import AudioRecTrainer
from .proxy import ProxyTrainer
from .rec import RecTrainer
from .decoder_tune import DecTuneTrainer

def get_trainer_cls(trainer_id: Literal["rec", "proxy", "audio_rec"]):
    match trainer_id:
        case "rec":
            return RecTrainer
        case "proxy":
            return ProxyTrainer
        case "audio_rec":
            return AudioRecTrainer
        case "dec_tune":
            return DecTuneTrainer
        case "audio_dec_tune":
            from .audio_decoder_tune import AudDecTuneTrainer
            return AudDecTuneTrainer
        case "diff_dec":
            from .diffdec_trainer import DiffusionDecoderTrainer
            return DiffusionDecoderTrainer
        case "dec_tune_v2":
            from .dec_tune_v2 import DecTuneV2Trainer
            return DecTuneV2Trainer
        case "distill_dec":
            from .distill_dec import DistillDecTrainer
            return DistillDecTrainer
        case "distill_enc":
            from .distill_enc import DistillEncTrainer
            return DistillEncTrainer
        case "diffdec_ode_tune":
            from .diffdec_ode_tune import DiffDecODETrainer
            return DiffDecODETrainer
        case "diffdec_dmd":
            from .diffdec_dmd_trainer import DiffDMDTrainer
            return DiffDMDTrainer
        case "dcae_tune":
            from .distill_dcae_test import DCAETuneTrainer
            return DCAETuneTrainer
        case "caus_diffdec":
            from .caus_diffdec_trainer import CausalDiffusionDecoderTrainer
            return CausalDiffusionDecoderTrainer
        case "caus_distill_dec":
            from .caus_distill_dec import CausalDistillDecoderTrainer
            return CausalDistillDecoderTrainer
        case _:
            raise NotImplementedError
