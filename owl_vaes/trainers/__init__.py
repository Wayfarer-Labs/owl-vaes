from typing import Literal

from .audio_rec import AudioRecTrainer
from .proxy import ProxyTrainer
from .rec import RecTrainer
from .decoder_tune import DecTuneTrainer
from .diffdec_trainer import DiffusionDecoderTrainer

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
            return DiffusionDecoderTrainer
        case "dec_tune_v2":
            from .dec_tune_v2 import DecTuneV2Trainer
            return DecTuneV2Trainer
        case "distill_dec":
            from .distill_dec import DistillDecTrainer
            return DistillDecTrainer
        case _:
            raise NotImplementedError
