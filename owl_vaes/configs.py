import os
from dataclasses import dataclass

import yaml
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("env", lambda k: os.environ.get(k))

@dataclass
class VAEConfig:
    model_id : str = None

    sample_size : int = 512
    channels : int = 3
    latent_size : int = 16 # for 1d this is tokens, for 2d it is h/w
    latent_channels : int = 32

    noise_decoder_inputs : float = 0.0

    # For VQ
    codebook_size : int = 1024
    codebook_dim : int = 8

    # For proxy
    proxy_size : int = 16
    proxy_channels : int = 16

@dataclass
class ResNetConfig(VAEConfig):
    ch_0 : int = 256
    ch_max : int = 2048

    encoder_blocks_per_stage : list = None
    decoder_blocks_per_stage : list = None

    attn_size : int = None  # When size is less than this, attention will be used

@dataclass
class TransformerConfig(VAEConfig):
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384

    patch_size : int = 1
    causal: bool = False
    mimetic_init: bool = True

    # Proxy stuff
    proxy_patch_size : int = 1

@dataclass
class AudioConfig(VAEConfig):
    in_channels: int = 2
    out_channels: int = 2
    channels: int = 128
    latent_dim: int = 32

    c_mults: list = None  # Channel multipliers per stage
    strides: list = None  # Downsampling strides

    use_snake: bool = False
    antialias_activation: bool = False
    use_nearest_upsample: bool = False
    final_tanh: bool = True

    def __post_init__(self):
        if self.c_mults is None:
            self.c_mults = [1, 2, 4, 8]
        if self.strides is None:
            self.strides = [2, 4, 8, 8]


@dataclass
class TrainingConfig:
    trainer_id : str = None
    data_id : str = None
    filepath : str = None  # For audio data path

    target_batch_size : int = 128
    batch_size : int = 2

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : dict = None
    d_opt_kwargs : dict = None # Only for GAN

    loss_weights : dict = None

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : str = None

    # Distillation related
    teacher_ckpt : str = None
    teacher_cfg : str = None

    sample_interval : int = 1000
    save_interval : int = 1000

    # Adversarial realted
    delay_adv: int = 20000
    warmup_adv:int = 5000

    # Causal regularization
    warmup_crt:int = 1000

    # For distillation, if you want to renormalize latents, scale by this amount before decode
    latent_scale:float = 1.0

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None

@dataclass
class Config:
    model: VAEConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))
