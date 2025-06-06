"""
Audio Reconstruction Trainer for Audio AutoEncoder - Optimized with torch.compile
"""

import einops
import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer
from ..utils.logging import LogHelper, log_audio_to_wandb
from .base import BaseTrainer

from ..losses.audio import (
    stft_loss,
    compute_ms_loss
)
from ..losses.basic import (
    latent_reg_loss
)

@torch.compile(mode="max-autotune")
def compute_forward_pass(model, batch):
    """
    Compiled forward pass function
    """
    return model(batch)

def compute_losses(
    batch_rec: Tensor,
    batch: Tensor,
    z: Tensor,
    recon_weight: float,
    stft_weight: float,
    kl_weight: float,
    lr_ms_ratio: float,
    accum_steps: int,
    n_fft_list: list[int],
):
    """
    Compiled loss computation function
    """
    total_loss = torch.tensor(0.0, device=batch.device)
    metrics = {}

    # Reconstruction loss
    if recon_weight > 0:
        recon_loss = F.mse_loss(batch_rec, batch) / accum_steps
        total_loss = total_loss + recon_loss * recon_weight
        metrics["recon_loss"] = recon_loss

    # STFT loss
    if stft_weight > 0:
        stft_l = stft_loss(batch_rec, batch, n_fft_list=n_fft_list) / accum_steps
        total_loss = total_loss + stft_l * stft_weight
        metrics["stft_loss"] = stft_l

    # M/S loss for stereo audio
    if lr_ms_ratio < 1.0 and batch.size(1) == 2:
        ms_loss = compute_ms_loss(batch_rec, batch) / accum_steps
        ms_weight = (1.0 + lr_ms_ratio) / 2.0
        total_loss = total_loss + ms_loss * ms_weight * recon_weight
        metrics["ms_loss"] = ms_loss

    # KL loss
    if kl_weight > 0:
        kl_loss = latent_reg_loss(z) / accum_steps
        total_loss = total_loss + kl_loss * kl_weight
        metrics["kl_loss"] = kl_loss

    # Z statistics
    metrics.update({
        "z_std": z.std(),
        "z_mean": z.mean(),
        "z_max": z.max(),
        "z_min": z.min(),
    })

    return total_loss, metrics


@torch.compile(mode="max-autotune")
def optimization_step(scaler, opt, model_parameters, scheduler=None):
    """
    Compiled optimization step
    """
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=1.0)
    scaler.step(opt)
    opt.zero_grad(set_to_none=True)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

class AudioRecTrainer(BaseTrainer):
    """
    Trainer for audio reconstruction with specialized audio losses

    Includes:
    - Reconstruction loss (MSE)
    - STFT loss (multi-scale spectral loss) - COMPILED
    - TODO: HuBERT perceptual loss)
    - KL divergence regularization (down-weighted by 1e-4) - COMPILED
    - L/R vs M/S component weighting (L/R down-weighted by 0.5) - COMPILED
    - Audio logging to W&B
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model: nn.Module = get_model_cls(model_id)(self.model_cfg).to(self.device)

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def _compile_model(self, model: nn.Module) -> nn.Module:
        print("Compiling model...")
        return torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)  # type: ignore

    def save(self):
        # Save the original model state, not the compiled one
        save_dict = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.total_step_counter,
        }
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scheduler.load_state_dict(save_dict["scheduler"])
        self.scaler.load_state_dict(save_dict["scaler"])
        self.total_step_counter = save_dict["steps"]

    def train(self):
        # Loss weights
        recon_weight = self.train_cfg.loss_weights.get("recon", 1.0)
        stft_weight = self.train_cfg.loss_weights.get("stft", 1.0)
        hubert_weight = self.train_cfg.loss_weights.get("hubert", 0.0)  # Placeholder
        kl_weight = self.train_cfg.loss_weights.get("kl", 1e-4)
        lr_ms_ratio = self.train_cfg.loss_weights.get(
            "lr_ms_ratio", 0.5
        )  # L/R weight relative to M/S

        # Audio-specific config
        sample_rate = getattr(self.train_cfg, "sample_rate", 44100)
        n_fft_list = getattr(self.train_cfg, "n_fft_list", [1024, 2048, 512])

        # Prepare model and EMA
        self.model = self.model.to(self.device).train()

        if self.world_size > 1:
            self.model = DDP(self.model)

        # Compile model after DDP setup
        self.model = self._compile_model(self.model)

        self.ema = EMA(self.model, beta=0.9999, update_after_step=0, update_every=1)

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(
                self.model,
                rank=self.rank,
                world_size=self.world_size,
                **self.train_cfg.opt_kwargs,
            )
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(
                self.model.parameters(), **self.train_cfg.opt_kwargs
            )

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(
                self.opt, **self.train_cfg.scheduler_kwargs
            )

        # Grad accum setup and scaler
        accum_steps = (
            self.train_cfg.target_batch_size
            // self.train_cfg.batch_size
            // self.world_size
        )
        accum_steps = max(1, accum_steps)
        self.scaler = torch.GradScaler()
        ctx = torch.autocast(self.device, torch.bfloat16)

        # Timer and metrics setup
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log="all")

        # Dataset setup
        loader = get_loader(
            self.train_cfg.data_id, self.train_cfg.batch_size, self.train_cfg.filepath
        )

        local_step = 0
        for epoch_idx in range(self.train_cfg.epochs):
            for batch in tqdm(
                loader, desc=f"Epoch {epoch_idx + 1}/{self.train_cfg.epochs}"
            ):
                batch = batch[0].to(self.device, dtype=torch.bfloat16)

                if batch.is_cuda:
                    torch.compiler.cudagraph_mark_step_begin()

                with ctx:
                    # Forward pass using compiled function
                    batch_rec, z = compute_forward_pass(self.model, batch)

                    # Compute all losses using compiled function
                    total_loss, loss_metrics = compute_losses(
                        batch_rec=batch_rec,
                        batch=batch,
                        z=z,
                        recon_weight=recon_weight,
                        stft_weight=stft_weight,
                        kl_weight=kl_weight,
                        lr_ms_ratio=lr_ms_ratio,
                        accum_steps=accum_steps,
                        n_fft_list=n_fft_list,
                    )

                # Update metrics
                metrics.log_dict(loss_metrics)

                # Backward pass
                self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Optimization step using compiled function
                    optimization_step(
                        self.scaler, self.opt, self.model.parameters(), self.scheduler
                    )

                    # EMA update (not compiled as it has its own optimized implementation)
                    self.ema.update()

                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict["time"] = timer.hit()
                        wandb_dict["lr"] = self.opt.param_groups[0]["lr"]
                        timer.reset()

                        # Audio logging
                        if (
                            self.total_step_counter % self.train_cfg.sample_interval
                            == 0
                        ):
                            audio_logs = log_audio_to_wandb(
                                batch.detach().contiguous().float(),
                                batch_rec.detach().contiguous().float(),
                                sample_rate=sample_rate,
                                max_samples=2,
                            )
                            wandb_dict.update(audio_logs)

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
