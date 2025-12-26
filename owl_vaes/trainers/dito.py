"""
Trainer for distilling decoder with adversarial loss
Combines image and audio training approaches with feature matching
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy

from ..data import get_loader
from ..models import get_model_cls
from ..discriminators import get_discriminator_cls
from ..muon import init_muon
from ..nn.lpips import get_lpips_cls
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb, to_wandb_grayscale
from .base import BaseTrainer
from ..configs import Config
from ..sampling import flow_sample, x0_sample

class DiToTrainer(BaseTrainer):
    """
    Trainer for diffusion tokenizer

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)
        self.model_input_size = [int(self.model_cfg.sample_size[0]), int(self.model_cfg.sample_size[1])]

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {model_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

        self.use_proxy = getattr(self.train_cfg, "use_proxy", False)
        if self.use_proxy:
            from diffusers import AutoModel
            self.proxy = AutoModel.from_pretrained("madebyollin/taef1")
            self.proxy = self.proxy.to(self.device).bfloat16()
            self.proxy.encoder = torch.compile(self.proxy.encoder)

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            return
        
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']
    
    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module.decoder
        return self.ema.ema_model.decoder

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model, find_unused_parameters=True)

        self.ema = EMA(
            self.model,
            beta = 0.995,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        opt_cls = getattr(torch.optim, self.train_cfg.opt)

        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)

        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(self.device, torch.bfloat16)

        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, rank=self.rank, world_size=self.world_size, **self.train_cfg.data_kwargs)

        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()
                batch = F.interpolate(batch, self.model_input_size, mode='bilinear', align_corners=False)
                proxy_batch = self.proxy.encoder(batch) if self.use_proxy else None

                with ctx:
                    diff_loss, z = self.model(batch, proxy = proxy_batch)

                metrics.log('diff_loss', diff_loss)
                total_loss += diff_loss
                self.scaler.scale(total_loss).backward()

                # Updates
                if self.train_cfg.opt.lower() != "muon":
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.opt)
                self.opt.zero_grad(set_to_none=True)
                
                self.scaler.update()
                self.ema.update()

                # Do logging stuff with sampling stuff in the middle
                with torch.no_grad():
                    wandb_dict = metrics.pop()
                    wandb_dict['time'] = timer.hit()
                    timer.reset()

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        with ctx:
                            sample_fn = x0_sample if self.model_cfg.x0_mode else flow_sample
                            ema_rec = sample_fn(
                                self.get_ema_core(),
                                proxy_batch.detach() if self.use_proxy else batch.detach(),
                                z.detach(),
                                self.train_cfg.sampling_steps,
                                self.proxy.decoder if self.use_proxy else None
                            )

                        wandb_dict['samples'] = to_wandb(
                            batch.detach().contiguous().bfloat16(),
                            ema_rec.detach().contiguous().bfloat16(),
                            gather = False
                        )

                    if self.rank == 0:
                        wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    if self.rank == 0:
                        self.save()

                self.barrier()

