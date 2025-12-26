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
from ..sampling import flow_sample

class DiffDecLiveDepthTrainer(BaseTrainer):
    """
    Trainer for diffusion decoder with proxy VAE + depth encoder

    :param train_cfg: Configuration for training
    :param logging_cfg: Configuration for logging
    :param model_cfg: Configuration for model
    :param global_rank: Rank across all devices.
    :param local_rank: Rank for current device on this process.
    :param world_size: Overall number of devices
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        teacher_ckpt_path = self.train_cfg.teacher_ckpt
        teacher_cfg_path = self.train_cfg.teacher_cfg

        teacher_ckpt = versatile_load(teacher_ckpt_path)
        teacher_cfg = Config.from_yaml(teacher_cfg_path).model

        teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        teacher.load_state_dict(teacher_ckpt)

        self.encoder = teacher.encoder
        del teacher.decoder

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {model_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

        # Depth 
        import sys
        sys.path.append("./FlashDepth")
        from flashdepth import FlashDepthModel
        self.depth = FlashDepthModel(
            model_size='vits',
            use_mamba=False,
            checkpoint_path='FlashDepth/configs/flashdepth/iter_43002.pth'
        )

        # TAEF1
        from diffusers import AutoencoderTiny
        self.proxy = AutoencoderTiny.from_pretrained("madebyollin/taef1")

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
            return self.ema.ema_model.module.core
        return self.ema.ema_model.core

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model, find_unused_parameters=True)
        
        self.encoder = self.encoder.to(self.device).bfloat16()
        freeze(self.encoder)
        self.encoder = torch.compile(self.encoder)#, mode='max-autotune',dynamic=False,fullgraph=True)

        # Depth prep
        freeze(self.depth)
        self.depth = self.depth.to(self.device).bfloat16()
        self.depth = torch.compile(self.depth)

        # TAEF1 prep
        freeze(self.proxy)
        self.proxy = self.proxy.to(self.device).bfloat16()
        self.proxy.encoder = torch.compile(self.proxy.encoder)
        self.proxy.decoder = torch.compile(self.proxy.decoder)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
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
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)

        @torch.no_grad()
        def teacher_sample(batch):
            batch = F.interpolate(batch, (360, 640), mode='bilinear', align_corners=False)
            batch_depth = self.depth(batch).unsqueeze(1)
            batch = torch.cat([batch, batch_depth], dim = 1)
            mu, logvar = self.encoder(batch)
            teacher_std = (logvar/2).exp()
            teacher_z = torch.randn_like(mu) * teacher_std + mu
            teacher_z = teacher_z / self.train_cfg.latent_scale
            return teacher_z
        
        @torch.no_grad()
        def proxy_encode(batch):
            proxy_z = self.proxy.encoder(batch[:,:3])
            proxy_z = proxy_z / self.train_cfg.ldm_scale # [b,16,45,80]
            return proxy_z

        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                with ctx:
                    with torch.no_grad():
                        teacher_z = teacher_sample(batch)
                        proxy_z = proxy_encode(batch)
                        
                    diff_loss = self.model(proxy_z, teacher_z)
                    diff_loss = diff_loss

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
                            cfg_scale = getattr(self.train_cfg, 'cfg_scale', 1.0)
                            ema_rec = flow_sample(
                                self.get_ema_core(),
                                proxy_z,
                                teacher_z,
                                self.train_cfg.sampling_steps,
                                self.proxy.decoder,
                                scaling_factor = self.train_cfg.ldm_scale,
                                cfg_scale = cfg_scale
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

