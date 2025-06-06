"""
Trainer for proxy models
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..configs import Config
from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer, versatile_load
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer

def latent_reg_loss(z):
    # z is [b,c,h,w]
    # KL divergence between N(z, 0.1) and N(0,1)
    mu = z
    logvar = 2 * torch.log(torch.tensor(0.1))  # log(0.1^2)

    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = eo.reduce(kl, 'b ... -> b', reduction='sum').mean()
    return kl

class ProxyTrainer(BaseTrainer):
    """
    Trainer for reconstruction only objective.
    Only does L2 + LPIPS

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

        teacher_cfg = Config.from_yaml(self.train_cfg.teacher_cfg).model
        teacher_ckpt = versatile_load(self.train_cfg.teacher_ckpt)
        self.teacher = get_model_cls(teacher_cfg.model_id)(teacher_cfg)
        self.teacher.load_state_dict(teacher_ckpt)

        if self.rank == 0:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {n_params:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scheduler' : self.scheduler.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is not None:
            save_dict = super().load(self.train_cfg.resume_ckpt)
        else:
            return

        self.model.load_state_dict(save_dict['model'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        reg_weight =  self.train_cfg.loss_weights.get('latent_reg', 0.0)

        # Prepare model, lpips, ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.teacher = self.teacher.eval().cuda().bfloat16()
        self.teacher.encoder = torch.compile(self.teacher.encoder)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(f'cuda:{self.local_rank}',torch.bfloat16)

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to('cuda').bfloat16()

                with ctx:
                    batch_rec, z = self.model(batch)

                with torch.no_grad(), ctx:
                    z_teacher = self.teacher.encoder(batch.bfloat16())

                if reg_weight > 0:
                    reg_loss = latent_reg_loss(z) / accum_steps
                    total_loss += reg_loss * reg_weight
                    metrics.log('reg_loss', reg_loss)

                mse_loss = F.mse_loss(batch_rec, z_teacher) / accum_steps
                total_loss += mse_loss
                metrics.log('mse_loss', mse_loss)

                self.scaler.scale(total_loss).backward()

                with torch.no_grad():
                    metrics.log_dict({
                        'z_std' : z.std() / accum_steps,
                        'z_shift' : z.mean() / accum_steps
                    })

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad()

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                batch_rec = self.teacher.decoder(batch_rec.bfloat16())
                            wandb_dict['samples'] = to_wandb(
                                batch.detach().contiguous().bfloat16(),
                                batch_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )
                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
