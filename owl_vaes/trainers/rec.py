"""
Trainer for reconstruction only
"""

import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..nn.lpips import get_lpips_cls
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze
from ..utils.logging import LogHelper, to_wandb, to_wandb_depth, to_wandb_flow
from .base import BaseTrainer
from ..losses.basic import latent_reg_loss
from ..losses.dwt import dwt_loss_fn

class RecTrainer(BaseTrainer):
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

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

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
        kl_weight =  self.train_cfg.loss_weights.get('kl', 0.0)
        lpips_weight = self.train_cfg.loss_weights.get('lpips', 0.0)
        dwt_weight = self.train_cfg.loss_weights.get('dwt', 1.0)
        l1_weight = self.train_cfg.loss_weights.get('l1', 0.0)
        l2_weight = self.train_cfg.loss_weights.get('l2', 0.0)

        # Prepare model, lpips, ema
        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model)

        lpips = None
        if lpips_weight > 0.0:
            lpips = get_lpips_cls(self.train_cfg.lpips_type)(self.device).to(self.device).eval()
            freeze(lpips)

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
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(f'cuda:{self.local_rank}', torch.bfloat16)
        
        self.load()

        # Timer reset
        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log = 'all')

        # Dataset setup
        loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()
                with ctx:
                    batch_rec, mu, logvar = self.model(batch)
                    z = mu # For logging
 
                    if kl_weight > 0.0:
                        reg_loss = latent_reg_loss(mu, logvar) / accum_steps
                        total_loss += reg_loss * kl_weight
                        metrics.log('kl_loss', reg_loss)

                    if l1_weight > 0.0:
                        l1_loss = F.l1_loss(batch_rec, batch) / accum_steps
                        total_loss += l1_loss * l1_weight
                        metrics.log('l1_loss', l1_loss)
                    
                    if l2_weight > 0.0:
                        l2_loss = F.mse_loss(batch_rec, batch) / accum_steps
                        total_loss += l2_loss * l2_weight
                        metrics.log('l2_loss', l2_loss)

                    if dwt_weight > 0.0:
                        with ctx:
                            dwt_loss = dwt_loss_fn(batch_rec[:,:3], batch[:,:3]) / accum_steps
                        total_loss += dwt_loss * dwt_weight
                        metrics.log('dwt_loss', dwt_loss)

                    if lpips_weight > 0.0:
                        lpips_loss = lpips(batch_rec[:,:3], batch[:,:3]) / accum_steps
                        total_loss += lpips_loss
                        metrics.log('lpips_loss', lpips_loss)

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
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.ema.update()

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            wandb_dict['samples'] = to_wandb(
                                batch.detach().contiguous().bfloat16(),
                                batch_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )
                            
                            # Log depth maps if present (4 or 7 channels)
                            if batch.shape[1] >= 4:
                                depth_samples = to_wandb_depth(
                                    batch.detach().contiguous().bfloat16(),
                                    batch_rec.detach().contiguous().bfloat16(),
                                    gather = False
                                )
                                if depth_samples:
                                    wandb_dict['depth_samples'] = depth_samples
                            
                            # Log optical flow if present (7 channels)
                            if batch.shape[1] >= 7:
                                flow_samples = to_wandb_flow(
                                    batch.detach().contiguous().bfloat16(),
                                    batch_rec.detach().contiguous().bfloat16(),
                                    gather = False
                                )
                                if flow_samples:
                                    wandb_dict['flow_samples'] = flow_samples
                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
