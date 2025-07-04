"""
Trainer for distilling encoder with simplified loss
"""

import torch
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..utils import Timer, freeze, versatile_load
from ..utils.logging import LogHelper, to_wandb, to_wandb_depth, to_wandb_flow
from .base import BaseTrainer
from ..configs import Config

class DistillEncTrainer(BaseTrainer):
    """
    Trainer for distilling the encoder, with frozen decoder.
    Does L1/L2 regression on latents

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

        self.teacher_encoder = teacher.encoder
        self.teacher_decoder = teacher.decoder

        model_id = self.model_cfg.model_id
        model = get_model_cls(model_id)(self.model_cfg)
        del model.decoder
        self.model = model.encoder

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {model_params:,}")

        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            return
        
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict['model'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        l1_weight = self.train_cfg.loss_weights.get('l1', 0.0)
        l2_weight = self.train_cfg.loss_weights.get('l2', 1.0)

        # Prepare model
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)

        self.teacher_encoder = self.teacher_encoder.to(self.device).bfloat16().eval()
        freeze(self.teacher_encoder)
        self.teacher_encoder = torch.compile(self.teacher_encoder, mode='max-autotune', dynamic=False, fullgraph=True)
        
        self.teacher_decoder = self.teacher_decoder.to(self.device).bfloat16().eval()
        freeze(self.teacher_decoder)

        # Set up optimizer and scheduler
        opt_cls = getattr(torch.optim, self.train_cfg.opt)
        self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
            from ..schedulers import get_scheduler_cls
            self.scheduler = get_scheduler_cls(self.train_cfg.scheduler)(self.opt, **self.train_cfg.scheduler_kwargs)

        # Grad accum setup and scaler
        accum_steps = self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        accum_steps = max(1, accum_steps)

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

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                with ctx:
                    with torch.no_grad():
                        teacher_z = self.teacher_encoder(batch) / self.train_cfg.latent_scale
                    
                    student_z = self.model(batch) / self.train_cfg.latent_scale

                    # Loss computation
                    if l2_weight > 0.0:
                        l2_loss = F.mse_loss(student_z, teacher_z) / accum_steps
                        total_loss += l2_loss * l2_weight
                        metrics.log('l2_loss', l2_loss)
                    
                    if l1_weight > 0.0:
                        l1_loss = F.l1_loss(student_z, teacher_z) / accum_steps
                        total_loss += l1_loss * l1_weight
                        metrics.log('l1_loss', l1_loss)

                    self.scaler.scale(total_loss).backward()

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

                    # Do logging stuff with sampling stuff in the middle
                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict['time'] = timer.hit()
                        wandb_dict['lr'] = self.opt.param_groups[0]['lr']
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                teacher_latent = self.teacher_encoder(batch) / self.train_cfg.latent_scale
                                student_latent = self.model(batch) / self.train_cfg.latent_scale
                                
                                teacher_rec = self.teacher_decoder(teacher_latent)[:,:3]
                                student_rec = self.teacher_decoder(student_latent)[:,:3]

                            wandb_dict['samples'] = to_wandb(
                                teacher_rec.detach().contiguous().bfloat16(),
                                student_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )

                            # Log depth maps if present (4 or 7 channels)
                            if batch.shape[1] >= 4:
                                depth_samples = to_wandb_depth(
                                    teacher_rec.detach().contiguous().bfloat16(),
                                    student_rec.detach().contiguous().bfloat16(),
                                    gather = False
                                )
                                if depth_samples:
                                    wandb_dict['depth_samples'] = depth_samples
                            
                            # Log optical flow if present (7 channels)
                            if batch.shape[1] >= 7:
                                flow_samples = to_wandb_flow(
                                    teacher_rec.detach().contiguous().bfloat16(),
                                    student_rec.detach().contiguous().bfloat16(),
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
