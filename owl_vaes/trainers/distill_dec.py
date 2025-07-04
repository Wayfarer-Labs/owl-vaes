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
from ..utils.logging import LogHelper, to_wandb, to_wandb_depth, to_wandb_flow
from .base import BaseTrainer
from ..configs import Config
from ..losses.dwt import dwt_loss_fn

class DistillDecTrainer(BaseTrainer):
    """
    Trainer for distilling the decoder, with frozen encoder.
    Does L2 + LPIPS + GAN + Feature Matching + R1/R2 regularization

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
        self.teacher_decoder = teacher.decoder

        model_id = self.model_cfg.model_id
        model = get_model_cls(model_id)(self.model_cfg)
        del model.encoder
        self.model = model.decoder

        disc_cfg = self.model_cfg.discriminator
        self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            print(f"Model parameters: {model_params:,}")
            print(f"Discriminator parameters: {disc_params:,}")

        self.ema = None
        self.opt = None
        self.d_opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'ema' : self.ema.state_dict(),
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
        self.discriminator.load_state_dict(save_dict['discriminator'])
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        lpips_weight = self.train_cfg.loss_weights.get('lpips', 0.0)
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)
        r12_weight = self.train_cfg.loss_weights.get('r12', 0.0)
        feature_matching_weight = self.train_cfg.loss_weights.get('feature_matching', 5.0)
        dwt_weight = self.train_cfg.loss_weights.get('dwt', 1.0)
        l1_weight = self.train_cfg.loss_weights.get('l1', 1.0)

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)
        
        self.discriminator = self.discriminator.to(self.device).train()
        if self.world_size > 1:
            self.discriminator = DDP(self.discriminator)
        freeze(self.discriminator)

        lpips = None
        if lpips_weight > 0.0:
            lpips = get_lpips_cls(self.train_cfg.lpips_id)(self.device).to(self.device).eval()
            freeze(lpips)

        self.encoder = self.encoder.to(self.device).bfloat16().eval()
        freeze(self.encoder)
        self.encoder = torch.compile(self.encoder, mode='max-autotune',dynamic=False,fullgraph=True)
        self.teacher_decoder = self.teacher_decoder.to(self.device).bfloat16().eval()
        freeze(self.teacher_decoder)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        opt_cls = getattr(torch.optim, self.train_cfg.opt)

        self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)
        self.d_opt = opt_cls(self.discriminator.parameters(), **self.train_cfg.opt_kwargs)

        if self.train_cfg.scheduler is not None:
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

        def warmup_weight():
            if self.total_step_counter < self.train_cfg.delay_adv:
                return 0.0
            else:
                x = (self.total_step_counter - self.train_cfg.delay_adv) / self.train_cfg.warmup_adv
                x = max(0.0, min(1.0, x))
                # Cosine annealing from 0 to 1
                ramp = 0.5 * (1 - torch.cos(torch.tensor(x * torch.pi)).item())
                return ramp

        def warmup_gan_weight(): return warmup_weight() * gan_weight
        def warmup_fm_weight(): return warmup_weight() * feature_matching_weight

        def fm_reduce(x, y):
            return abs(x - y).mean()

        def merged_d_losses(d, x_fake, x_real, sigma=0.01):
            fake_scores, _ = d(x_fake.detach())
            real_scores, _ = d(x_real.detach())

            fake_scores_noisy, _ = d((x_fake + sigma*torch.randn_like(x_fake)).detach())
            real_scores_noisy, _ = d((x_real + sigma*torch.randn_like(x_real)).detach())

            r1_penalty = 0.
            r2_penalty = 0.
            disc_loss = 0.

            for fake_out, real_out, fake_noisy, real_noisy in zip(
                fake_scores, real_scores, fake_scores_noisy, real_scores_noisy
            ):
                r1_penalty += (fake_noisy - fake_out).pow(2).mean()
                r2_penalty += (real_noisy - real_out).pow(2).mean()
                
                disc_loss += F.relu(1 + fake_out).mean() + F.relu(1 - real_out).mean()


            r1_penalty = r1_penalty / len(fake_scores)
            r2_penalty = r2_penalty / len(fake_scores) 
            disc_loss = disc_loss / len(fake_scores)

            return r1_penalty, r2_penalty, disc_loss
        
        def d_loss(d, x_fake, x_real):
            fake_scores, _ = d(x_fake.detach())
            real_scores, _ = d(x_real.detach())

            disc_loss = 0.
            for fake_out, real_out in zip(fake_scores, real_scores):
                disc_loss += F.relu(1 + fake_out).mean() + F.relu(1 - real_out).mean()

            disc_loss = disc_loss / len(fake_scores)
            return disc_loss

        def g_loss(d, x_fake, x_real):
            fake_scores, fake_features = d(x_fake)
            _, real_features = d(x_real)

            gan_loss = 0.
            fm_loss = 0.

            for fake_out in fake_scores:
                gan_loss += -fake_out.mean()
            gan_loss = gan_loss / len(fake_scores)

            for h_fake, h_real in zip(fake_features, real_features):
                fm_loss += sum(map(fm_reduce, h_fake, h_real)) / len(h_fake)
            fm_loss = fm_loss / len(fake_features)

            return gan_loss, fm_loss

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                with ctx:
                    with torch.no_grad():
                        teacher_z = self.encoder(batch) / self.train_cfg.latent_scale
                        teacher_z = teacher_z + torch.randn_like(teacher_z) * 0.01
                        batch = batch[:,:3]

                    batch_rec = self.model(teacher_z)

                    # Discriminator training - RGB only
                    unfreeze(self.discriminator)
                    if r12_weight == 0.0:
                        disc_loss = d_loss(self.discriminator, batch_rec[:,:3].detach(), batch[:,:3].detach()) / accum_steps
                        metrics.log('disc_loss', disc_loss)
                    else:
                        r1_penalty, r2_penalty, d_loss = merged_d_losses(
                            self.discriminator, 
                            batch_rec[:,:3].detach(), 
                            batch[:,:3].detach()
                        )
                        r1_penalty = r1_penalty / accum_steps
                        r2_penalty = r2_penalty / accum_steps
                        d_loss = d_loss / accum_steps

                        metrics.log('r1_penalty', r1_penalty)
                        metrics.log('r2_penalty', r2_penalty)
                        metrics.log('disc_loss', d_loss)

                        disc_loss = d_loss + (r1_penalty + r2_penalty) * 0.5 * r12_weight

                    self.scaler.scale(disc_loss).backward()
                    freeze(self.discriminator)

                    mse_loss = F.mse_loss(batch_rec, batch) / accum_steps
                    total_loss += mse_loss
                    metrics.log('mse_loss', mse_loss)

                    if lpips_weight > 0.0:
                        with ctx:
                            lpips_loss = lpips(batch_rec[:,:3], batch[:,:3]) / accum_steps
                        total_loss += lpips_loss * lpips_weight
                        metrics.log('lpips_loss', lpips_loss)
                    
                    if dwt_weight > 0.0:
                        with ctx:
                            dwt_loss = dwt_loss_fn(batch_rec[:,:3], batch[:,:3]) / accum_steps
                        total_loss += dwt_loss * dwt_weight
                        metrics.log('dwt_loss', dwt_loss)
                    
                    crnt_gan_weight = warmup_gan_weight()
                    crnt_fm_weight = warmup_fm_weight()
                    if crnt_gan_weight > 0.0:
                        with ctx:
                            gan_loss, fm_loss = g_loss(self.discriminator, batch_rec[:,:3], batch[:,:3])
                            gan_loss = gan_loss / accum_steps
                            fm_loss = fm_loss / accum_steps
                        metrics.log('gan_loss', gan_loss)
                        metrics.log('fm_loss', fm_loss)
                        total_loss += crnt_gan_weight * gan_loss + crnt_fm_weight * fm_loss

                    self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    self.scaler.unscale_(self.d_opt)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    self.scaler.step(self.d_opt)
                    self.d_opt.zero_grad(set_to_none=True)
                    
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
                            with ctx:
                                ema_rec = self.ema.ema_model(teacher_z)
                                teacher_rec = self.teacher_decoder(teacher_z)[:,:3]

                            wandb_dict['samples'] = to_wandb(
                                teacher_rec.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )

                            # Log depth maps if present (4 or 7 channels)
                            if batch.shape[1] >= 4:
                                depth_samples = to_wandb_depth(
                                    teacher_rec.detach().contiguous().bfloat16(),
                                    ema_rec.detach().contiguous().bfloat16(),
                                    gather = False
                                )
                                if depth_samples:
                                    wandb_dict['depth_samples'] = depth_samples
                            
                            # Log optical flow if present (7 channels)
                            if batch.shape[1] >= 7:
                                flow_samples = to_wandb_flow(
                                    teacher_rec.detach().contiguous().bfloat16(),
                                    ema_rec.detach().contiguous().bfloat16(),
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

