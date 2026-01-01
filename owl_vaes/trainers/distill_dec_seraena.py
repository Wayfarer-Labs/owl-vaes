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
from ..losses.dwt import dwt_loss_fn
from ..losses.gan import (
    merged_d_losses,
    d_loss,
    g_loss,
    rec_d_loss,
    rec_g_loss,
)

import random

"""
Seraena GAN training inspired by https://github.com/madebyollin/seraena
"""

class Buffer:
    def __init__(self, max_len):
        self.buff = []
        self.max_len = max_len
        self.total_pushes = 0

    @torch.no_grad()
    def push_and_pull(self, fake, z):
        # Add new batch to the buffer, but also make it so half of the batch is random
        for (fake_i, z_i) in zip(fake, z):
            if len(self.buff) > self.max_len:
                idx = random.randint(0, len(self.buff) - 1)
                self.buff[idx][0].copy_(fake_i.detach().clone())
                self.buff[idx][1].copy_(z_i.detach().clone())
            else:
                self.buff.append((fake_i.detach().clone(), z_i.detach().clone()))

        self.total_pushes += 1
        if self.total_pushes <= 4:
            return fake, z 

        # Pull half from buffer
        n_elem = len(fake) // 2
        inds = random.sample(range(len(self.buff)), n_elem)
        fake_pull = torch.stack([self.buff[i][0] for i in inds])
        z_pull = torch.stack([self.buff[i][1] for i in inds])

        return torch.cat([fake[:n_elem], fake_pull], dim=0), torch.cat([z[:n_elem], z_pull], dim=0)

def ser_d_loss(d, real, fake, z, buffer):
    fake_shuf, z_shuf = buffer.push_and_pull(fake, z)
    mask = torch.rand(len(fake), device=fake.device) < 0.5
    mask = mask.view(-1, 1, 1, 1)
    
    mask_img = mask.expand_as(fake)
    mask_z = mask.expand_as(z_shuf)

    x = mask_img * fake_shuf + ~mask_img * real
    z = mask_z * z_shuf + ~mask_z * z
    scores = d(x, z)
    targets = mask.float().mul(2).sub(1).expand_as(scores)

    return F.mse_loss(scores, targets)

def ser_g_loss(d, real, fake, z):
    # Assumes d is frozen
    def correction(d, real, fake, z):
        correction = torch.zeros_like(fake).requires_grad_(True)
        with torch.no_grad():
            real_scores = d(real, z)
        
        loss = F.mse_loss(d(fake + correction, z), real_scores.detach(), reduction = "none")
        loss = loss.mean((1,2,3), keepdim=True)
        loss.sum().backward(inputs=[correction])
        correction = correction.grad.detach().neg()
        correction.div_(correction.std(correction=0).add(1.0e-5))
        return correction

    def make_targets(d, real, fake, z):
        real, fake, z = real.detach(), fake.detach(), z.detach()
        c = correction(d, real, fake, z)
        return fake + c

    targets = make_targets(d, real, fake, z)
    return F.mse_loss(fake, targets.detach())

class SerDistillDecTrainer(BaseTrainer):
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

        self.teacher_decoder = teacher.decoder

        if hasattr(self.train_cfg, 'encoder_cfg') and hasattr(self.train_cfg, 'encoder_ckpt'):
            del teacher.encoder
            encoder_cfg = Config.from_yaml(self.train_cfg.encoder_cfg).model
            encoder_ckpt = versatile_load(self.train_cfg.encoder_ckpt)
            encoder = get_model_cls(encoder_cfg.model_id)(encoder_cfg)
            try:
                encoder.load_state_dict(encoder_ckpt)
            except:
                encoder.encoder.load_state_dict(encoder_ckpt)
            self.encoder = encoder.encoder
        else:
            self.encoder = teacher.encoder

        model_id = self.model_cfg.model_id
        model = get_model_cls(model_id)(self.model_cfg)
        del model.encoder
        self.model = model.decoder

        # Only create discriminator if GAN is used
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)

        if gan_weight > 0.0:
            disc_cfg = self.model_cfg.discriminator
            self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)
            self.use_rec_disc = (self.model_cfg.discriminator.model_id == 'recgan')
        else:
            self.discriminator = None

        # Checking for DCAE "stage 3"
        self.model_input_size = [int(self.model_cfg.sample_size[0]), int(self.model_cfg.sample_size[1])]
        

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {model_params:,}")
            if self.discriminator is not None:
                disc_params = sum(p.numel() for p in self.discriminator.parameters())
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
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.discriminator is not None:
            save_dict['discriminator'] = self.discriminator.state_dict()
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            return
        
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict['model'])
        if self.discriminator is not None and 'discriminator' in save_dict:
            try:
                self.discriminator.load_state_dict(save_dict['discriminator'])
            except:
                print("Failed to load discriminator, starting it from scratch.")
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Loss weights
        lpips_weight = self.train_cfg.loss_weights.get('lpips', 0.0)
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)
        r12_weight = self.train_cfg.loss_weights.get('r12', 0.0)
        dwt_weight = self.train_cfg.loss_weights.get('dwt', 0.0)
        l1_weight = self.train_cfg.loss_weights.get('l1', 0.0)
        l2_weight = self.train_cfg.loss_weights.get('l2', 0.0)

        # Prepare model, lpips, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)
        
        # Only setup discriminator if needed
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device).train()
            if self.world_size > 1:
                self.discriminator = DDP(self.discriminator)
            freeze(self.discriminator)

        lpips = None
        if lpips_weight > 0.0:
            lpips = get_lpips_cls(self.train_cfg.lpips_type)(self.device).to(self.device).eval()
            freeze(lpips)

        self.encoder = self.encoder.to(self.device).bfloat16()
        freeze(self.encoder)
        #self.encoder = torch.compile(self.encoder)
        self.teacher_decoder = self.teacher_decoder.to(self.device).bfloat16().eval()
        freeze(self.teacher_decoder)

        self.ema = EMA(
            self.model,
            beta = 0.995,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        opt_cls = getattr(torch.optim, self.train_cfg.opt)

        self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)
        if self.discriminator is not None:
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

        buffer = Buffer(8192)

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
        
        def teacher_sample(batch):
            mu, logvar = self.encoder(batch)
            teacher_std = (logvar/2).exp()
            teacher_z = torch.randn_like(mu) * teacher_std + mu
            teacher_z = teacher_z / self.train_cfg.latent_scale
            return teacher_z

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()

                unequal_size = (self.model_input_size != batch.shape[-2:])
                if unequal_size:
                    full_batch = batch.clone()
                    batch = F.interpolate(batch, self.model_input_size, mode='bilinear', align_corners=False)

                with ctx:
                    with torch.no_grad():
                        teacher_z = teacher_sample(batch)
                        batch = batch[:,:3]

                    batch_rec = self.model(teacher_z)

                    # Discriminator training - RGB only
                    if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        unfreeze(self.discriminator)
                        disc_loss = ser_d_loss(self.discriminator, batch_rec.detach(), batch.detach(), teacher_z.detach(), buffer) / accum_steps
                        metrics.log('disc_loss', disc_loss)

                        self.scaler.scale(disc_loss).backward()
                        freeze(self.discriminator)

                    if l2_weight > 0.0:
                        l2_loss = F.mse_loss(batch_rec, batch) / accum_steps
                        total_loss += l2_loss * l2_weight
                        metrics.log('l2_loss', l2_loss)

                    if lpips_weight > 0.0:
                        with ctx:
                            lpips_loss = lpips(batch_rec, batch) / accum_steps
                        total_loss += lpips_loss * lpips_weight
                        metrics.log('lpips_loss', lpips_loss)
                    
                    if dwt_weight > 0.0:
                        with ctx:
                            dwt_loss = dwt_loss_fn(batch_rec, batch) / accum_steps
                        total_loss += dwt_loss * dwt_weight
                        metrics.log('dwt_loss', dwt_loss)
                    
                    if self.discriminator is not None:
                        self.discriminator.eval()
                        crnt_gan_weight = warmup_gan_weight()
                        if crnt_gan_weight > 0.0:
                            gan_loss = ser_g_loss(self.discriminator, batch_rec, batch.detach(), teacher_z.detach())
                            gan_loss = gan_loss / accum_steps
                            metrics.log('gan_loss', gan_loss)
                            total_loss += crnt_gan_weight * gan_loss
                        self.discriminator.train()

                    self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        self.scaler.unscale_(self.d_opt)
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
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
                                if unequal_size:
                                    teacher_z = teacher_sample(full_batch)
                                ema_rec = self.ema.ema_model(teacher_z)
                                teacher_rec = self.teacher_decoder(teacher_z)[:,:3]

                            wandb_dict['samples'] = to_wandb(
                                teacher_rec.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16(),
                                gather = False
                            )

                            # Log depth maps if present (4 or 7 channels)
                            if batch.shape[1] >= 4:
                                depth_samples = to_wandb_grayscale(
                                    teacher_rec[:,3:4].detach().contiguous().bfloat16(),
                                    ema_rec[:,3:4].detach().contiguous().bfloat16(),
                                    gather = False
                                )
                                if depth_samples:
                                    wandb_dict['depth_samples'] = depth_samples

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()

