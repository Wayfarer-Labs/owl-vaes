import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..discriminators import get_discriminator_cls
from ..nn.lpips import get_lpips_cls
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb_video_sidebyside
from .base import BaseTrainer
from ..configs import Config
from ..sampling.causdec import CausDecSampler

import gc

from ..losses.gan import (
    merged_d_losses,
    d_loss,
    g_loss,
    rec_d_loss,
    rec_g_loss,
)

class CausalDistillDecoderTrainer(BaseTrainer):
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

        if hasattr(self.train_cfg, 'encoder_cfg') and hasattr(self.train_cfg, 'encoder_ckpt'):
            encoder_cfg = Config.from_yaml(self.train_cfg.encoder_cfg).model
            encoder_ckpt = versatile_load(self.train_cfg.encoder_ckpt)
            encoder = get_model_cls(encoder_cfg.model_id)(encoder_cfg)
            try:
                encoder.load_state_dict(encoder_ckpt)
            except:
                encoder.encoder.load_state_dict(encoder_ckpt)
            encoder.encoder.skip_logvar = False
            encoder.encoder.conv_out_logvar = self.encoder.conv_out_logvar
            self.encoder = encoder.encoder

        model_id = self.model_cfg.model_id
        model = get_model_cls(model_id)(self.model_cfg)
        del model.encoder
        self.model = model.decoder

        # Only create discriminator if GAN is used
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.1)

        if gan_weight > 0.0:
            disc_cfg = self.model_cfg.discriminator
            self.discriminator = get_discriminator_cls(disc_cfg.model_id)(disc_cfg)
            self.disc_2 = get_discriminator_cls("patchgan")(disc_cfg)
            self.use_rec_disc = (self.model_cfg.discriminator.model_id == 'recgan')
        else:
            self.discriminator = None
            self.disc_2 = None

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
        self.d_2_opt = None
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
        gan_weight = self.train_cfg.loss_weights.get('gan', 0.5)
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
        
        if self.disc_2 is not None:
            self.disc_2 = self.disc_2.to(self.device).train()
            if self.world_size > 1:
                self.disc_2 = DDP(self.disc_2)
            freeze(self.disc_2)

        lpips = None
        if lpips_weight > 0.0:
            lpips = get_lpips_cls(self.train_cfg.lpips_type)(self.device).to(self.device).eval()
            freeze(lpips)

        self.encoder = self.encoder.to(self.device).bfloat16()
        freeze(self.encoder)
        self.encoder = torch.compile(self.encoder)#, mode='max-autotune',dynamic=False,fullgraph=True)
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
        if self.discriminator is not None:
            self.d_opt = opt_cls(self.discriminator.parameters(), **self.train_cfg.opt_kwargs)

        if self.disc_2 is not None:
            self.d_2_opt = opt_cls(self.disc_2.parameters(), **self.train_cfg.opt_kwargs)

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
        sample_loader = get_loader(
            "procedural_rgb",
            1,
            root_dir="/mnt/data/waypoint_1/data/MKIF_360P",
            output_dir="/mnt/data/waypoint_1/data_pt/MKIF_360P",
            window_size=32,
            target_size=[512,512]
        )
        sample_loader = iter(sample_loader)
        sampler = CausDecSampler()

        def pack(x):
            # b n c h w -> (b*n) c h w
            b, n, c, h, w = x.shape
            return x.contiguous().view(b * n, c, h, w)
        
        def unpack(x, b=self.train_cfg.target_batch_size):
            # (b*n) c h w -> b n c h w
            bn, c, h, w = x.shape
            n = bn // b
            return x.contiguous().view(b, n, c, h, w)

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
        
        @torch.no_grad()
        def sample_from_teacher(batch_rgb):
            orig_b = batch_rgb.shape[0]
            batch_rgb = batch_rgb.cuda().bfloat16()
            batch_rgb = pack(batch_rgb)
            # convert to [bt,]
            t_mu, t_logvar = self.encoder(batch_rgb)
            t_std = (t_logvar/2).exp()
            t_z = torch.randn_like(t_mu) * t_std + t_mu
            t_z = t_z / self.train_cfg.latent_scale
            return unpack(t_z, orig_b)

        def video_interpolate(vid, target_size, mode = 'bilinear', align_corners = False):
            b,t,c,h,w = vid.shape
            vid = vid.reshape(b*t, c, h, w)
            vid = F.interpolate(vid, size=target_size, mode=mode, align_corners=align_corners)
            vid = vid.reshape(b, t, c, *vid.shape[2:])
            return vid

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()
                batch = video_interpolate(batch, self.model_input_size, mode='bilinear', align_corners=False)

                with ctx:
                    with torch.no_grad():
                        teacher_z = sample_from_teacher(batch)

                    batch_rec = self.model(teacher_z)

                    # Discriminator training - RGB only
                    if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        unfreeze(self.discriminator)
                        disc_loss = d_loss(self.discriminator, batch_rec.detach(), batch.detach()) / accum_steps
                        metrics.log('disc_loss', disc_loss)

                        self.scaler.scale(disc_loss).backward()
                        freeze(self.discriminator)
                    
                    if self.disc_2 is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        unfreeze(self.disc_2)
                        disc_loss = d_loss(self.disc_2, pack(batch_rec.detach()), pack(batch.detach())) / accum_steps
                        metrics.log('disc_loss_2d', disc_loss)
                        self.scaler.scale(disc_loss).backward()
                        freeze(self.disc_2)

                    if l2_weight > 0.0:
                        l2_loss = F.mse_loss(batch_rec, batch) / accum_steps
                        total_loss += l2_loss * l2_weight
                        metrics.log('l2_loss', l2_loss)

                    if lpips_weight > 0.0:
                        with ctx:
                            lpips_loss = lpips(pack(batch_rec), pack(batch)) / accum_steps
                        total_loss += lpips_loss * lpips_weight
                        metrics.log('lpips_loss', lpips_loss)
                    
                    if self.discriminator is not None:
                        crnt_gan_weight = warmup_gan_weight()
                        if crnt_gan_weight > 0.0:
                            with ctx:
                                gan_loss = g_loss(self.discriminator, batch_rec)
                                if self.disc_2 is not None:
                                    gan_loss_2 = g_loss(self.disc_2, pack(batch_rec))
                                    gan_loss = gan_loss + gan_loss_2
                                gan_loss = gan_loss / accum_steps
                            metrics.log('gan_loss', gan_loss)
                            total_loss += crnt_gan_weight * gan_loss

                    self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    #self.scaler.unscale_(self.opt)
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    #if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                    #    self.scaler.unscale_(self.d_opt)
                    #    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

                    #if self.disc_2 is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        #self.scaler.unscale_(self.d_2_opt)
                        #torch.nn.utils.clip_grad_norm_(self.disc_2.parameters(), max_norm=1.0)

                    self.scaler.step(self.opt)
                    self.opt.zero_grad(set_to_none=True)

                    if self.discriminator is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        self.scaler.step(self.d_opt)
                        self.d_opt.zero_grad(set_to_none=True)

                    if self.disc_2 is not None and self.total_step_counter >= self.train_cfg.delay_adv:
                        self.scaler.step(self.d_2_opt)
                        self.d_2_opt.zero_grad(set_to_none=True)

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
                            sample_rgb = next(sample_loader) # bnchw
                            sample_rgb = sample_rgb.cuda().bfloat16()
                            sample_rgb = video_interpolate(sample_rgb, (512,512), mode='bilinear', align_corners=False)
                            teacher_z = sample_from_teacher(sample_rgb)
                            with ctx:
                                ema_rec = sampler(self.ema.ema_model, teacher_z, window_size = 4)

                            if self.world_size > 1:
                                sample_rgb_list = [torch.zeros_like(sample_rgb) for _ in range(self.world_size)]
                                torch.distributed.all_gather(sample_rgb_list, sample_rgb.contiguous())
                                sample_rgb = torch.cat(sample_rgb_list, dim=0)
                                
                                ema_rec_list = [torch.zeros_like(ema_rec) for _ in range(self.world_size)]
                                torch.distributed.all_gather(ema_rec_list, ema_rec)
                                ema_rec = torch.cat(ema_rec_list, dim=0)

                            if self.rank == 0:
                                wandb_dict['samples'] = to_wandb_video_sidebyside(
                                    sample_rgb.detach().contiguous().bfloat16(),
                                    ema_rec.detach().contiguous().bfloat16()
                                )
                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()

