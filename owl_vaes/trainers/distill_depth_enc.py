"""
Trainer for distilling encoder with simplified loss, with live depth generation
"""

import torch
import torch.nn.functional as F
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

from ema_pytorch import EMA

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..nn.lpips import get_lpips_cls
from ..utils import Timer, freeze, versatile_load
from ..utils.logging import LogHelper, to_wandb, to_wandb_grayscale
from .base import BaseTrainer
from ..configs import Config

class DistillDepthEncTrainer(BaseTrainer):
    """
    Trainer for distilling the encoder with live depth generation, with frozen decoder.
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
        self.ema = None

        self.model_input_size = [int(self.model_cfg.sample_size[0]), int(self.model_cfg.sample_size[1])]

        import sys
        sys.path.append("./FlashDepth")
        from flashdepth import FlashDepthModel
        self.depth = FlashDepthModel(
            model_size='vits',
            use_mamba=False,
            checkpoint_path='FlashDepth/configs/flashdepth/iter_43002.pth'
        )

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter,
            'ema' : self.ema.state_dict()
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
        if self.ema is not None:
            self.ema.load_state_dict(save_dict['ema'])

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
        self.teacher_encoder = torch.compile(self.teacher_encoder)#, mode='max-autotune', dynamic=False, fullgraph=True)

        self.teacher_decoder = self.teacher_decoder.to(self.device).bfloat16().eval()
        freeze(self.teacher_decoder)
        self.teacher_decoder = torch.compile(self.teacher_decoder)#, mode='max-autotune', dynamic=False, fullgraph=True)

        # Depth prep
        self.depth = self.depth.to(self.device).bfloat16()
        self.depth = torch.compile(self.depth)

        lpips = get_lpips_cls(self.train_cfg.lpips_type)(self.device).to(self.device).eval()
        freeze(lpips)

        self.ema = EMA(
            self.model,
            beta = 0.9999,
            update_after_step = 0,
            update_every = 1
        )

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

        if isinstance(self.train_cfg.latent_scale, (list, tuple)):
            self.train_cfg.latent_scale = torch.Tensor(self.train_cfg.latent_scale)

        def divide_latent_scale(latent):
            if isinstance(self.train_cfg.latent_scale, torch.Tensor):
                # latent is b c h w
                # latent scale [c,]
                shift = self.train_cfg.latent_shift.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                scale = self.train_cfg.latent_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                return (latent - shift) / scale
            else:
                return (latent - self.train_cfg.latent_shift) / self.train_cfg.latent_scale

        def multiply_latent_scale(latent):
            if isinstance(self.train_cfg.latent_scale, torch.Tensor):
                # latent is b c h w
                # latent scale [c,]
                shift = self.train_cfg.latent_shift.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                scale = self.train_cfg.latent_scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                return (latent * scale) + shift
            else:
                shift = self.train_cfg.latent_shift
                scale = self.train_cfg.latent_scale
                return (latent * scale) + shift

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                total_loss = 0.
                batch = batch.to(self.device).bfloat16()
                with torch.no_grad():
                    depth = self.depth(batch).unsqueeze(1)

                batch = torch.cat([batch, depth], dim = 1)
                full_batch = batch.clone()
                batch = F.interpolate(batch, self.model_input_size, mode='bilinear', align_corners=False)

                with ctx:
                    with torch.no_grad():
                        teacher_z = self.teacher_encoder(batch)
                        target = divide_latent_scale(teacher_z)
                    target = target.bfloat16()

                    student_z = self.model(batch)

                    # Loss computation
                    if l2_weight > 0.0:
                        l2_loss = F.mse_loss(student_z, target) / accum_steps
                        total_loss += l2_loss * l2_weight
                        metrics.log('l2_loss', l2_loss)

                    if l1_weight > 0.0:
                        l1_loss = F.l1_loss(student_z, target) / accum_steps
                        total_loss += l1_loss * l1_weight
                        metrics.log('l1_loss', l1_loss)

                    self.scaler.scale(total_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    if self.ema is not None:
                        self.ema.update()

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

                        student_z = student_z.clone()

                        #if True: # Logging LPIPs, might slow shit down
                        #    with ctx:
                        #        student_rec = self.teacher_decoder(
                        #            multiply_latent_scale(student_z)
                        #        )
                        #        lpips_loss = lpips(student_rec, batch[:,:3])
                        #        wandb_dict['lpips_loss'] = lpips_loss.item()


                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            with ctx:
                                teacher_latent = self.teacher_encoder(full_batch).clone()
                                student_latent = self.ema.ema_model(full_batch).clone()
                                student_latent = multiply_latent_scale(student_latent)

                                teacher_rec = self.teacher_decoder(teacher_latent.detach().clone()).clone()
                                student_rec = self.teacher_decoder(student_latent.detach().clone()).clone()

                            wandb_dict['samples'] = to_wandb(
                                teacher_rec[:,:3].detach().contiguous().bfloat16(),
                                student_rec[:,:3].detach().contiguous().bfloat16(),
                                gather = False
                            )

                            # Log depth maps
                            depth_samples = to_wandb_grayscale(
                                teacher_rec[:,3].unsqueeze(1).detach().contiguous().bfloat16(),
                                student_rec[:,3].unsqueeze(1).detach().contiguous().bfloat16(),
                                gather = False
                            )
                            wandb_dict['depth_samples'] = depth_samples

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
