import einops as eo
import torch
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data import get_loader
from ..models import get_model_cls
from ..muon import init_muon
from ..schedulers import get_scheduler_cls
from ..utils import Timer, freeze, unfreeze, versatile_load
from ..utils.logging import LogHelper, to_wandb
from .base import BaseTrainer
from ..configs import Config

from ..sampling import flow_sample

from diffusers import AutoencoderTiny, AutoencoderDC

def get_vae(vae_id):
    if vae_id == "taef1":
        return AutoencoderTiny.from_pretrained("madebyollin/taef1")
    elif vae_id == "dcae":
        return AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers")
    else:
        raise ValueError(f"VAE {vae_id} not found")

class DiffusionDecoderTrainer(BaseTrainer):
    """
    Trainer for diffusion decoder with frozen encoder.
    Does diffusion loss

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
        try:
            teacher.load_state_dict(teacher_ckpt)
        except Exception as e:
            teacher.encoder.load_state_dict(teacher_ckpt)

        self.encoder = teacher.encoder
        self.teacher_cfg = teacher_cfg
        self.teacher_size = teacher_cfg.sample_size
        del teacher.decoder

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg)

        if self.rank == 0:
            model_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model parameters: {model_params:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def get_ema_core(self):
        if self.world_size > 1:
            return self.ema.ema_model.module.core
        return self.ema.ema_model.core

    def save(self):
        save_dict = {
            'model' : self.model.state_dict(),
            'ema' : self.ema.state_dict(),
            'opt' : self.opt.state_dict(),
            'scaler' : self.scaler.state_dict(),
            'steps': self.total_step_counter
        }
        if self.scheduler is not None:
            save_dict['scheduler'] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        resume_ckpt = getattr(self.train_cfg, 'resume_ckpt', None)
        if resume_ckpt is None:
            return
        save_dict = super().load(resume_ckpt)

        self.model.load_state_dict(save_dict['model'], strict=False)
        self.ema.load_state_dict(save_dict['ema'])
        self.opt.load_state_dict(save_dict['opt'])
        if self.scheduler is not None and 'scheduler' in save_dict:
            self.scheduler.load_state_dict(save_dict['scheduler'])
        self.scaler.load_state_dict(save_dict['scaler'])
        self.total_step_counter = save_dict['steps']

    def train(self):
        torch.cuda.set_device(self.local_rank)

        # Prepare model, ema
        self.model = self.model.to(self.device).train()
        if self.world_size > 1:
            self.model = DDP(self.model)

        self.encoder = self.encoder.to(self.device).bfloat16().train()
        self.flux_vae = get_vae(self.train_cfg.vae_id).bfloat16().cuda().eval()
        self.flux_vae.decoder.to(self.device)
        self.flux_vae.decoder.eval()
        self.flux_vae.decoder.to(self.device)
        
        freeze(self.encoder)
        freeze(self.flux_vae)
        self.encoder = torch.compile(self.encoder, dynamic=False,fullgraph=True)
        self.flux_vae.encoder = torch.compile(self.flux_vae.encoder, dynamic=False,fullgraph=True)

        self.ema = EMA(
            self.model,
            beta = 0.999,
            update_after_step = 0,
            update_every = 1
        )

        # Set up optimizer and scheduler
        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(self.model, rank=self.rank,world_size=self.world_size,**self.train_cfg.opt_kwargs)
        else:
            opt_cls = getattr(torch.optim, self.train_cfg.opt)
            self.opt = opt_cls(self.model.parameters(), **self.train_cfg.opt_kwargs)

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

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch in loader:
                batch = batch.to(self.device).bfloat16()
                teacher_enc_input = batch.clone()
                batch = batch[:,:3]

                with torch.no_grad():
                    teacher_enc_input = F.interpolate(teacher_enc_input, size=tuple(self.teacher_size), mode='bilinear', align_corners=False)
                    teacher_mu, teacher_logvar = self.encoder(teacher_enc_input)
                    teacher_std = (teacher_logvar/2).exp()
                    teacher_z = torch.randn_like(teacher_mu) * teacher_std + teacher_mu
                    teacher_z = teacher_z / self.train_cfg.latent_scale
                    
                    batch = F.interpolate(batch, size=tuple(self.train_cfg.vae_size), mode='bilinear', align_corners=False)
                    latent_batch = self.flux_vae.encoder(batch) / self.train_cfg.ldm_scale # [b,16,45,80]
                
                with ctx:
                    diff_loss = self.model(latent_batch, teacher_z)
                    diff_loss = diff_loss / accum_steps

                metrics.log('diff_loss', diff_loss)

                self.scaler.scale(diff_loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    # Updates
                    if self.train_cfg.opt.lower() != "muon":
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
                            with ctx:
                                ema_rec = flow_sample(self.get_ema_core(), latent_batch, teacher_z, self.train_cfg.sampling_steps, self.flux_vae.decoder, scaling_factor = self.train_cfg.ldm_scale)

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
