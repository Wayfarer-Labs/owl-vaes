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
from ..utils.logging import LogHelper, to_wandb_video_sidebyside
from .base import BaseTrainer
from ..configs import Config

from ..sampling.diffdec_samplers import SameNoiseSampler

from diffusers import AutoencoderTiny, AutoencoderDC
import gc

def get_vae(vae_id):
    if vae_id == "taef1":
        return AutoencoderTiny.from_pretrained("madebyollin/taef1")
    elif vae_id == "dcae":
        return AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers")
    else:
        raise ValueError(f"VAE {vae_id} not found")

class CausalDiffusionDecoderTrainer(BaseTrainer):
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
        
        freeze(self.encoder)
        self.encoder = torch.compile(self.encoder, dynamic=False,fullgraph=True)

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
        sample_loader = get_loader(
            "local_latent",
            1,
            root_dir="/mnt/data/datasets/cod_yt_latents",
            window_size=32
        )
        sample_loader = iter(sample_loader)
        sampler = SameNoiseSampler(self.train_cfg.sampling_steps)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for batch_rgb, batch_depth in loader:
                batch_rgb = batch_rgb.to(self.device) # b3hw
                batch_depth = batch_depth.to(self.device) # b1hw
                
                # Assumption is that it's already the right size for the teacher
                t_input = torch.cat([batch_rgb, batch_depth], dim=1).cuda().bfloat16()



                with torch.no_grad():
                    t_mu, t_logvar = self.encoder(t_input)
                    t_std = (t_logvar/2).exp()
                    t_z = torch.randn_like(t_mu) * t_std + t_mu
                    t_z = t_z / self.train_cfg.latent_scale

                with ctx:
                    diff_loss = self.model(batch_rgb, t_z)
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
                                sample_rgb, sample_depth = next(sample_loader)

                                sample_rgb = sample_rgb[0].cuda().bfloat16() # n3hw
                                sample_depth = sample_depth[0].cuda().bfloat16() # n1hw
                                sample_input = torch.cat([sample_rgb, sample_depth], dim=1) # n4hw
                                ema_rec = sampler(self.encoder, self.train_cfg.latent_scale, self.get_ema_core(), sample_input)
                                # outputs are [n,c,h,w] on all devices
                                if self.world_size > 1:
                                    # Gather ema_rec across all devices to get [b, n, c, h, w]
                                    ema_rec_list = [torch.zeros_like(ema_rec) for _ in range(self.world_size)]
                                    torch.distributed.all_gather(ema_rec_list, ema_rec)
                                    ema_rec = torch.stack(ema_rec_list, dim=0)
                                    
                                    # Also gather original RGB for side-by-side comparison
                                    sample_rgb_list = [torch.zeros_like(sample_rgb) for _ in range(self.world_size)]
                                    torch.distributed.all_gather(sample_rgb_list, sample_rgb.contiguous())
                                    sample_rgb_gathered = torch.stack(sample_rgb_list, dim=0)
                                else:
                                    # For single device, add batch dimension to match ema_rec
                                    sample_rgb_gathered = sample_rgb.unsqueeze(0)
                        if self.rank == 0:
                            wandb_dict['samples'] = to_wandb_video_sidebyside(
                                sample_rgb_gathered.detach().contiguous().bfloat16(),
                                ema_rec.detach().contiguous().bfloat16()
                            )
                            torch.save(sample_rgb_gathered.detach().contiguous().bfloat16(), "sample_rgb.pt")
                            wandb.log(wandb_dict)

                    self.total_step_counter += 1
                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()

                    gc.collect()
                    torch.cuda.empty_cache()
