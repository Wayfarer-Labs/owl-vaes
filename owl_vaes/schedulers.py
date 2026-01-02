import torch.optim.lr_scheduler as lr_scheduler
import math

class LinearWarmupScheduler(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps=1000, min_lr=None, start_mult=None, last_epoch=-1):
        """
        Linear warmup scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            min_lr: Absolute minimum LR to start from (mutually exclusive with start_mult)
            start_mult: Multiplier for base_lr to start from (e.g., 0.01 means start at 1% of base_lr)
            last_epoch: Last epoch number for resuming
        """
        self.warmup_steps = warmup_steps

        # Validate arguments
        if min_lr is not None and start_mult is not None:
            raise ValueError("Cannot specify both min_lr and start_mult")
        if min_lr is None and start_mult is None:
            min_lr = 1e-6  # Default to old behavior

        self.min_lr = min_lr
        self.start_mult = start_mult
        self.base_lrs = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_steps:
            return self.base_lrs

        # Linear interpolation
        scale = self.last_epoch / self.warmup_steps

        if self.start_mult is not None:
            # Multiplier mode: interpolate from start_mult * base_lr to base_lr
            return [base_lr * (self.start_mult + (1.0 - self.start_mult) * scale)
                    for base_lr in self.base_lrs]
        else:
            # Absolute mode: interpolate from min_lr to base_lr
            return [self.min_lr + (base_lr - self.min_lr) * scale
                    for base_lr in self.base_lrs]

class LinearWarmupWithCosineDecay(lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps=1000, decay_after=10000, decay_steps=10000,
                 decay_to=None, start_mult=None, decay_to_mult=None, last_epoch=-1):
        """
        Linear warmup followed by cosine decay.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            decay_after: Absolute step when cosine decay starts (NOT relative to warmup)
            decay_steps: Duration of the cosine decay phase (how many steps to decay over)
            decay_to: Absolute minimum LR to decay to (mutually exclusive with decay_to_mult)
            start_mult: Multiplier for base_lr to start warmup from (e.g., 0.01)
            decay_to_mult: Multiplier for base_lr to decay to (e.g., 0.1)
            last_epoch: Last epoch number for resuming
        """
        self.warmup_steps = warmup_steps
        self.decay_after = decay_after
        self.decay_steps = decay_steps

        # Validate arguments
        if decay_to is not None and decay_to_mult is not None:
            raise ValueError("Cannot specify both decay_to and decay_to_mult")
        if decay_to is None and decay_to_mult is None:
            decay_to_mult = 0.0  # Default to decaying to 0

        if start_mult is None:
            start_mult = 0.01  # Default: start at 1% of base_lr

        self.decay_to = decay_to
        self.start_mult = start_mult
        self.decay_to_mult = decay_to_mult
        self.base_lrs = None
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            scale = self.last_epoch / self.warmup_steps
            return [base_lr * (self.start_mult + (1.0 - self.start_mult) * scale)
                    for base_lr in self.base_lrs]

        elif self.last_epoch < self.decay_after:
            # Constant phase (stay at base_lr)
            return self.base_lrs

        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.decay_after) / max(1, self.decay_steps)
            # Cosine annealing: 1.0 at start, 0.0 at end (clamped after decay_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            if self.decay_to_mult is not None:
                # Multiplier mode
                return [base_lr * (self.decay_to_mult + (1.0 - self.decay_to_mult) * cosine_factor)
                        for base_lr in self.base_lrs]
            else:
                # Absolute mode
                return [self.decay_to + (base_lr - self.decay_to) * cosine_factor
                        for base_lr in self.base_lrs]

def get_scheduler_cls(scheduler_id):
    if scheduler_id == "LinearWarmup":
        return LinearWarmupScheduler
    elif scheduler_id == "LinearWarmupWithCosineDecay":
        return LinearWarmupWithCosineDecay
    raise ValueError(f"Unknown scheduler {scheduler_id}")
