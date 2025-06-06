import torch
from torch import Tensor

import einops

@torch.compile(mode="max-autotune")
def latent_reg_loss(z: Tensor, target_var: float = 0.1) -> Tensor:
    """
    Latent regularization loss - Compiled for performance. (Fake KL)
    """
    # z is [b,c,h,w]
    # KL divergence between N(z, 0.1) and N(0,1)
    mu = z
    log_target_var = 2 * torch.log(
        torch.tensor(target_var, device=z.device, dtype=z.dtype)
    )  # log(0.1^2)

    kl = -0.5 * (1 + log_target_var - mu.pow(2) - log_target_var.exp())
    kl = einops.reduce(kl, "b ... -> b", reduction="sum").mean()

    return kl