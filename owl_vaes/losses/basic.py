import torch
from torch import Tensor

import einops

def latent_reg_loss(z: Tensor, logvar: Tensor = None, target_var: float = 0.1) -> Tensor:
    """
    Latent regularization loss - Compiled for performance. (Fake KL)
    Args:
        z: Tensor of shape [b,c,h,w] representing mu if logvar is provided, otherwise the samples
        logvar: Optional log variance tensor of same shape as z
        target_var: Target variance value, only used if logvar is None
    """
    mu = z
    if logvar is not None:
        log_target_var = logvar
    else:
        log_target_var = 2 * torch.log(
            torch.tensor(target_var, device=z.device, dtype=z.dtype)
        )  # log(target_var^2)

    kl = -0.5 * (1 + log_target_var - mu.pow(2) - log_target_var.exp())
    kl = einops.reduce(kl, "b ... -> b", reduction="sum").mean()

    return kl
