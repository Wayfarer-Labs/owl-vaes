import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self, var_coeff, cov_coeff, *, inv_coeff = 0.0, mlp_spec: str = None):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.inv_coeff = inv_coeff

        # only create projector if we need it
        self.projector = None
        needs_vicreg_projector = var_coeff > 0.0 or cov_coeff > 0.0
        if needs_vicreg_projector:
            assert mlp_spec is not None, "mlp_spec must be provided if using VICReg"
            self.projector = VICRegProjector(mlp_spec)
    
        self.use_augmentation = inv_coeff > 0.0
        self.augmentation = None if not self.use_augmentation else Augmentation()


    def forward(self, z):
        var_loss = self.var_coeff * self.var_loss(z) if self.projector is not None else 0.
        cov_loss = self.cov_coeff * self.cov_loss(z) if self.projector is not None else 0.
        inv_loss = self.inv_coeff * self.inv_loss(z) if self.use_augmentation else 0.
        return var_loss + cov_loss + inv_loss

    def var_loss(self, z):
        z_proj = self.projector(z)
        std_z = torch.sqrt(z_proj.var(dim=0) + 1e-04)
        return torch.mean(torch.relu(1 - std_z)) # penalty if std < 1

    def cov_loss(self, z):
        z_proj = self.projector(z)
        # center
        z_proj = z_proj - z_proj.mean(dim=0)
        cov_z: Tensor = (z_proj.T @ z_proj) / (z_proj.shape[0] - 1)
        return cov_z.fill_diagonal_(0).pow_(2).sum() / z_proj.shape[1] # off diagonal penalty

    def inv_loss(self, z):
        # TODO: Might not wanna do this cause lpips + inv loss might be redundant.
        # we will see
        z1, z2 = z, self.augmentation(z)
        return F.mse_loss(z1, z2)
    

class Augmentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError('Augmentation must be implemented.')


class VICRegProjector(nn.Module):
    def __init__(self, mlp_spec: str):
        """
        Create VICReg projector from mlp_spec string.
        
        Args:
            mlp_spec: String like "512-2048-2048-512" defining layer sizes
        """
        super().__init__()
        
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        
        # All layers except the last get BatchNorm + ReLU
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        
        # Final layer has no bias (as per VICReg paper)
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.projector(x)