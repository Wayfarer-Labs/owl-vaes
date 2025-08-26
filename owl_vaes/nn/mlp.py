from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config : 'TransformerConfig'):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(getattr(config, "dropout", 0))

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLPSimple(nn.Module):
    def __init__(self, dim_in, dim_middle=None, dim_out=None):
        super().__init__()
        
        dim_out = dim_out if dim_out is not None else dim_in
        dim_middle = dim_middle if dim_middle is not None else dim_out * 4

        self.fc_uv = nn.Linear(dim_in, dim_middle)
        self.fc_vw = nn.Linear(dim_middle, dim_out)

    def forward(self, x):
        x = self.fc_uv(x)
        x = F.silu(x)
        x = self.fc_vw(x)
        return x
