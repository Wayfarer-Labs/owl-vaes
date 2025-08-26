import torch.nn.functional as F
from torch import Tensor, nn

from ..configs import TransformerConfig
from .attn import StackedTransformer

class CRT(nn.Module):
    def __init__(self, dim, config: TransformerConfig | None = None):
        super().__init__()

        if config is None:
            config = TransformerConfig(
                n_layers=2,
                n_heads=6,
                d_model=384,
                causal=True,
                block_size=1,
                rope_impl='simple'
            )

        self.core = StackedTransformer(config)
        self.proj_in = nn.Linear(dim, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, dim, bias = False)


    def forward(self, x: Tensor):
        # x is [b,n,d]
        x_in = x[:,:-1]
        x_out = x[:,1:]

        x_in = self.proj_in(x_in)
        pred = self.core(x_in)
        pred = self.proj_out(pred)
        return F.mse_loss(pred, x_out)

if __name__ == "__main__":
    import torch

    crt = CRT(768).cuda().bfloat16()
    x = torch.randn(1,16,768).cuda().bfloat16()

    with torch.no_grad():
        print(crt(x))
