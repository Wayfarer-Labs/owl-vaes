import torch.nn.functional as F
from torch import Tensor, nn

from ..configs import TransformerConfig
from .attn import StackedTransformer

class CausalTransformerEncoder(nn.Module):
    """
    Originally used for CRT VQVAE (Navigating Compression-Generatio tradeoff paper).
    Also usable as small causal transformer
    """
    def __init__(self, dim_in, dim_out = None, config: TransformerConfig | None = None):
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
        if dim_out is None:
            dim_out = dim_in

        self.core = StackedTransformer(config)
        self.proj_in = nn.Linear(dim_in, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, dim_out, bias = False)

    def forward(self, x: Tensor):
        # x is [b,n,d]

        x = self.proj_in(x)
        pred = self.core(x)
        pred = self.proj_out(pred)
        return pred

if __name__ == "__main__":
    import torch

    crt = CRT(768).cuda().bfloat16()
    x = torch.randn(1,16,768).cuda().bfloat16()

    with torch.no_grad():
        print(crt(x))
