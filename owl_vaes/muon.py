import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Modified version of muon optimizer that still works when world size is 1
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device='cuda')
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]

            if self.world_size == 1:
                # Special case for single GPU
                for p in params:
                    g = p.grad
                    if g is None:
                        continue
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4:
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    if g.ndim == 2:
                        g = g.view_as(p)
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                continue

            # Multi-GPU case
            handle = None
            params_world = None
            def update_prev():
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    if g is None:
                        continue
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4:
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0 and handle is not None:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

class CombinedOptimizer(Optimizer):
    def __init__(self, model, rank=0, world_size=1, **kwargs):
        # We need this for the parent Optimizer class
        self.defaults = {}

        adamw_keys = kwargs.pop('adamw_keys', [])
        if world_size > 1:
            adamw_keys = ['module.' + key for key in adamw_keys]

        adamw_parameters = [p for n, p in model.named_parameters() if any(key in n for key in adamw_keys) or p.ndim < 2]
        muon_parameters = [p for n, p in model.named_parameters() if not any(key in n for key in adamw_keys) and p.ndim >= 2]

        # Initialize sub-optimizers
        self.adamw = AdamW(
            adamw_parameters,
            lr=kwargs.get('adamw_lr'),
            betas=kwargs.get('adamw_betas', (0.9, 0.999)),
            weight_decay=kwargs.get('adamw_wd', 0.01),
            eps=kwargs.get('adamw_eps', 1.0e-15)
        )

        self.muon = Muon(
            muon_parameters,
            lr=kwargs.get('lr'),
            momentum=kwargs.get('momentum'),
            rank = rank,
            world_size = world_size
        )

        # For LR scheduler compatibility
        self.param_groups = self.adamw.param_groups + self.muon.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.adamw.zero_grad(set_to_none)
        self.muon.zero_grad(set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.adamw.step()
        self.muon.step()

        return loss

    def state_dict(self):
        return {
            'adamw': self.adamw.state_dict(),
            'muon': self.muon.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.adamw.load_state_dict(state_dict['adamw'])
        self.muon.load_state_dict(state_dict['muon'])

def init_muon(model, rank = 0, world_size = 1, **kwargs):
    return CombinedOptimizer(model, rank, world_size, **kwargs)
