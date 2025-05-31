import torch
from torch import nn

from ..configs import TransformerConfig

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32, s=1.0):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = s*y.flatten()[:, None] * omega[None, :]
    x = s*x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

def get_ortho_like(dim, heads, alpha, beta, sign=1, dist='uniform'):
    if dist == 'normal':
        A = alpha * torch.randn(dim, dim) / (dim**0.5) + sign * beta * torch.eye(dim)
    if dist == 'uniform':
        A = alpha * (torch.rand(dim, dim) * 2 * 3**0.5 / (dim**0.5) - 3**0.5 / (dim**0.5)) \
            + sign * beta * torch.eye(dim)

    U, S, V = torch.linalg.svd(A)
    L = U @ torch.diag(torch.sqrt(S))
    R = torch.diag(torch.sqrt(S)) @ V
    return L, R

def mimetic_init(qkv : nn.Linear, out : nn.Linear, config : TransformerConfig):
    dim = config.d_model
    head_dim = config.d_model // config.n_heads

    alpha = 0.7
    beta = 0.7

    for h in range(config.n_heads):
        Q,K = get_ortho_like(dim, -float('inf'), alpha, beta, 1)
        Q = Q[:,:head_dim]
        K = K.T[:,:head_dim]

        qkv.weight.data[(h*head_dim):((h+1)*head_dim)] = Q.T.float()
        qkv.weight.data[dim+(h*head_dim):dim+((h+1)*head_dim)] = K.T.float()

    V, Proj = get_ortho_like(dim, config.n_heads, 0.4, 0.4, -1)
    qkv.weight.data[2*dim:] = V.float()
    out.weight.data = Proj
