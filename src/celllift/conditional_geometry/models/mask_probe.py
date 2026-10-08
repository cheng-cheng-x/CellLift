from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.conditional_geometry.mask_protocol import MASK_CONTEXT_DIM, RAY_DIM, TARGET_DIM

def _branch(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim), nn.GELU())

class MaskOnly3DProbe(nn.Module):

    def __init__(self, hidden_dim: int=256, dropout: float=0.1) -> None:
        super().__init__()
        self.local = _branch(RAY_DIM, 128, 128, dropout)
        self.context = _branch(MASK_CONTEXT_DIM, 128, 128, dropout)
        self.head = nn.Sequential(nn.Linear(256, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, TARGET_DIM))

    def forward(self, rays: Tensor, context: Tensor) -> Tensor:
        if rays.shape[-1] != RAY_DIM or context.shape[-1] != MASK_CONTEXT_DIM:
            raise ValueError('mask-only probe input width mismatch')
        if rays.shape[0] != context.shape[0]:
            raise ValueError('mask-only probe batch sizes differ')
        return self.head(torch.cat((self.local(rays), self.context(context)), dim=-1))
