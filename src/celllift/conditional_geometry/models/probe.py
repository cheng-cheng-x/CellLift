from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.conditional_geometry.protocol import ANCHOR_2D_DIM, CONTEXT_DIM, RGB_DIM, TARGET_DIM

def _branch(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim), nn.GELU())

class Conditional3DProbe(nn.Module):

    def __init__(self, hidden_dim: int=256, dropout: float=0.1) -> None:
        super().__init__()
        self.rgb = _branch(RGB_DIM, hidden_dim, 128, dropout)
        self.anchor = _branch(ANCHOR_2D_DIM, 128, 128, dropout)
        self.context = _branch(CONTEXT_DIM, 128, 128, dropout)
        self.head = nn.Sequential(nn.Linear(384, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, TARGET_DIM))

    def forward(self, rgb: Tensor, anchor_2d: Tensor, context: Tensor) -> Tensor:
        if rgb.shape[-1] != RGB_DIM or anchor_2d.shape[-1] != ANCHOR_2D_DIM or context.shape[-1] != CONTEXT_DIM:
            raise ValueError('conditional probe input width mismatch')
        if not rgb.shape[0] == anchor_2d.shape[0] == context.shape[0]:
            raise ValueError('conditional probe batch sizes differ')
        return self.head(torch.cat((self.rgb(rgb), self.anchor(anchor_2d), self.context(context)), dim=-1))
