from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
from .encoders import EMBED_DIM, build_set_encoder
VALID_OBJECT_MODES = frozenset({'rgb', 'nucleus', 'cell', 'both'})

class DualSetFusion(nn.Module):
    output_dim = EMBED_DIM
    fusion_input_dim = 3 * EMBED_DIM

    def __init__(self, set_encoder: str='meanpool', object_mode: str='both', use_rgb: bool=True, rgb_dim: int=512, dropout: float=0.1) -> None:
        super().__init__()
        if object_mode not in VALID_OBJECT_MODES:
            raise ValueError(f'object_mode must be one of {sorted(VALID_OBJECT_MODES)}')
        if object_mode == 'rgb' and (not use_rgb):
            raise ValueError("object_mode='rgb' requires use_rgb=True")
        self.object_mode = object_mode
        self.use_rgb = use_rgb
        self.rgb_dim = rgb_dim
        self.rgb_projection = nn.Sequential(nn.Linear(rgb_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.nucleus_encoder = build_set_encoder(set_encoder, dropout)
        self.cell_encoder = build_set_encoder(set_encoder, dropout)
        self.fusion = nn.Sequential(nn.Linear(self.fusion_input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    @property
    def uses_nucleus(self) -> bool:
        return self.object_mode in {'nucleus', 'both'}

    @property
    def uses_cell(self) -> bool:
        return self.object_mode in {'cell', 'both'}

    def forward(self, rgb_features: Optional[Tensor]=None, nucleus_tokens: Optional[Tensor]=None, nucleus_mask: Optional[Tensor]=None, cell_tokens: Optional[Tensor]=None, cell_mask: Optional[Tensor]=None, *, return_branches: bool=False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        batch_size, device, dtype = self._infer_batch_device_dtype(rgb_features, nucleus_tokens, cell_tokens)
        zero = torch.zeros((batch_size, EMBED_DIM), device=device, dtype=dtype)
        if self.use_rgb:
            if rgb_features is None:
                raise ValueError('rgb_features are required when use_rgb=True')
            if rgb_features.shape != (batch_size, self.rgb_dim):
                raise ValueError(f'rgb_features must have shape [B, {self.rgb_dim}], got {tuple(rgb_features.shape)}')
            rgb_embedding = self.rgb_projection(rgb_features)
        else:
            rgb_embedding = zero
        if self.uses_nucleus:
            if nucleus_tokens is None:
                raise ValueError('nucleus_tokens are required by object_mode')
            nucleus_embedding = self.nucleus_encoder(nucleus_tokens, nucleus_mask)
        else:
            nucleus_embedding = zero
        if self.uses_cell:
            if cell_tokens is None:
                raise ValueError('cell_tokens are required by object_mode')
            cell_embedding = self.cell_encoder(cell_tokens, cell_mask)
        else:
            cell_embedding = zero
        branches = {'rgb': rgb_embedding, 'nucleus': nucleus_embedding, 'cell': cell_embedding}
        fused = self.fusion(torch.cat(tuple(branches.values()), dim=-1))
        if return_branches:
            return (fused, branches)
        return fused

    @staticmethod
    def _infer_batch_device_dtype(rgb: Optional[Tensor], nucleus: Optional[Tensor], cell: Optional[Tensor]) -> tuple[int, torch.device, torch.dtype]:
        tensors = [value for value in (rgb, nucleus, cell) if value is not None]
        if not tensors:
            raise ValueError('at least one input tensor is required')
        batch_size = tensors[0].shape[0]
        if any((value.shape[0] != batch_size for value in tensors)):
            raise ValueError('all inputs must have the same batch size')
        if any((value.device != tensors[0].device for value in tensors)):
            raise ValueError('all inputs must be on the same device')
        return (batch_size, tensors[0].device, tensors[0].dtype)
