from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.conditional_geometry.models.mask_downstream import EMBED_DIM, MaskResidualSetEncoder

class RGBMaskDualSetFusion(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, set_encoder: str='meanpool', object_mode: str='both', use_rgb: bool=True, rgb_dim: int=512, dropout: float=0.1) -> None:
        super().__init__()
        if object_mode != 'both' or not use_rgb:
            raise ValueError('RGB+mask rgb_mask_conditioning requires both branches and RGB')
        self.object_mode, self.use_rgb, self.rgb_dim = (object_mode, use_rgb, rgb_dim)
        self.rgb_projection = nn.Sequential(nn.Linear(rgb_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.nucleus_encoder = MaskResidualSetEncoder(set_encoder, dropout)
        self.cell_encoder = MaskResidualSetEncoder(set_encoder, dropout)
        self.fusion = nn.Sequential(nn.Linear(3 * EMBED_DIM, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, *, rgb_features: Tensor, nucleus_tokens: Tensor, nucleus_mask: Tensor, cell_tokens: Tensor, cell_mask: Tensor) -> Tensor:
        if rgb_features.ndim != 2 or rgb_features.shape[-1] != self.rgb_dim:
            raise ValueError('RGB+mask fusion RGB width mismatch')
        return self.fusion(torch.cat((self.rgb_projection(rgb_features), self.nucleus_encoder(nucleus_tokens, nucleus_mask), self.cell_encoder(cell_tokens, cell_mask)), -1))
