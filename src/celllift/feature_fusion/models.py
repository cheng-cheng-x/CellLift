from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.cross_fitted_correction.models import DualSetSummary
EMBED_DIM = 128

class FeatureFusionModel(nn.Module):

    def __init__(self, dataset: str, *, geometry_enabled: bool, dropout: float=0.1) -> None:
        super().__init__()
        self.dataset = dataset
        self.geometry_enabled = bool(geometry_enabled)
        self.rgb_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.geometry = DualSetSummary('multistat', dropout)
        self.geometry_projection = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.fused_norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, 4 if dataset == 'sicapv2' else 1)

    def forward(self, *, rgb_features: Tensor, nucleus_summary: Tensor, cell_summary: Tensor) -> dict[str, Tensor]:
        if rgb_features.ndim != 2 or rgb_features.shape[-1] != 512:
            raise ValueError('feature_fusion requires [B,512] fold-specific paper-RGB features')
        rgb = self.rgb_projection(rgb_features)
        if self.geometry_enabled:
            geometry = self.geometry_projection(self.geometry(nucleus_summary=nucleus_summary, cell_summary=cell_summary))
        else:
            geometry = torch.zeros_like(rgb)
        fused = self.fused_norm(rgb + geometry)
        logits = self.head(fused)
        if self.dataset == 'tcga_crc_msi':
            logits = logits.squeeze(-1)
        return {'logits': logits, 'rgb_embedding': rgb, 'geometry_embedding': geometry, 'embedding': fused}

def paired_parameter_count(dataset: str) -> int:
    real = FeatureFusionModel(dataset, geometry_enabled=True)
    shuffled = FeatureFusionModel(dataset, geometry_enabled=True)
    left = sum((parameter.numel() for parameter in real.parameters()))
    right = sum((parameter.numel() for parameter in shuffled.parameters()))
    if left != right:
        raise AssertionError('real/shuffled capacity mismatch')
    return left
