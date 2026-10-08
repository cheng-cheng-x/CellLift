from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.matched_geometry_controls.models import GatedAttentionMIL
from .config import ATTN_DIM, DROPOUT, TILE_EMBED

class PatientMIL(nn.Module):

    def __init__(self, embed_dim: int, classes: int, attn_dim: int=ATTN_DIM, dropout: float=DROPOUT):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.value = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Sigmoid())
        self.attention = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, classes)

    def forward(self, embeddings: Tensor, mask: Tensor | None=None) -> tuple[Tensor, Tensor]:
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        elif mask.ndim == 1:
            mask = mask.unsqueeze(0)
        scores = self.attention(self.value(embeddings) * self.gate(embeddings)).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * embeddings).sum(1)
        logits = self.head(self.dropout(pooled))
        if squeeze:
            return (logits.squeeze(0), weights.squeeze(0))
        return (logits, weights)

class ImageProjection(nn.Module):

    def __init__(self, in_dim: int=384, out_dim: int=TILE_EMBED, dropout: float=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)
