from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
TOKEN_DIM = 41
EMBED_DIM = 128

def _mlp(input_dim: int, output_dim: int, dropout: float):
    return nn.Sequential(nn.Linear(input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(EMBED_DIM, output_dim), nn.ReLU(inplace=True))

def _masked_mean(values: Tensor, mask: Tensor):
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(1)
    return ((values * weight).sum(1) / count.clamp_min(1), count.squeeze(-1))

class FixedSetEncoder(nn.Module):

    def __init__(self, kind: str, dropout: float=0.1):
        super().__init__()
        normalized = kind.lower().replace('_', '')
        if normalized not in {'meanpool', 'deepsets'}:
            raise ValueError('only MeanPool/DeepSets are registered')
        self.kind = normalized
        self.token_mlp = None if normalized == 'meanpool' else _mlp(TOKEN_DIM, EMBED_DIM, dropout)
        self.set_mlp = _mlp((TOKEN_DIM if self.token_mlp is None else EMBED_DIM) + 1, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Tensor | None=None, count_override: Tensor | None=None):
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError('expected [B,N,41]')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        values = tokens if self.token_mlp is None else self.token_mlp(tokens)
        pooled, observed = _masked_mean(values, mask)
        count = observed if count_override is None else count_override.to(tokens)
        if self.token_mlp is not None and count_override is not None:
            raise ValueError('count_override is MeanPool-only')
        output = self.set_mlp(torch.cat((pooled, torch.log1p(count)[:, None]), -1))
        return output * (count > 0)[:, None].to(output.dtype)

class DualGeometryExpert(nn.Module):

    def __init__(self, kind: str, dropout: float=0.1):
        super().__init__()
        self.nucleus = FixedSetEncoder(kind, dropout)
        self.cell = FixedSetEncoder(kind, dropout)
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, nucleus_tokens, nucleus_mask, cell_tokens, cell_mask, nucleus_count=None, cell_count=None):
        n = self.nucleus(nucleus_tokens, nucleus_mask, nucleus_count)
        c = self.cell(cell_tokens, cell_mask, cell_count)
        return self.fusion(torch.cat((n, c), -1))

class PatchGeometryClassifier(nn.Module):

    def __init__(self, encoder: str, classes: int, dropout: float=0.1):
        super().__init__()
        self.expert = DualGeometryExpert(encoder, dropout)
        self.head = nn.Linear(128, classes)

    def forward(self, **batch):
        embedding = self.expert(**batch)
        return self.head(embedding)

class GatedAttentionMIL(nn.Module):

    def __init__(self, classes: int, dropout: float=0.1):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(128, 128), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(128, 128), nn.Sigmoid())
        self.attention = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, classes)

    def forward(self, embeddings: Tensor, mask: Tensor):
        scores = self.attention(self.value(embeddings) * self.gate(embeddings)).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = scores.softmax(-1)
        pooled = (weights[:, :, None] * embeddings).sum(1)
        return (self.head(self.dropout(pooled)), weights)

class GroupGeometryClassifier(nn.Module):

    def __init__(self, encoder: str, classes: int, dropout: float=0.1):
        super().__init__()
        self.expert = DualGeometryExpert(encoder, dropout)
        self.mil = GatedAttentionMIL(classes, dropout)

    def forward(self, tile_mask: Tensor, **batch):
        shape = tile_mask.shape
        flat = {key: value.reshape(shape[0] * shape[1], *value.shape[2:]) for key, value in batch.items()}
        embedding = self.expert(**flat).reshape(shape[0], shape[1], -1)
        return self.mil(embedding, tile_mask)
