from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.conditional_geometry.protocol import RESIDUAL_TOKEN_DIM, RGB_DIM
EMBED_DIM = 128

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(1)
    return ((values * weight).sum(1) / count.clamp_min(1.0), count.squeeze(-1))

def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(EMBED_DIM, output_dim), nn.ReLU(inplace=True))

class ResidualSetEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, kind: str, dropout: float) -> None:
        super().__init__()
        normalized = kind.lower().replace('_', '')
        if normalized not in {'meanpool', 'deepsets'}:
            raise ValueError('conditional_geometry permits only meanpool or deepsets')
        self.kind = normalized
        self.token_mlp = None if normalized == 'meanpool' else _mlp(RESIDUAL_TOKEN_DIM, EMBED_DIM, dropout)
        pooled_dim = RESIDUAL_TOKEN_DIM if self.token_mlp is None else EMBED_DIM
        self.set_mlp = _mlp(pooled_dim + 1, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != RESIDUAL_TOKEN_DIM:
            raise ValueError(f'expected [B,N,{RESIDUAL_TOKEN_DIM}] residual tokens')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.shape != tokens.shape[:2]:
            raise ValueError('token mask shape mismatch')
        mask = mask.bool()
        values = tokens if self.token_mlp is None else self.token_mlp(tokens)
        pooled, count = _masked_mean(values, mask)
        output = self.set_mlp(torch.cat((pooled, torch.log1p(count).unsqueeze(-1)), -1))
        return output * (count > 0).unsqueeze(-1).to(output.dtype)

class DualSetFusion(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, set_encoder: str='meanpool', object_mode: str='both', use_rgb: bool=True, rgb_dim: int=RGB_DIM, dropout: float=0.1) -> None:
        super().__init__()
        if object_mode != 'both' or not use_rgb:
            raise ValueError('conditional_geometry residual arms require RGB and both object branches')
        self.object_mode, self.use_rgb, self.rgb_dim = (object_mode, use_rgb, rgb_dim)
        self.rgb_projection = nn.Sequential(nn.Linear(rgb_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.nucleus_encoder = ResidualSetEncoder(set_encoder, dropout)
        self.cell_encoder = ResidualSetEncoder(set_encoder, dropout)
        self.fusion = nn.Sequential(nn.Linear(3 * EMBED_DIM, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, *, rgb_features: Tensor, nucleus_tokens: Tensor, nucleus_mask: Tensor, cell_tokens: Tensor, cell_mask: Tensor) -> Tensor:
        branches = (self.rgb_projection(rgb_features), self.nucleus_encoder(nucleus_tokens, nucleus_mask), self.cell_encoder(cell_tokens, cell_mask))
        return self.fusion(torch.cat(branches, -1))

class SICAPClassifier(nn.Module):

    def __init__(self, fusion: DualSetFusion) -> None:
        super().__init__()
        self.fusion = fusion
        self.grade_head, self.cribriform_head = (nn.Linear(128, 4), nn.Linear(128, 1))

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        embedding = self.fusion(**inputs)
        return {'embedding': embedding, 'grade_logits': self.grade_head(embedding), 'cribriform_logits': self.cribriform_head(embedding).squeeze(-1)}

class GatedAttentionMIL(nn.Module):

    def __init__(self, input_dim: int=128, attention_dim: int=128, dropout: float=0.1) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention, self.dropout, self.classifier = (nn.Linear(attention_dim, 1), nn.Dropout(dropout), nn.Linear(input_dim, 1))

    def forward(self, values: Tensor, mask: Tensor) -> dict[str, Tensor]:
        logits = self.attention(self.dropout(self.value(values) * self.gate(values))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, 1).masked_fill(~mask, 0.0)
        embedding = (weights.unsqueeze(-1) * values).sum(1)
        return {'logits': self.classifier(self.dropout(embedding)).squeeze(-1), 'patient_embedding': embedding, 'attention': weights}

class CRCFusionMIL(nn.Module):

    def __init__(self, fusion: DualSetFusion, attention_dim: int=128, dropout: float=0.1) -> None:
        super().__init__()
        self.fusion = fusion
        self.mil = GatedAttentionMIL(128, attention_dim, dropout)

    def forward(self, tile_mask: Tensor, **inputs: Tensor) -> dict[str, Tensor]:
        patients, tiles = tile_mask.shape
        selector = tile_mask.reshape(-1)
        flat = {name: value.reshape(patients * tiles, *value.shape[2:])[selector] for name, value in inputs.items()}
        valid = self.fusion(**flat)
        padded = valid.new_zeros((patients * tiles, 128))
        padded[selector] = valid
        return self.mil(padded.reshape(patients, tiles, 128), tile_mask.bool())
