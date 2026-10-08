from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
from .fusion import DualSetFusion

class GatedAttentionMIL(nn.Module):

    def __init__(self, input_dim: int=128, attention_dim: int=128, dropout: float=0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.value_gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.sigmoid_gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, tile_embeddings: Tensor, tile_mask: Optional[Tensor]=None) -> dict[str, Tensor]:
        if tile_embeddings.ndim != 3 or tile_embeddings.shape[-1] != self.input_dim:
            raise ValueError(f'tile_embeddings must have shape [B, T, {self.input_dim}], got {tuple(tile_embeddings.shape)}')
        if tile_mask is None:
            tile_mask = torch.ones(tile_embeddings.shape[:2], dtype=torch.bool, device=tile_embeddings.device)
        if tile_mask.shape != tile_embeddings.shape[:2]:
            raise ValueError('tile_mask shape does not match tile_embeddings')
        tile_mask = tile_mask.to(device=tile_embeddings.device, dtype=torch.bool)
        if not torch.all(tile_mask.any(dim=1)):
            raise ValueError('each patient must contain at least one tile')
        gated = self.value_gate(tile_embeddings) * self.sigmoid_gate(tile_embeddings)
        attention_logits = self.attention(self.dropout(gated)).squeeze(-1)
        min_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(~tile_mask, min_value)
        weights = torch.softmax(attention_logits, dim=1)
        weights = weights.masked_fill(~tile_mask, 0.0)
        patient_embedding = torch.sum(weights.unsqueeze(-1) * tile_embeddings, dim=1)
        logits = self.classifier(self.dropout(patient_embedding)).squeeze(-1)
        return {'logits': logits, 'patient_embedding': patient_embedding, 'attention': weights}

class CRCFusionMIL(nn.Module):

    def __init__(self, fusion: DualSetFusion, attention_dim: int=128, dropout: float=0.1) -> None:
        super().__init__()
        self.fusion = fusion
        self.mil = GatedAttentionMIL(fusion.output_dim, attention_dim, dropout)

    def forward(self, tile_mask: Tensor, *, rgb_features: Optional[Tensor]=None, nucleus_tokens: Optional[Tensor]=None, nucleus_mask: Optional[Tensor]=None, cell_tokens: Optional[Tensor]=None, cell_mask: Optional[Tensor]=None) -> dict[str, Tensor]:
        if tile_mask.ndim != 2:
            raise ValueError('tile_mask must have shape [patients, tiles]')
        tile_mask = tile_mask.bool()
        if not torch.all(tile_mask.any(dim=1)):
            raise ValueError('each patient must contain at least one tile')
        patients, tiles = tile_mask.shape
        flat_selector = tile_mask.reshape(-1)
        fusion_inputs: dict[str, Tensor] = {}
        if rgb_features is not None:
            self._validate_prefix(rgb_features, patients, tiles, 'rgb_features')
            fusion_inputs['rgb_features'] = rgb_features.reshape(patients * tiles, *rgb_features.shape[2:])[flat_selector]
        if nucleus_tokens is not None:
            self._validate_prefix(nucleus_tokens, patients, tiles, 'nucleus_tokens')
            fusion_inputs['nucleus_tokens'] = nucleus_tokens.reshape(patients * tiles, *nucleus_tokens.shape[2:])[flat_selector]
        if nucleus_mask is not None:
            self._validate_prefix(nucleus_mask, patients, tiles, 'nucleus_mask')
            fusion_inputs['nucleus_mask'] = nucleus_mask.reshape(patients * tiles, *nucleus_mask.shape[2:])[flat_selector]
        if cell_tokens is not None:
            self._validate_prefix(cell_tokens, patients, tiles, 'cell_tokens')
            fusion_inputs['cell_tokens'] = cell_tokens.reshape(patients * tiles, *cell_tokens.shape[2:])[flat_selector]
        if cell_mask is not None:
            self._validate_prefix(cell_mask, patients, tiles, 'cell_mask')
            fusion_inputs['cell_mask'] = cell_mask.reshape(patients * tiles, *cell_mask.shape[2:])[flat_selector]
        valid_embeddings = self.fusion(**fusion_inputs)
        padded_embeddings = valid_embeddings.new_zeros((patients * tiles, self.fusion.output_dim))
        padded_embeddings[flat_selector] = valid_embeddings
        padded_embeddings = padded_embeddings.reshape(patients, tiles, self.fusion.output_dim)
        output = self.mil(padded_embeddings, tile_mask)
        output['tile_embeddings'] = padded_embeddings
        return output

    @staticmethod
    def _validate_prefix(tensor: Tensor, patients: int, tiles: int, name: str) -> None:
        if tensor.ndim < 3 or tensor.shape[:2] != (patients, tiles):
            raise ValueError(f'{name} must start with [patients, tiles]={(patients, tiles)}, got {tuple(tensor.shape)}')
