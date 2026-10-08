from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
TOKEN_DIM = 41
EMBED_DIM = 128
RGB_DIM = 512

def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim), nn.ReLU(inplace=True))

def _validate_set_inputs(tokens: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
        raise ValueError(f'tokens must have shape [tiles, objects, {TOKEN_DIM}], got {tuple(tokens.shape)}')
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f'object mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}')
    return (tokens, mask.to(device=tokens.device, dtype=torch.bool))

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=1)
    pooled = (values * weight).sum(dim=1) / count.clamp_min(1.0)
    return (pooled, count.squeeze(-1))

class MeanPoolEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, dropout: float=0.1) -> None:
        super().__init__()
        self.set_mlp = _mlp(TOKEN_DIM + 1, EMBED_DIM, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        tokens, mask = _validate_set_inputs(tokens, mask)
        pooled, counts = _masked_mean(tokens, mask)
        features = torch.cat((pooled, torch.log1p(counts).unsqueeze(-1)), dim=-1)
        output = self.set_mlp(features)
        return output * (counts > 0).unsqueeze(-1).to(output.dtype)

class DeepSetsEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, dropout: float=0.1) -> None:
        super().__init__()
        self.token_mlp = _mlp(TOKEN_DIM, EMBED_DIM, EMBED_DIM, dropout)
        self.set_mlp = _mlp(EMBED_DIM + 1, EMBED_DIM, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        tokens, mask = _validate_set_inputs(tokens, mask)
        encoded = self.token_mlp(tokens)
        pooled, counts = _masked_mean(encoded, mask)
        features = torch.cat((pooled, torch.log1p(counts).unsqueeze(-1)), dim=-1)
        output = self.set_mlp(features)
        return output * (counts > 0).unsqueeze(-1).to(output.dtype)

def build_set_encoder(name: str, dropout: float=0.1) -> nn.Module:
    normalized = name.lower().replace('-', '_')
    if normalized in {'mean', 'meanpool', 'mean_pool'}:
        return MeanPoolEncoder(dropout)
    if normalized in {'deepsets', 'deep_sets'}:
        return DeepSetsEncoder(dropout)
    if normalized in {'set_transformer', 'settransformer'}:
        raise ValueError('Set Transformer is intentionally disabled for BRACS breast_roi')
    raise ValueError(f'unknown set encoder: {name!r}')

class DualObjectTileEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, set_encoder: str='meanpool', *, use_rgb: bool, rgb_dim: int=RGB_DIM, dropout: float=0.1) -> None:
        super().__init__()
        self.use_rgb = bool(use_rgb)
        self.rgb_dim = int(rgb_dim)
        self.nucleus_encoder = build_set_encoder(set_encoder, dropout)
        self.cell_encoder = build_set_encoder(set_encoder, dropout)
        if self.nucleus_encoder is self.cell_encoder:
            raise AssertionError('nucleus/cell SetEncoders unexpectedly share parameters')
        self.rgb_projection = nn.Sequential(nn.Linear(self.rgb_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout)) if self.use_rgb else None
        fusion_width = (3 if self.use_rgb else 2) * EMBED_DIM
        self.fusion = nn.Sequential(nn.Linear(fusion_width, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, nucleus_tokens: Tensor, cell_tokens: Tensor, nucleus_mask: Optional[Tensor]=None, cell_mask: Optional[Tensor]=None, *, rgb_features: Optional[Tensor]=None, return_branches: bool=False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if nucleus_tokens.shape[0] != cell_tokens.shape[0]:
            raise ValueError('nucleus and cell batches have different tile counts')
        nucleus = self.nucleus_encoder(nucleus_tokens, nucleus_mask)
        cell = self.cell_encoder(cell_tokens, cell_mask)
        branches: dict[str, Tensor] = {'nucleus': nucleus, 'cell': cell}
        values: list[Tensor] = []
        if self.use_rgb:
            if rgb_features is None:
                raise ValueError('rgb_features are required for the RGB+mask expert')
            if rgb_features.shape != (nucleus.shape[0], self.rgb_dim):
                raise ValueError(f'rgb_features must have shape [{nucleus.shape[0]}, {self.rgb_dim}], got {tuple(rgb_features.shape)}')
            if rgb_features.device != nucleus.device:
                raise ValueError('RGB and object tokens must be on the same device')
            rgb = self.rgb_projection(rgb_features)
            branches['rgb'] = rgb
            values.append(rgb)
        elif rgb_features is not None:
            raise ValueError('geometry-only expert must not receive RGB features')
        values.extend((nucleus, cell))
        embedding = self.fusion(torch.cat(values, dim=-1))
        if return_branches:
            return (embedding, branches)
        return embedding

class RGBMaskTileEncoder(DualObjectTileEncoder):

    def __init__(self, set_encoder: str='meanpool', *, rgb_dim: int=RGB_DIM, dropout: float=0.1) -> None:
        super().__init__(set_encoder, use_rgb=True, rgb_dim=rgb_dim, dropout=dropout)

class GeometryTileEncoder(DualObjectTileEncoder):

    def __init__(self, set_encoder: str='meanpool', *, dropout: float=0.1) -> None:
        super().__init__(set_encoder, use_rgb=False, dropout=dropout)

class PackedGatedAttentionMIL(nn.Module):

    def __init__(self, input_dim: int=EMBED_DIM, attention_dim: int=EMBED_DIM, num_classes: int=7, dropout: float=0.1) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError('num_classes must be at least two')
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.value_gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.sigmoid_gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    @staticmethod
    def _validate_offsets(tile_count: int, offsets: Tensor, device: torch.device) -> Tensor:
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError('roi_offsets must be a one-dimensional CSR vector')
        offsets = offsets.to(device=device, dtype=torch.long)
        if int(offsets[0]) != 0 or int(offsets[-1]) != tile_count:
            raise ValueError('roi_offsets must start at zero and end at total_tiles')
        lengths = offsets[1:] - offsets[:-1]
        if torch.any(lengths <= 0):
            raise ValueError('every ROI must contain at least one tile')
        return offsets

    def forward(self, tile_embeddings: Tensor, roi_offsets: Tensor) -> dict[str, Tensor]:
        if tile_embeddings.ndim != 2 or tile_embeddings.shape[1] != self.input_dim:
            raise ValueError(f'tile_embeddings must have shape [total_tiles, {self.input_dim}], got {tuple(tile_embeddings.shape)}')
        offsets = self._validate_offsets(len(tile_embeddings), roi_offsets, tile_embeddings.device)
        lengths = offsets[1:] - offsets[:-1]
        roi_count = len(lengths)
        roi_index = torch.repeat_interleave(torch.arange(roi_count, device=tile_embeddings.device), lengths)
        gated = self.value_gate(tile_embeddings) * self.sigmoid_gate(tile_embeddings)
        scores = self.attention(self.dropout(gated)).squeeze(-1).float()
        maxima = scores.new_full((roi_count,), -torch.inf)
        maxima.scatter_reduce_(0, roi_index, scores, reduce='amax', include_self=True)
        unnormalized = torch.exp(scores - maxima[roi_index])
        denominator = scores.new_zeros((roi_count,))
        denominator.scatter_add_(0, roi_index, unnormalized)
        attention = unnormalized / denominator[roi_index]
        roi_embedding = scores.new_zeros((roi_count, self.input_dim))
        roi_embedding.scatter_add_(0, roi_index.unsqueeze(-1).expand(-1, self.input_dim), attention.unsqueeze(-1) * tile_embeddings.float())
        logits = self.classifier(self.dropout(roi_embedding))
        return {'logits': logits, 'roi_embedding': roi_embedding, 'attention': attention, 'roi_index': roi_index}

class BRACSROIModel(nn.Module):

    def __init__(self, tile_encoder: DualObjectTileEncoder, *, num_classes: int, attention_dim: int=EMBED_DIM, dropout: float=0.1) -> None:
        super().__init__()
        self.tile_encoder = tile_encoder
        self.mil = PackedGatedAttentionMIL(tile_encoder.output_dim, attention_dim, num_classes, dropout)

    def forward(self, roi_offsets: Tensor, **tile_inputs: Tensor) -> dict[str, Tensor]:
        tile_embedding = self.tile_encoder(**tile_inputs)
        if not isinstance(tile_embedding, Tensor):
            raise AssertionError('tile encoder unexpectedly returned branch diagnostics')
        output = self.mil(tile_embedding, roi_offsets)
        output['tile_embedding'] = tile_embedding
        return output
