from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
TOKEN_DIM = 15
EMBED_DIM = 128

def _validate_inputs(tokens: Tensor, mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f'tokens must have shape [B, N, D], got {tuple(tokens.shape)}')
    if tokens.shape[-1] != TOKEN_DIM:
        raise ValueError(f'expected {TOKEN_DIM} token features, got {tokens.shape[-1]}')
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f'mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}')
    return (tokens, mask.to(device=tokens.device, dtype=torch.bool))

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(values.dtype)
    counts = weights.sum(dim=1)
    pooled = (values * weights).sum(dim=1) / counts.clamp_min(1.0)
    return (pooled, counts.squeeze(-1))

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim), nn.ReLU(inplace=True))

class MeanPoolEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, dropout: float=0.1) -> None:
        super().__init__()
        self.set_mlp = _mlp(TOKEN_DIM + 1, EMBED_DIM, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        tokens, mask = _validate_inputs(tokens, mask)
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
        tokens, mask = _validate_inputs(tokens, mask)
        token_features = self.token_mlp(tokens)
        pooled, counts = _masked_mean(token_features, mask)
        features = torch.cat((pooled, torch.log1p(counts).unsqueeze(-1)), dim=-1)
        output = self.set_mlp(features)
        return output * (counts > 0).unsqueeze(-1).to(output.dtype)

class MultiheadAttentionBlock(nn.Module):

    def __init__(self, dim: int=EMBED_DIM, num_heads: int=4, dropout: float=0.1) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError('dim must be divisible by num_heads')
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, query: Tensor, key_value: Tensor, key_mask: Optional[Tensor]=None) -> Tensor:
        key_padding_mask = None if key_mask is None else ~key_mask
        attended, _ = self.attention(query, key_value, key_value, key_padding_mask=key_padding_mask, need_weights=False)
        hidden = self.norm1(query + attended)
        return self.norm2(hidden + self.feed_forward(hidden))

class InducedSetAttentionBlock(nn.Module):

    def __init__(self, dim: int=EMBED_DIM, num_heads: int=4, num_inducing: int=32, dropout: float=0.1) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_inducing, dim))
        nn.init.xavier_uniform_(self.inducing_points)
        self.to_inducing = MultiheadAttentionBlock(dim, num_heads, dropout)
        self.to_objects = MultiheadAttentionBlock(dim, num_heads, dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        inducing = self.inducing_points.expand(values.shape[0], -1, -1)
        summaries = self.to_inducing(inducing, values, mask)
        return self.to_objects(values, summaries)

class PoolingByMultiheadAttention(nn.Module):

    def __init__(self, dim: int=EMBED_DIM, num_heads: int=4, num_seeds: int=1, dropout: float=0.1) -> None:
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seed_vectors)
        self.mab = MultiheadAttentionBlock(dim, num_heads, dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        seeds = self.seed_vectors.expand(values.shape[0], -1, -1)
        return self.mab(seeds, values, mask)

class SetTransformerEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, dropout: float=0.1, num_heads: int=4, num_inducing: int=32) -> None:
        super().__init__()
        self.input_projection = nn.Linear(TOKEN_DIM, EMBED_DIM)
        self.isab1 = InducedSetAttentionBlock(EMBED_DIM, num_heads, num_inducing, dropout)
        self.isab2 = InducedSetAttentionBlock(EMBED_DIM, num_heads, num_inducing, dropout)
        self.pool = PoolingByMultiheadAttention(EMBED_DIM, num_heads, 1, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        tokens, mask = _validate_inputs(tokens, mask)
        output = tokens.new_zeros((tokens.shape[0], EMBED_DIM))
        nonempty = mask.any(dim=1)
        if not torch.any(nonempty):
            return output
        active_tokens = self.input_projection(tokens[nonempty])
        active_mask = mask[nonempty]
        active_tokens = self.isab1(active_tokens, active_mask)
        active_tokens = active_tokens * active_mask.unsqueeze(-1).to(active_tokens.dtype)
        active_tokens = self.isab2(active_tokens, active_mask)
        active_tokens = active_tokens * active_mask.unsqueeze(-1).to(active_tokens.dtype)
        pooled = self.pool(active_tokens, active_mask).squeeze(1)
        output[nonempty] = pooled
        return output

def build_set_encoder(name: str, dropout: float=0.1) -> nn.Module:
    normalized = name.lower().replace('-', '_')
    if normalized in {'mean', 'meanpool', 'mean_pool'}:
        return MeanPoolEncoder(dropout)
    if normalized in {'deepsets', 'deep_sets'}:
        return DeepSetsEncoder(dropout)
    if normalized in {'set_transformer', 'settransformer'}:
        return SetTransformerEncoder(dropout)
    raise ValueError(f'unknown set encoder: {name!r}')
