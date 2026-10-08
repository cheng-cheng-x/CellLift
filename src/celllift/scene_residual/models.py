from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from .data import EDGE_DIM, NODE_DIM
NODE_SCALE = torch.tensor([0.2, 0.2, 0.2, 0.5, 0.2, 0.2, 0.2, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
PROB_CLIP = 1e-07
NETWORK_ARMS = ('M', 'XY', 'G', 'G-Het', 'Recal')
ARMS = ('B', 'M', 'XY', 'G', 'G-Zperm', 'Recal')

def segment_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size,) + tuple(values.shape[1:]))
    if values.shape[0]:
        output.index_add_(0, index, values)
    return output

def segment_count(index: Tensor, size: int, dtype=torch.float32) -> Tensor:
    return segment_sum(torch.ones(index.shape[0], dtype=dtype, device=index.device), index, size)

def segment_mean(values: Tensor, index: Tensor, size: int) -> Tensor:
    total = segment_sum(values, index, size)
    count = segment_count(index, size, values.dtype).clamp_min(1.0)
    return total / count.view((-1,) + (1,) * (values.dim() - 1))

def segment_std(values: Tensor, index: Tensor, size: int) -> Tensor:
    mean = segment_mean(values, index, size)
    centered = values - mean[index]
    std = torch.sqrt(segment_mean(centered.square(), index, size).clamp_min(0.0))
    return torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)

def segment_softmax(scores: Tensor, index: Tensor, size: int) -> Tensor:
    if scores.numel() == 0:
        return scores
    maximum = scores.new_full((size,), -torch.inf)
    maximum = maximum.scatter_reduce(0, index, scores, reduce='amax', include_self=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(scores - maximum[index])
    return exponent / segment_sum(exponent, index, size).clamp_min(1e-12)[index]

class MessageLayer(nn.Module):

    def __init__(self, width: int, edge_dim: int=EDGE_DIM, dropout: float=0.1):
        super().__init__()
        payload_dim = 2 * width + edge_dim
        self.message = nn.Sequential(nn.Linear(payload_dim, width), nn.ReLU(inplace=True), nn.Linear(width, width))
        self.gate = nn.Sequential(nn.Linear(payload_dim, width), nn.Sigmoid())
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, edge_index: Tensor, edge: Tensor, chunk: int) -> Tensor:
        if edge_index.shape[1] == 0:
            return hidden
        aggregate = torch.zeros_like(hidden)
        degree = torch.zeros(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
        for start in range(0, edge_index.shape[1], chunk):
            source = edge_index[0, start:start + chunk]
            target = edge_index[1, start:start + chunk]
            payload = torch.cat((hidden[source], hidden[target], edge[start:start + chunk]), -1)
            aggregate.index_add_(0, target, self.message(payload) * self.gate(payload))
            degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype))
        update = self.update(torch.cat((hidden, aggregate / degree.clamp_min(1.0).unsqueeze(-1)), -1))
        return hidden + self.dropout(update)

class SceneCorrectionNet(nn.Module):

    def __init__(self, *, classes: int, width: int=64, layers: int=2, dropout: float=0.1, arm: str='G', bag: bool=False, tile_rgb: bool=False, edge_chunk: int=200000, baseline_dim: int | None=None):
        super().__init__()
        if arm == 'G-Zperm':
            arm = 'G'
        if arm == 'B':
            raise ValueError('arm B is the frozen baseline and has no trainable network')
        if arm not in NETWORK_ARMS:
            raise ValueError(f'unsupported arm: {arm}')
        self.classes = int(classes)
        self.arm = arm
        self.heterogeneity = arm == 'G-Het'
        self.bag = bool(bag)
        self.edge_chunk = int(edge_chunk)
        self.baseline_dim = self.classes if baseline_dim is None else int(baseline_dim)
        self.recal = arm == 'Recal'
        geometry_arm = 'G' if arm == 'G-Het' else arm
        if self.recal:
            self.node_encoder = None
            self.messages = nn.ModuleList()
            self.tile_rgb_projection = None
            self.tile_value = None
            self.tile_gate = None
            self.tile_score = None
            head_input = self.baseline_dim + 1
        else:
            self.register_buffer('node_scale', NODE_SCALE.clone())
            self.node_encoder = nn.Sequential(nn.Linear(NODE_DIM, width), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(width, width), nn.ReLU(inplace=True))
            self.messages = nn.ModuleList() if geometry_arm == 'M' else nn.ModuleList((MessageLayer(width, EDGE_DIM, dropout) for _ in range(int(layers))))
            self.tile_rgb_projection = nn.Linear(2, width) if tile_rgb else None
            self.tile_value = nn.Sequential(nn.Linear(width, width), nn.Tanh())
            self.tile_gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid())
            self.tile_score = nn.Linear(width, 1, bias=False)
            head_input = width + self.baseline_dim + 1
            if self.heterogeneity:
                head_input += width
        self.head = nn.Sequential(nn.Linear(head_input, width), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(width, self.classes))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if self.heterogeneity:
            nn.init.zeros_(self.head[0].weight[:, width:2 * width])

    def _baseline_anchor(self, batch: Any) -> tuple[Tensor, Tensor]:
        if self.classes == 1:
            probability = batch.baseline[:, 0].clamp(PROB_CLIP, 1.0 - PROB_CLIP)
            anchor = torch.log(probability) - torch.log1p(-probability)
            entropy = -(probability * torch.log(probability) + (1.0 - probability) * torch.log1p(-probability))
            return (anchor, entropy)
        probability = batch.baseline.clamp(min=PROB_CLIP, max=1.0)
        anchor = torch.log(probability)
        entropy = -(probability * torch.log(probability.clamp_min(PROB_CLIP))).sum(-1)
        return (anchor, entropy)

    def encode_nodes(self, batch: Any) -> Tensor:
        if self.recal or self.node_encoder is None:
            raise ValueError('Recal has no node encodings')
        total = int(batch.tile_total)
        if batch.node_tile.numel() and int(batch.node_tile.max()) >= total:
            raise ValueError(f'node_tile index outside the {total} batched tiles')
        encoded = self.node_encoder(batch.node / self.node_scale.to(batch.node.device))
        for layer in self.messages:
            encoded = layer(encoded, batch.edge_index, batch.edge, self.edge_chunk)
        return encoded

    def tile_representation(self, batch: Any, encoded: Tensor) -> Tensor:
        total = int(batch.tile_total)
        mean = segment_mean(encoded, batch.node_tile, total)
        count = segment_count(batch.node_tile, total, encoded.dtype)
        if self.tile_rgb_projection is not None:
            mean = mean + self.tile_rgb_projection(batch.tile_rgb)
        return mean * (count > 0).unsqueeze(-1).to(mean.dtype)

    def tile_heterogeneity(self, batch: Any, encoded: Tensor) -> Tensor:
        total = int(batch.tile_total)
        spread = segment_std(encoded, batch.node_tile, total)
        count = segment_count(batch.node_tile, total, encoded.dtype)
        return spread * (count > 0).unsqueeze(-1).to(spread.dtype)

    def bag_weights(self, tiles: Tensor, batch: Any) -> tuple[Tensor, int]:
        size = int(batch.bag_ptr.shape[0]) - 1
        score = self.tile_score(self.tile_value(tiles) * self.tile_gate(tiles)).squeeze(-1)
        return (segment_softmax(score, batch.tile_to_bag, size), size)

    def bag_representation(self, tiles: Tensor, batch: Any, extra: Tensor | None=None):
        weight, size = self.bag_weights(tiles, batch)
        pooled = segment_sum(tiles * weight.unsqueeze(-1), batch.tile_to_bag, size)
        if extra is None:
            return pooled
        return (pooled, segment_sum(extra * weight.unsqueeze(-1), batch.tile_to_bag, size))

    def forward(self, batch: Any) -> dict[str, Tensor]:
        if batch.baseline.shape[1] != self.baseline_dim:
            raise ValueError(f'baseline width {batch.baseline.shape[1]} does not match configured {self.baseline_dim}')
        anchor, entropy = self._baseline_anchor(batch)
        tiles = None
        if self.recal:
            head_input = torch.cat((batch.baseline, entropy.unsqueeze(-1)), -1)
        else:
            encoded = self.encode_nodes(batch)
            tiles = self.tile_representation(batch, encoded)
            if self.heterogeneity:
                tile_std = self.tile_heterogeneity(batch, encoded)
                if self.bag:
                    representation, pooled_std = self.bag_representation(tiles, batch, extra=tile_std)
                else:
                    representation, pooled_std = (tiles, tile_std)
                head_input = torch.cat((representation, pooled_std, batch.baseline, entropy.unsqueeze(-1)), -1)
            else:
                representation = self.bag_representation(tiles, batch) if self.bag else tiles
                head_input = torch.cat((representation, batch.baseline, entropy.unsqueeze(-1)), -1)
        delta = self.head(head_input)
        if self.classes > 1:
            delta = delta - delta.mean(dim=-1, keepdim=True)
            logits = anchor + delta
        else:
            delta = delta.squeeze(-1)
            logits = anchor + delta
        if tiles is None:
            tile_weight = batch.baseline.new_zeros((int(batch.tile_total),))
        else:
            tile_weight = self.tile_score(self.tile_value(tiles) * self.tile_gate(tiles)).squeeze(-1)
        return {'delta': delta, 'tile_weight': tile_weight, 'logits': logits}

class LocalReadoutAdapter(nn.Module):

    def __init__(self, *, width: int=64, hidden: int=16, classes: int=4):
        super().__init__()
        self.width = int(width)
        self.hidden = int(hidden)
        self.classes = int(classes)
        self.project = nn.Linear(self.width, self.hidden)
        self.query = nn.Parameter(torch.zeros(self.hidden))
        self.correct = nn.Linear(self.width, self.classes, bias=False)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.query)

    def parameter_count(self) -> int:
        return int(sum((parameter.numel() for parameter in self.parameters())))

    def forward(self, hidden: Tensor, valid: Tensor, index: Tensor, size: int, logits: Tensor) -> dict[str, Tensor]:
        if hidden.ndim != 2 or hidden.shape[1] != self.width:
            raise ValueError(f'hidden shape {tuple(hidden.shape)} does not match width {self.width}')
        valid = valid.reshape(-1).to(dtype=torch.bool)
        score = torch.tanh(self.project(hidden)) @ self.query
        score = score.masked_fill(~valid, -1000000000.0)
        weight = segment_softmax(score, index, size) * valid.to(hidden.dtype)
        normaliser = segment_sum(weight, index, size).clamp_min(1e-12)
        weight = weight / normaliser[index]
        count = segment_sum(valid.to(hidden.dtype), index, size)
        pooled = segment_sum(hidden * weight.unsqueeze(-1), index, size)
        mean = segment_sum(hidden * valid.to(hidden.dtype).unsqueeze(-1), index, size) / count.clamp_min(1.0).unsqueeze(-1)
        delta = pooled - mean
        delta = torch.where(count.unsqueeze(-1) > 1.0, delta, torch.zeros_like(delta))
        epsilon = self.correct(delta)
        if self.classes > 1:
            epsilon = epsilon - epsilon.mean(dim=-1, keepdim=True)
        else:
            epsilon = epsilon.squeeze(-1)
            logits = logits.reshape(-1)
        return {'weight': weight, 'delta': delta, 'epsilon': epsilon, 'logits': logits + epsilon, 'valid_count': count}
