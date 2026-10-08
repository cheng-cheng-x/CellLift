from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from .features import EDGE_DIM, NODE_DIM
PROB_CLIP = 1e-07

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
    variance = segment_mean(centered.square(), index, size).clamp_min(1e-06)
    return torch.sqrt(variance)

def segment_softmax(scores: Tensor, index: Tensor, size: int) -> Tensor:
    if scores.numel() == 0:
        return scores
    maximum = scores.new_full((size,), -torch.inf)
    maximum = maximum.scatter_reduce(0, index, scores, reduce='amax', include_self=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.exp(scores - maximum[index])
    return exponent / segment_sum(exponent, index, size).clamp_min(1e-12)[index]

class GeometryExpert(nn.Module):

    def __init__(self, *, classes: int, width: int=64, message: int=32, dropout: float=0.1, bag: bool=False):
        super().__init__()
        self.classes = int(classes)
        self.width = int(width)
        self.bag = bool(bag)
        self.node_encoder = nn.Sequential(nn.Linear(NODE_DIM, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.SiLU())
        payload = 2 * width + EDGE_DIM
        self.message_dim = int(message)
        self.message = nn.Sequential(nn.Linear(payload, message), nn.SiLU(), nn.Linear(message, message))
        self.local = nn.Sequential(nn.Linear(width + message, width), nn.SiLU())
        self.patch = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.SiLU())
        self.tile_score = nn.Linear(width, 1, bias=False) if bag else None
        self.tile_value = nn.Sequential(nn.Linear(width, width), nn.Tanh()) if bag else None
        self.tile_gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid()) if bag else None
        self.head = nn.Linear(width, self.classes)

    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.parameters()))

    def encode_nodes(self, batch: Any) -> Tensor:
        hidden = self.node_encoder(batch.node)
        if batch.edge_index.shape[1] == 0:
            neighbour = hidden.new_zeros((hidden.shape[0], self.message_dim))
        else:
            source, target = (batch.edge_index[0], batch.edge_index[1])
            payload = torch.cat((hidden[source], hidden[target], batch.edge), -1)
            messages = self.message(payload)
            neighbour = segment_mean(messages, target, hidden.shape[0])
        local = self.local(torch.cat((hidden, neighbour), -1))
        valid = batch.include.unsqueeze(-1).to(local.dtype)
        local = local * valid
        return local

    def tile_representation(self, local: Tensor, batch: Any) -> Tensor:
        tiles = int(batch.n_tiles)
        mean = segment_mean(local, batch.node_tile, tiles)
        std = segment_std(local, batch.node_tile, tiles)
        count = segment_sum(batch.include.to(local.dtype), batch.node_tile, tiles).clamp_min(0.0)
        return self.patch(torch.cat((mean, std, torch.log1p(count).unsqueeze(-1)), -1))

    def bag_representation(self, tiles: Tensor, batch: Any) -> Tensor:
        if not self.bag:
            return tiles
        score = self.tile_score(tiles).squeeze(-1)
        weight = segment_softmax(score, batch.tile_bag, int(batch.n_bags))
        value = self.tile_value(tiles) * self.tile_gate(tiles)
        return segment_sum(weight.unsqueeze(-1) * value, batch.tile_bag, int(batch.n_bags))

    def forward(self, batch: Any) -> dict[str, Tensor]:
        local = self.encode_nodes(batch)
        tiles = self.tile_representation(local, batch)
        pooled = self.bag_representation(tiles, batch)
        logits = self.head(pooled)
        return {'logits': logits, 'representation': pooled}

class FusionHead(nn.Module):

    def __init__(self, classes: int):
        super().__init__()
        self.classes = int(classes)
        self.a = nn.Parameter(torch.zeros(self.classes))

    @property
    def scale(self) -> Tensor:
        return self.a.clamp_min(0.0)

    def project_(self) -> None:
        with torch.no_grad():
            self.a.clamp_(min=0.0)

    def forward(self, baseline_logits: Tensor, evidence: Tensor) -> Tensor:
        if self.classes == 1:
            baseline = baseline_logits.reshape(-1)
            scaled = (self.scale.reshape(-1) * evidence.reshape(-1, self.classes)).reshape(-1)
            return baseline + scaled
        scaled = self.scale * evidence
        return baseline_logits + scaled - scaled.mean(dim=-1, keepdim=True)
