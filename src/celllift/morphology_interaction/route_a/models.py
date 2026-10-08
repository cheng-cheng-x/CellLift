from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from ..dataset import ARM_EDGE_3D, ARM_NODE_3D, DINO_DIM
from ..features import EDGE_2D, EDGE_3D, NODE_2D, NODE_3D
from ..foundation.ops import segment_count, segment_mean, segment_softmax, segment_std, segment_sum
from ..foundation.spatial import sample_maps_at
WIDTH = 128
HEADS = 4
HEAD_DIM = WIDTH // HEADS

class RelationAttention(nn.Module):

    def __init__(self, width: int=WIDTH, heads: int=HEADS):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.b2 = nn.Linear(EDGE_2D, heads)
        self.b3 = nn.Linear(EDGE_3D, heads)
        self.b2_extra = nn.Linear(EDGE_2D, heads)
        self.conditional_geometry = nn.Linear(EDGE_2D, width)
        self.mask_conditioning = nn.Linear(EDGE_3D, width)
        self.conditional_geometry_extra = nn.Linear(EDGE_2D, width)
        self.out = nn.Linear(width, width)

    def forward(self, hidden: Tensor, edge: Tensor, edge_index: Tensor, use_edge3d: bool) -> Tensor:
        if edge_index.shape[1] == 0:
            return hidden
        source, target = (edge_index[0], edge_index[1])
        query = self.q(hidden).view(-1, self.heads, self.head_dim)
        key = self.k(hidden).view(-1, self.heads, self.head_dim)
        value = self.v(hidden).view(-1, self.heads, self.head_dim)
        score = (query[target] * key[source]).sum(-1) / self.head_dim ** 0.5
        score = score + self.b2(edge[:, :EDGE_2D])
        if use_edge3d:
            score = score + self.b3(edge[:, EDGE_2D:])
        else:
            score = score + self.b2_extra(edge[:, :EDGE_2D])
        weight = segment_softmax(score, target, hidden.shape[0])
        message = value[source] + self.conditional_geometry(edge[:, :EDGE_2D]).view(-1, self.heads, self.head_dim)
        if use_edge3d:
            message = message + self.mask_conditioning(edge[:, EDGE_2D:]).view(-1, self.heads, self.head_dim)
        else:
            message = message + self.conditional_geometry_extra(edge[:, :EDGE_2D]).view(-1, self.heads, self.head_dim)
        gathered = (weight.unsqueeze(-1) * message).reshape(-1, self.width)
        pooled = segment_sum(gathered, target, hidden.shape[0])
        return hidden + self.out(pooled)

class RouteA(nn.Module):

    def __init__(self, *, classes: int, arm: str, width: int=WIDTH, bag: bool=False, dropout: float=0.1):
        super().__init__()
        if arm not in ARM_NODE_3D:
            raise ValueError(arm)
        self.classes = int(classes)
        self.arm = str(arm)
        self.width = int(width)
        self.bag = bool(bag)
        self.use_node3d = bool(ARM_NODE_3D[arm])
        self.use_edge3d = bool(ARM_EDGE_3D[arm])
        self.appear = nn.Sequential(nn.Linear(DINO_DIM, width), nn.SiLU())
        self.morph2d = nn.Sequential(nn.Linear(NODE_2D, width), nn.SiLU())
        self.morph3d = nn.Sequential(nn.Linear(NODE_3D + 18, width), nn.SiLU())
        self.morph2d_extra = nn.Sequential(nn.Linear(NODE_2D, width), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(width * 3, width), nn.SiLU(), nn.Dropout(dropout))
        self.physical = nn.Linear(2, width, bias=False)
        self.layer1 = RelationAttention(width)
        self.layer2 = RelationAttention(width)
        self.norm = nn.LayerNorm(width)
        self.tile = nn.Sequential(nn.Linear(width * 5, width), nn.SiLU())
        self.queries = nn.Parameter(torch.randn(2, width) * 0.02)
        self.spatial_proj = nn.Conv2d(DINO_DIM, width, 1)
        self.cell_to_grid = nn.Linear(width, width)
        self.fuse_spatial = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU())
        self.tile_score = nn.Linear(width, 1, bias=False) if bag else None
        self.tile_value = nn.Linear(width, width) if bag else None
        self.queries_bag = nn.Parameter(torch.randn(2, width) * 0.02) if bag else None
        self.head = nn.Linear(width, self.classes)

    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.parameters()))

    def _shape_tensor(self, transform: Tensor) -> Tensor:
        flat = transform.reshape(transform.shape[0], 9)
        volume = transform.det().abs().clamp_min(1e-08).log().unsqueeze(-1)
        return torch.cat((flat, volume.repeat(1, 9)), -1)[:, :18]

    def encode_nodes(self, batch: Any) -> Tensor:
        appear = self.appear(batch.dino)
        two = self.morph2d(batch.node[:, :NODE_2D])
        if self.use_node3d:
            rich = torch.cat((batch.node[:, NODE_2D:], self._shape_tensor(batch.nucleus_transform)), -1)
            three = self.morph3d(rich)
        else:
            three = self.morph2d_extra(batch.node[:, :NODE_2D])
            _ = self.morph3d(torch.zeros(batch.node.shape[0], NODE_3D + 18, device=batch.node.device, dtype=batch.node.dtype))
        hidden = self.fuse(torch.cat((appear, two, three), -1))
        hidden = hidden + self.physical(batch.node[:, :2])
        hidden = hidden * batch.include.unsqueeze(-1).to(hidden.dtype)
        h0 = hidden
        h1 = self.norm(self.layer1(hidden, batch.edge, batch.edge_index, self.use_edge3d))
        h2 = self.norm(self.layer2(h1, batch.edge, batch.edge_index, self.use_edge3d))
        return (h0, h1, h2)

    def tile_from_cells(self, h0: Tensor, h1: Tensor, h2: Tensor, batch: Any) -> Tensor:
        tiles = int(batch.n_tiles)
        parts = []
        for hidden in (h0, h1, h2):
            parts.append(segment_mean(hidden, batch.node_tile, tiles))
        std = segment_std(h2, batch.node_tile, tiles)
        summaries = []
        for query in self.queries:
            weight = segment_softmax((h2 * query).sum(-1), batch.node_tile, tiles)
            summaries.append(segment_sum(weight.unsqueeze(-1) * h2, batch.node_tile, tiles))
        return self.tile(torch.cat((parts[0], parts[1], parts[2], std, summaries[0] + summaries[1]), -1))

    def fuse_image(self, cells: Tensor, batch: Any, dataset: str) -> Tensor:
        tiles = self.tile_from_cells(*cells, batch)
        if batch.spatial is None:
            return tiles
        spatial = self.spatial_proj(batch.spatial)
        tiles_n = int(batch.n_tiles)
        sampled = sample_maps_at(spatial, batch.center_xy, batch.node_tile, dataset)
        cell = self.cell_to_grid(cells[2])
        scale = float(max(1, cell.shape[-1])) ** 0.5
        gate = torch.sigmoid((cell * sampled).sum(-1, keepdim=True) / scale)
        local = self.fuse_spatial(torch.cat((sampled, cell), -1)) * gate
        local = local * batch.include.unsqueeze(-1).to(local.dtype)
        image = segment_mean(local, batch.node_tile, tiles_n)
        present = (segment_count(batch.node_tile, tiles_n, sampled.dtype) > 0).unsqueeze(-1)
        return tiles + image * present

    def bag_representation(self, tiles: Tensor, batch: Any) -> Tensor:
        if not self.bag:
            return tiles
        score = self.tile_score(tiles).squeeze(-1)
        weight = segment_softmax(score, batch.tile_bag, int(batch.n_bags))
        value = self.tile_value(tiles)
        extra = []
        for query in self.queries_bag:
            w = segment_softmax((tiles * query).sum(-1), batch.tile_bag, int(batch.n_bags))
            extra.append(segment_sum(w.unsqueeze(-1) * tiles, batch.tile_bag, int(batch.n_bags)))
        pooled = segment_sum(weight.unsqueeze(-1) * value, batch.tile_bag, int(batch.n_bags))
        return pooled + extra[0] + extra[1]

    def forward(self, batch: Any, dataset: str='sicapv2') -> dict[str, Tensor]:
        dataset = getattr(batch, 'dataset', dataset)
        cells = self.encode_nodes(batch)
        tiles = self.fuse_image(cells, batch, dataset)
        logits = self.head(self.bag_representation(tiles, batch))
        if self.classes == 1:
            logits = logits.reshape(-1)
        return {'logits': logits}
