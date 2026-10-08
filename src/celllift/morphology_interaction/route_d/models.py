from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from ..dataset import ARM_NODE_3D, DINO_DIM, SPATIAL_HW
from ..features import EDGE_3D, NODE_2D, NODE_3D
from ..foundation.ops import segment_mean, segment_softmax, segment_sum
from ..foundation.spatial import sample_maps_at
WIDTH = 64

class RouteD(nn.Module):

    def __init__(self, *, classes: int, arm: str, bag: bool=False):
        super().__init__()
        self.classes = int(classes)
        self.arm = str(arm)
        self.bag = bool(bag)
        self.use_3d = bool(ARM_NODE_3D[arm])
        self.adapter = nn.Sequential(nn.Conv2d(DINO_DIM, WIDTH, 1), nn.SiLU(), nn.Conv2d(WIDTH, DINO_DIM, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.shape_head = nn.Linear(DINO_DIM, NODE_3D if self.use_3d else NODE_2D)
        self.rel_head = nn.Linear(DINO_DIM * 2, EDGE_3D if self.use_3d else 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tile = nn.Sequential(nn.Linear(DINO_DIM, WIDTH), nn.SiLU())
        self.tile_score = nn.Linear(WIDTH, 1, bias=False) if bag else None
        self.tile_value = nn.Linear(WIDTH, WIDTH) if bag else None
        self.head = nn.Linear(WIDTH, self.classes)

    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.parameters()))

    def adapted(self, spatial: Tensor) -> Tensor:
        return spatial + self.adapter(spatial)

    def read_cells(self, field: Tensor, xy: Tensor, dataset: str, tile_index: Tensor | None=None) -> Tensor:
        if tile_index is None:
            tile_index = torch.zeros(xy.shape[0], dtype=torch.long, device=field.device)
        return sample_maps_at(field, xy, tile_index, dataset)

    def encode_tiles(self, batch: Any, dataset: str) -> Tensor:
        if batch.spatial is None:
            return self.tile(segment_mean(batch.dino, batch.node_tile, int(batch.n_tiles)))
        adapted = self.adapted(batch.spatial)
        return self.tile(self.pool(adapted).flatten(1))

    def forward(self, batch: Any, dataset: str='sicapv2') -> dict[str, Tensor]:
        dataset = getattr(batch, 'dataset', dataset)
        tiles = self.encode_tiles(batch, dataset)
        if self.bag:
            score = self.tile_score(tiles).squeeze(-1)
            weight = segment_softmax(score, batch.tile_bag, int(batch.n_bags))
            tiles = segment_sum(weight.unsqueeze(-1) * self.tile_value(tiles), batch.tile_bag, int(batch.n_bags))
        logits = self.head(tiles)
        if self.classes == 1:
            logits = logits.reshape(-1)
        return {'logits': logits}

    def pretrain_losses(self, batch: Any, dataset: str) -> Tensor:
        if batch.spatial is None:
            tokens = batch.dino
        else:
            adapted = self.adapted(batch.spatial)
            tokens = self.read_cells(adapted, batch.center_xy, dataset, batch.node_tile)
            if tokens.shape[0] == 0:
                tokens = batch.dino
        target = batch.node[:, NODE_2D:] if self.use_3d else batch.node[:, :NODE_2D]
        shape = torch.nn.functional.smooth_l1_loss(self.shape_head(tokens), target)
        if batch.edge_index.shape[1]:
            source, target_e = (batch.edge_index[0], batch.edge_index[1])
            pair = torch.cat((tokens[source], tokens[target_e]), -1)
            rel_t = batch.edge[:, 4:] if self.use_3d else batch.edge[:, :4]
            relation = torch.nn.functional.smooth_l1_loss(self.rel_head(pair), rel_t)
        else:
            relation = tokens.sum() * 0
        return shape + relation
