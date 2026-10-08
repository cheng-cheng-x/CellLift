from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from ..dataset import DINO_DIM, SPATIAL_HW
from ..foundation.ops import segment_softmax, segment_sum
WIDTH = 64

class RouteC(nn.Module):

    def __init__(self, *, classes: int, arm: str, bag: bool=False):
        super().__init__()
        self.classes = int(classes)
        self.arm = str(arm)
        self.bag = bool(bag)
        self.encoder = nn.Sequential(nn.Conv2d(10, WIDTH, 3, padding=1), nn.SiLU(), nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.SiLU())
        self.mix = nn.Conv2d(WIDTH, WIDTH, 1)
        self.image = nn.Conv2d(DINO_DIM, WIDTH, 1)
        self.fuse1 = nn.Sequential(nn.Conv2d(WIDTH * 2, WIDTH, 3, padding=1), nn.SiLU())
        self.down = nn.AvgPool2d(2)
        self.fuse2 = nn.Sequential(nn.Conv2d(WIDTH * 2, WIDTH, 3, padding=1), nn.SiLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tile_score = nn.Linear(WIDTH, 1, bias=False) if bag else None
        self.tile_value = nn.Linear(WIDTH, WIDTH) if bag else None
        self.head = nn.Linear(WIDTH, self.classes)

    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.parameters()))

    def encode_tile(self, field: Tensor, spatial: Tensor | None) -> Tensor:
        geom = self.mix(self.encoder(field))
        if spatial is None:
            image = geom
        else:
            image = self.image(spatial)
            if image.shape[-1] != geom.shape[-1]:
                image = torch.nn.functional.interpolate(image, size=geom.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fuse1(torch.cat((geom, image), 1))
        low = self.fuse2(torch.cat((self.down(fused), self.down(image)), 1))
        return self.pool(fused + torch.nn.functional.interpolate(low, size=fused.shape[-2:], mode='nearest')).flatten(1)

    def forward(self, batch: Any, dataset: str='sicapv2') -> dict[str, Tensor]:
        del dataset
        field = getattr(batch, 'field', None)
        if field is None:
            field = batch.node.new_zeros((int(batch.n_tiles), 10, SPATIAL_HW, SPATIAL_HW))
        tiles = self.encode_tile(field, batch.spatial)
        if self.bag:
            score = self.tile_score(tiles).squeeze(-1)
            weight = segment_softmax(score, batch.tile_bag, int(batch.n_bags))
            tiles = segment_sum(weight.unsqueeze(-1) * self.tile_value(tiles), batch.tile_bag, int(batch.n_bags))
        logits = self.head(tiles)
        if self.classes == 1:
            logits = logits.reshape(-1)
        return {'logits': logits}
