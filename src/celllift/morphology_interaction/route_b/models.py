from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from ..dataset import ARM_EDGE_3D, ARM_NODE_3D, DINO_DIM
from ..features import EDGE_2D, EDGE_3D, NODE_2D, NODE_3D
from ..foundation.ops import segment_sum
PROTOTYPES = 8
PCA_DIM = 32
RANK = 16
WIDTH = 64

class RouteB(nn.Module):

    def __init__(self, *, classes: int, arm: str, bag: bool=False, prototypes: int=PROTOTYPES):
        super().__init__()
        if arm not in ARM_NODE_3D:
            raise ValueError(arm)
        self.classes = int(classes)
        self.arm = str(arm)
        self.bag = bool(bag)
        self.use_3d = bool(ARM_NODE_3D[arm] or ARM_EDGE_3D[arm])
        self.k = int(prototypes)
        self.register_buffer('pca_mean', torch.zeros(DINO_DIM))
        self.register_buffer('pca_basis', torch.eye(DINO_DIM, PCA_DIM))
        self.prototypes = nn.Parameter(torch.randn(self.k, PCA_DIM) * 0.05)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        geom_dim = NODE_3D + EDGE_3D if self.use_3d else NODE_2D + EDGE_2D
        self.geom_embed = nn.Sequential(nn.Linear(geom_dim, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.appear_embed = nn.Sequential(nn.Linear(PCA_DIM, WIDTH), nn.SiLU())
        self.U = nn.Linear(WIDTH, RANK, bias=False)
        self.V = nn.Linear(WIDTH, RANK, bias=False)
        evidence = 1 + WIDTH + WIDTH + RANK
        self.cell_head = nn.Sequential(nn.Linear(evidence, WIDTH), nn.SiLU())
        self.tile_score = nn.Linear(WIDTH, 1, bias=False) if bag else None
        self.tile_value = nn.Linear(WIDTH, WIDTH) if bag else None
        self.head = nn.Linear(WIDTH, self.classes)

    def parameter_count(self) -> int:
        return sum((parameter.numel() for parameter in self.parameters()))

    def assign(self, dino: Tensor) -> tuple[Tensor, Tensor]:
        centered = dino - self.pca_mean
        coord = centered @ self.pca_basis
        dist = torch.cdist(coord, self.prototypes)
        weight = torch.softmax(-dist / self.temperature.clamp_min(0.05), -1)
        return (coord, weight)

    def geometry_token(self, batch: Any) -> Tensor:
        if self.use_3d:
            node = batch.node[:, NODE_2D:]
            if batch.edge_index.shape[1]:
                source, target = (batch.edge_index[0], batch.edge_index[1])
                rel = segment_sum(batch.edge[:, EDGE_2D:], target, batch.node.shape[0])
                degree = segment_sum(torch.ones(target.shape[0], 1, device=target.device), target, batch.node.shape[0]).clamp_min(1.0)
                rel = rel / degree
            else:
                rel = node.new_zeros((node.shape[0], EDGE_3D))
            return torch.cat((node, rel), -1)
        node = batch.node[:, :NODE_2D]
        if batch.edge_index.shape[1]:
            source, target = (batch.edge_index[0], batch.edge_index[1])
            rel = segment_sum(batch.edge[:, :EDGE_2D], target, batch.node.shape[0])
            degree = segment_sum(torch.ones(target.shape[0], 1, device=target.device), target, batch.node.shape[0]).clamp_min(1.0)
            rel = rel / degree
        else:
            rel = node.new_zeros((node.shape[0], EDGE_2D))
        return torch.cat((node, rel), -1)

    def forward(self, batch: Any, dataset: str='sicapv2') -> dict[str, Tensor]:
        del dataset
        coord, assign = self.assign(batch.dino)
        geom = self.geom_embed(self.geometry_token(batch))
        appear = self.appear_embed(coord)
        include = batch.include.to(assign.dtype).unsqueeze(-1)
        assign = assign * include
        mass = assign.new_zeros((int(batch.n_tiles), self.k))
        mass.index_add_(0, batch.node_tile, assign)
        tile_nodes = assign.new_zeros((int(batch.n_tiles), 1))
        tile_nodes.index_add_(0, batch.node_tile, include)
        frac = mass / tile_nodes.clamp_min(1.0)
        u = assign.new_zeros((int(batch.n_tiles), self.k, appear.shape[-1]))
        v = assign.new_zeros((int(batch.n_tiles), self.k, geom.shape[-1]))
        u.index_add_(0, batch.node_tile, assign.unsqueeze(-1) * appear.unsqueeze(1))
        v.index_add_(0, batch.node_tile, assign.unsqueeze(-1) * geom.unsqueeze(1))
        denom = mass.unsqueeze(-1).clamp_min(1.0)
        u = u / denom
        v = v / denom
        interact = self.U(u) * self.V(v)
        evidence = torch.cat((frac.unsqueeze(-1), u, v, interact), -1)
        proto = self.cell_head(evidence).sum(1)
        if self.bag:
            score = self.tile_score(proto).squeeze(-1)
            from ..foundation.ops import segment_softmax, segment_sum as ssum
            weight = segment_softmax(score, batch.tile_bag, int(batch.n_bags))
            proto = ssum(weight.unsqueeze(-1) * self.tile_value(proto), batch.tile_bag, int(batch.n_bags))
        logits = self.head(proto)
        if self.classes == 1:
            logits = logits.reshape(-1)
        return {'logits': logits, 'frac': frac}

def fit_prototypes(dino: np.ndarray, k: int=PROTOTYPES, dim: int=PCA_DIM, seed: int=42):
    centered = dino - dino.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:dim].T.astype(np.float32)
    coord = centered @ basis
    rng = np.random.default_rng(seed)
    centers = coord[rng.choice(len(coord), size=k, replace=len(coord) < k)]
    for _ in range(12):
        dist = ((coord[:, None] - centers[None]) ** 2).sum(-1)
        assign = np.argmin(dist, 1)
        for index in range(k):
            members = coord[assign == index]
            if len(members):
                centers[index] = members.mean(0)
    return (dino.mean(0).astype(np.float32), basis, centers.astype(np.float32))
