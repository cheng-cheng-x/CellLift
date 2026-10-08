from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.geometry_experts.features import apply_arm
from celllift.geometry_experts.models import GeometryExpert
from celllift.morphology_interaction.route_b.models import RouteB, fit_prototypes
from .config import DINO_DIM, DROPOUT, GEOM_EMBED, NODE_DIM, PROTO_WIDTH, TILE_EMBED
from .geometry import DeepSetsTile, DualStreamTile, isolate_tokens, pad_sets, sanitize_finite
from .mil import ImageProjection, PatientMIL

def _chunk_bounds(n: int, chunk: int) -> list[tuple[int, int]]:
    step = max(1, int(chunk))
    return [(start, min(n, start + step)) for start in range(0, n, step)]

def _maybe_checkpoint(fn, *args, enabled: bool):
    if enabled and any((getattr(item, 'requires_grad', False) for item in args if torch.is_tensor(item))):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)

class ImageMIL(nn.Module):

    def __init__(self, classes: int):
        super().__init__()
        self.proj = ImageProjection()
        self.mil = PatientMIL(TILE_EMBED, classes)
        self.has_geometry = False

    def forward(self, dino: Tensor, **_unused) -> Tensor:
        logits, _ = self.mil(self.proj(dino))
        return logits

class GeometryMIL(nn.Module):

    def __init__(self, classes: int, mode: str):
        super().__init__()
        self.mode = mode
        self.combined = mode in {'g23', 'g2r'}
        self.single = DeepSetsTile() if not self.combined else None
        self.dual = DualStreamTile() if self.combined else None
        self.mil = PatientMIL(GEOM_EMBED, classes)
        self.has_geometry = True

    def encode_chunk(self, packed: dict[str, Tensor], packed_3d: dict[str, Tensor] | None=None) -> Tensor:
        if self.combined:
            return self.dual(packed, packed_3d)
        return self.single(packed)

    def forward(self, tile_tokens: Sequence[dict[str, Any]], *, chunk: int=8, node_cap: int | None=None) -> Tensor:
        del node_cap
        embeds = []
        device = next(self.parameters()).device
        for start, stop in _chunk_bounds(len(tile_tokens), chunk):
            rows = tile_tokens[start:stop]
            if self.combined:
                packed_2d = pad_sets([row['g2'] for row in rows], device)
                packed_3d = pad_sets([row['g3'] for row in rows], device)
                embed = self.encode_chunk(packed_2d, packed_3d)
            else:
                packed = pad_sets([row['single'] for row in rows], device)
                embed = self.encode_chunk(packed)
            embeds.append(embed)
        logits, _ = self.mil(torch.cat(embeds, 0))
        return logits

class FusionMIL(nn.Module):

    def __init__(self, classes: int, mode: str):
        super().__init__()
        self.mode = mode
        self.image = ImageProjection()
        self.geometry = DeepSetsTile()
        self.fuse = nn.Sequential(nn.LayerNorm(TILE_EMBED + GEOM_EMBED), nn.Linear(TILE_EMBED + GEOM_EMBED, TILE_EMBED), nn.ReLU(inplace=True), nn.Dropout(DROPOUT))
        self.mil = PatientMIL(TILE_EMBED, classes)
        self.has_geometry = True

    def geometry_parameters(self):
        return self.geometry.parameters()

    def encode_geom(self, packed: dict[str, Tensor]) -> Tensor:
        return self.geometry(packed)

    def forward(self, dino: Tensor, tile_tokens: Sequence[dict[str, Any]], *, chunk: int=8, node_cap: int | None=None) -> Tensor:
        del node_cap
        device = dino.device
        geom = []
        for start, stop in _chunk_bounds(len(tile_tokens), chunk):
            packed = pad_sets([row['single'] for row in tile_tokens[start:stop]], device)
            geom.append(self.encode_geom(packed))
        fused = self.fuse(torch.cat((self.image(dino), torch.cat(geom, 0)), dim=-1))
        logits, _ = self.mil(fused)
        return logits

class PrototypeMIL(nn.Module):

    def __init__(self, classes: int, route_arm: str):
        super().__init__()
        self.route = RouteB(classes=classes, arm=route_arm, bag=False)
        self.image = nn.Sequential(nn.LayerNorm(DINO_DIM), nn.Linear(DINO_DIM, PROTO_WIDTH), nn.ReLU(inplace=True), nn.Dropout(DROPOUT))
        self.mil = PatientMIL(PROTO_WIDTH + PROTO_WIDTH, classes)
        self.has_geometry = True

    def load_prototypes(self, mean: np.ndarray, basis: np.ndarray, centers: np.ndarray) -> None:
        with torch.no_grad():
            self.route.pca_mean.copy_(torch.as_tensor(mean))
            self.route.pca_basis.copy_(torch.as_tensor(basis))
            self.route.prototypes.copy_(torch.as_tensor(centers))

    def forward(self, dino: Tensor, scenes: Sequence[dict[str, Any]], *, chunk: int=8, node_cap: int | None=None) -> Tensor:
        del chunk, node_cap
        proto = self._proto_from_route(self._batch(scenes, dino.device))
        fused = torch.cat((proto, self.image(dino)), dim=-1)
        logits, _ = self.mil(fused)
        return logits

    def _proto_from_route(self, batch) -> Tensor:
        coord, assign = self.route.assign(batch.dino)
        geom = self.route.geom_embed(self.route.geometry_token(batch))
        appear = self.route.appear_embed(coord)
        include = batch.include.to(assign.dtype).unsqueeze(-1)
        assign = assign * include
        mass = assign.new_zeros((int(batch.n_tiles), self.route.k))
        mass.index_add_(0, batch.node_tile, assign)
        tile_nodes = assign.new_zeros((int(batch.n_tiles), 1))
        tile_nodes.index_add_(0, batch.node_tile, include)
        frac = mass / tile_nodes.clamp_min(1.0)
        u = assign.new_zeros((int(batch.n_tiles), self.route.k, appear.shape[-1]))
        v = assign.new_zeros((int(batch.n_tiles), self.route.k, geom.shape[-1]))
        u.index_add_(0, batch.node_tile, assign.unsqueeze(-1) * appear.unsqueeze(1))
        v.index_add_(0, batch.node_tile, assign.unsqueeze(-1) * geom.unsqueeze(1))
        denom = mass.unsqueeze(-1).clamp_min(1.0)
        interact = self.route.U(u / denom) * self.route.V(v / denom)
        evidence = torch.cat((frac.unsqueeze(-1), u / denom, v / denom, interact), -1)
        return self.route.cell_head(evidence).sum(1)

    def _batch(self, scenes: Sequence[dict[str, Any]], device) -> SimpleNamespace:
        nodes, edges, eidx, dino, include, node_tile = ([], [], [], [], [], [])
        cursor = 0
        for index, scene in enumerate(scenes):
            node = sanitize_finite(np.concatenate((np.asarray(scene['node2d'], np.float32), np.asarray(scene['node3d'], np.float32)), 1))
            edge = sanitize_finite(np.concatenate((np.asarray(scene['edge2d'], np.float32), np.asarray(scene['edge3d'], np.float32)), 1))
            n = len(node)
            if n == 0:
                node = np.zeros((1, NODE_DIM), np.float32)
                edge = np.zeros((0, 12), np.float32)
                ei = np.zeros((2, 0), np.int64)
                d = np.zeros((1, DINO_DIM), np.float32)
                inc = np.zeros(1, bool)
                n = 1
            else:
                ei = np.asarray(scene['edge_index'], np.int64)
                d = sanitize_finite(np.asarray(scene['dino'], np.float32))
                inc = np.asarray(scene['include'], bool)
            nodes.append(node)
            edges.append(edge)
            eidx.append(ei + cursor if ei.size else ei)
            dino.append(d)
            include.append(inc)
            node_tile.append(np.full(n, index, np.int64))
            cursor += n
        return SimpleNamespace(node=torch.as_tensor(np.concatenate(nodes, 0), device=device), edge=torch.as_tensor(np.concatenate(edges, 0) if edges else np.zeros((0, 12), np.float32), device=device), edge_index=torch.as_tensor(np.concatenate(eidx, 1) if eidx else np.zeros((2, 0), np.int64), device=device, dtype=torch.long), dino=torch.as_tensor(np.concatenate(dino, 0), device=device), include=torch.as_tensor(np.concatenate(include, 0), device=device), node_tile=torch.as_tensor(np.concatenate(node_tile, 0), device=device, dtype=torch.long), n_tiles=len(scenes))

class ExpertMIL(nn.Module):

    def __init__(self, classes: int, arm: str):
        super().__init__()
        self.arm = arm
        self.expert = GeometryExpert(classes=classes, width=64, dropout=DROPOUT, bag=False)
        self.mil = PatientMIL(64, classes)
        self.has_geometry = True

    def forward(self, scenes: Sequence[dict[str, Any]], *, chunk: int=8, node_cap: int | None=None) -> Tensor:
        del chunk, node_cap
        batch = self._batch(scenes)
        tiles = self.expert.tile_representation(self.expert.encode_nodes(batch), batch)
        logits, _ = self.mil(tiles)
        return logits

    def _batch(self, scenes: Sequence[dict[str, Any]]):
        device = next(self.parameters()).device
        nodes, edges, eidx, include, node_tile = ([], [], [], [], [])
        cursor = 0
        for index, scene in enumerate(scenes):
            node = np.concatenate((np.asarray(scene['node2d'], np.float32), np.asarray(scene['node3d'], np.float32)), 1)
            edge = np.concatenate((np.asarray(scene['edge2d'], np.float32), np.asarray(scene['edge3d'], np.float32)), 1)
            node, edge = apply_arm(sanitize_finite(node), sanitize_finite(edge), self.arm)
            n = len(node)
            if n == 0:
                node = np.zeros((1, NODE_DIM), np.float32)
                edge = np.zeros((0, 12), np.float32)
                ei = np.zeros((2, 0), np.int64)
                inc = np.zeros(1, bool)
                n = 1
            else:
                ei = np.asarray(scene['edge_index'], np.int64)
                inc = np.asarray(scene['include'], bool)
            nodes.append(node)
            edges.append(edge)
            eidx.append(ei + cursor if ei.size else ei)
            include.append(inc)
            node_tile.append(np.full(n, index, np.int64))
            cursor += n
        return SimpleNamespace(node=torch.as_tensor(np.concatenate(nodes, 0), device=device), edge=torch.as_tensor(np.concatenate(edges, 0) if edges else np.zeros((0, 12), np.float32), device=device), edge_index=torch.as_tensor(np.concatenate(eidx, 1) if eidx else np.zeros((2, 0), np.int64), device=device, dtype=torch.long), include=torch.as_tensor(np.concatenate(include, 0), device=device), node_tile=torch.as_tensor(np.concatenate(node_tile, 0), device=device, dtype=torch.long), n_tiles=len(scenes))

def build_model(arm: str, classes: int, recipe: dict[str, Any]) -> nn.Module:
    kind = recipe['kind']
    if kind == 'B':
        return ImageMIL(classes)
    if kind == 'G':
        return GeometryMIL(classes, recipe.get('mode') or arm.lower())
    if kind == 'H':
        return FusionMIL(classes, recipe.get('mode') or ('g2' if arm == 'H2' else 'g3'))
    if kind == 'P':
        return PrototypeMIL(classes, recipe['route_arm'])
    if kind == 'E':
        return ExpertMIL(classes, arm)
    raise ValueError(arm)

def _included_dino(scene: dict[str, Any]) -> np.ndarray:
    dino = np.asarray(scene.get('dino'), np.float32)
    include = np.asarray(scene.get('include'), bool)
    if dino.size == 0:
        return np.zeros((0, DINO_DIM), np.float32)
    return dino[include] if include.shape[0] == len(dino) else dino

def collect_fit_nucleus_dino(scenes: Sequence[dict[str, Any]], limit: int=200000, seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = [_included_dino(scene) for scene in scenes]
    blocks = [block for block in blocks if len(block)]
    if not blocks:
        return np.zeros((8, DINO_DIM), np.float32)
    stacked = np.concatenate(blocks, 0)
    if len(stacked) > limit:
        stacked = stacked[rng.choice(len(stacked), size=limit, replace=False)]
    return stacked

def collect_fit_nucleus_dino_from_store(store, bags: Sequence[dict[str, Any]], limit: int=200000, seed: int=42) -> np.ndarray:
    counts: list[list[int]] = []
    for bag in bags:
        counts.append([len(_included_dino(scene)) for scene in store.scenes(bag)])
        store._scenes.pop(bag['patient_id'], None)
    total = int(sum((sum(rows) for rows in counts)))
    if total == 0:
        return np.zeros((8, DINO_DIM), np.float32)
    rng = np.random.default_rng(seed)
    keep = np.ones(total, bool) if total <= limit else np.zeros(total, bool)
    if total > limit:
        keep[rng.choice(total, size=limit, replace=False)] = True
    out = np.zeros((int(keep.sum()), DINO_DIM), np.float32)
    cursor = 0
    written = 0
    for bag, rows in zip(bags, counts):
        for scene, count in zip(store.scenes(bag), rows):
            sl = slice(cursor, cursor + count)
            if keep[sl].any():
                selected = _included_dino(scene)[keep[sl]]
                out[written:written + len(selected)] = selected
                written += len(selected)
            cursor += count
        store._scenes.pop(bag['patient_id'], None)
    return out

def isolate_tile(row: dict[str, Any], mode: str, residual9: np.ndarray | None=None, node_cap: int | None=None, scale=None) -> dict[str, Any]:
    kwargs = {'nucleus_id': row.get('nucleus_id'), 'node_cap': node_cap, 'scale': scale}
    if mode in {'g23', 'g2r'}:
        three = 'g3' if mode == 'g23' else 'gr'
        return {'g2': isolate_tokens(row['rays'], row['node3d'], row['include'], row['valid3d'], 'g2', residual9, **kwargs), 'g3': isolate_tokens(row['rays'], row['node3d'], row['include'], row['valid3d'], three, residual9, **kwargs)}
    return {'single': isolate_tokens(row['rays'], row['node3d'], row['include'], row['valid3d'], mode, residual9, **kwargs)}
