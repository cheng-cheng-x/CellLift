from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from celllift.runtime import json
import numpy as np
from .dataset import DINO_DIM, SPATIAL_HW, cache_index_path
from .features import EDGE_2D, EDGE_3D, EDGE_DIM, NODE_2D, NODE_3D, NODE_DIM, apply_arm
from .foundation.spatial import load_spatial, resolve_spatial, spatial_path
from .io_utils import load_npz

@dataclass
class Sample:
    bag_id: str
    graph_ids: list[str]
    label_id: int
    fold: int
    group_id: str
    baseline: np.ndarray

@dataclass
class Batch:
    dino: torch.Tensor
    node: torch.Tensor
    edge: torch.Tensor
    edge_index: torch.Tensor
    include: torch.Tensor
    valid3d: torch.Tensor
    center_xy: torch.Tensor
    nucleus_transform: torch.Tensor
    cell_transform: torch.Tensor
    nucleus_center: torch.Tensor
    cell_center: torch.Tensor
    node_tile: torch.Tensor
    tile_bag: torch.Tensor
    labels: torch.Tensor
    baseline: torch.Tensor
    spatial: torch.Tensor | None
    spatial_tile: torch.Tensor | None
    n_tiles: int
    n_bags: int

class SceneGraph:

    def __init__(self, graph_id: str, payload: Mapping[str, np.ndarray], spatial_file: str | None=None):
        self.graph_id = graph_id
        self.dino = np.asarray(payload['dino'], np.float32)
        self.node2d = np.asarray(payload['node2d'], np.float32)
        self.node3d = np.asarray(payload['node3d'], np.float32)
        self.include = np.asarray(payload['include'], bool)
        self.valid3d = np.asarray(payload['valid3d'], bool)
        self.center_xy = np.asarray(payload['center_xy'], np.float32)
        self.center_z = np.asarray(payload['center_z'], np.float32)
        self.nucleus_center = np.asarray(payload['nucleus_center'], np.float32)
        self.cell_center = np.asarray(payload['cell_center'], np.float32)
        self.nucleus_transform = np.asarray(payload['nucleus_transform'], np.float32)
        self.cell_transform = np.asarray(payload['cell_transform'], np.float32)
        self.edge_index = np.asarray(payload['edge_index'], np.int64)
        self.edge2d = np.asarray(payload['edge2d'], np.float32)
        self.edge3d = np.asarray(payload['edge3d'], np.float32)
        self.spatial_file = spatial_file
        self._spatial = None
        self._spatial_cpu = None
        if self.dino.ndim != 2 or self.dino.shape[1] != DINO_DIM:
            raise RuntimeError(f'{graph_id} DINO must be [N,{DINO_DIM}], got {self.dino.shape}')

    @property
    def spatial(self) -> np.ndarray | None:
        if self._spatial is None and self.spatial_file:
            resolved = resolve_spatial(self.spatial_file)
            if resolved is not None:
                self._spatial = np.ascontiguousarray(load_spatial(resolved), np.float32)
        return self._spatial

    def spatial_or_zero(self) -> np.ndarray:
        cached = self._spatial_cpu
        if cached is None:
            field = self.spatial
            cached = field if field is not None else np.zeros((DINO_DIM, SPATIAL_HW, SPATIAL_HW), np.float32)
            self._spatial_cpu = cached
        return cached

    @property
    def node(self) -> np.ndarray:
        cached = getattr(self, '_node', None)
        if cached is None:
            cached = np.concatenate((self.node2d, self.node3d), axis=1)
            self._node = cached
        return cached

    @property
    def edge(self) -> np.ndarray:
        cached = getattr(self, '_edge', None)
        if cached is None:
            cached = np.concatenate((self.edge2d, self.edge3d), axis=1)
            self._edge = cached
        return cached

    @property
    def both3d(self) -> np.ndarray:
        cached = getattr(self, '_both3d', None)
        if cached is None:
            if not self.edge_index.shape[1]:
                cached = np.zeros((0,), bool)
            else:
                source, target = (self.edge_index[0], self.edge_index[1])
                cached = self.valid3d[source] & self.valid3d[target] & np.isfinite(self.edge3d).all(axis=1)
            self._both3d = cached
        return cached

def Pathish(path: str):
    from celllift.runtime import ResourcePath as Path
    return Path(path)

class SceneCache:

    def __init__(self, cfg: Mapping[str, Any], dataset: str):
        self.dataset = dataset
        self.index_path = cache_index_path(cfg, dataset)
        self.graphs: dict[str, SceneGraph] = {}
        self.entries: dict[str, dict] = {}

    def preload(self, load_spatial_maps: bool=False, workers: int=8) -> None:
        payload = json.loads(self.index_path.read_text(encoding='utf-8'))
        entries = list(payload['graphs'])

        def one(entry):
            arrays = load_npz(entry['path'])
            spatial_file = entry.get('spatial_path') or str(spatial_path(entry['path']))
            graph = SceneGraph(entry['graph_id'], arrays, spatial_file)
            if load_spatial_maps:
                _ = graph.spatial
            return (graph, entry)
        if workers <= 1 or len(entries) < 16:
            loaded = [one(entry) for entry in entries]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                loaded = list(pool.map(one, entries))
        for graph, entry in loaded:
            self.graphs[graph.graph_id] = graph
            self.entries[graph.graph_id] = entry

    def get(self, graph_id: str) -> SceneGraph:
        return self.graphs[graph_id]

def group_equal_stats(cache: SceneCache, graph_ids: Sequence[str], groups: Sequence[str]):
    by_group: dict[str, list[str]] = {}
    for graph_id, group in zip(graph_ids, groups):
        by_group.setdefault(str(group), []).append(str(graph_id))
    mean2d, second2d, mean3d, second3d = ([], [], [], [])
    for members in by_group.values():
        two, three = ([], [])
        for graph_id in members:
            graph = cache.get(graph_id)
            node = graph.node
            if graph.include.any():
                two.append(node[graph.include, :NODE_2D])
            if (graph.include & graph.valid3d).any():
                three.append(node[graph.include & graph.valid3d, NODE_2D:])
        if two:
            stacked = np.concatenate(two, 0)
            mean2d.append(stacked.mean(0))
            second2d.append((stacked * stacked).mean(0))
        if three:
            stacked = np.concatenate(three, 0)
            mean3d.append(stacked.mean(0))
            second3d.append((stacked * stacked).mean(0))
    if not mean2d:
        raise RuntimeError('no included nodes for standardization')
    m2 = np.mean(np.stack(mean2d, 0), 0)
    s2 = np.sqrt(np.maximum(np.mean(np.stack(second2d, 0), 0) - np.square(m2), 0.0))
    if mean3d:
        m3 = np.mean(np.stack(mean3d, 0), 0)
        s3 = np.sqrt(np.maximum(np.mean(np.stack(second3d, 0), 0) - np.square(m3), 0.0))
    else:
        m3 = np.zeros(NODE_3D, np.float32)
        s3 = np.ones(NODE_3D, np.float32)
    mean = np.concatenate((m2, m3)).astype(np.float32)
    std = np.maximum(np.concatenate((s2, s3)).astype(np.float32), 0.001)
    return (mean, std, mean)

def group_equal_edge_stats(cache: SceneCache, graph_ids: Sequence[str], groups: Sequence[str]):
    by_group: dict[str, list[str]] = {}
    for graph_id, group in zip(graph_ids, groups):
        by_group.setdefault(str(group), []).append(str(graph_id))
    mean2d, second2d, mean3d, second3d = ([], [], [], [])
    for members in by_group.values():
        two, three = ([], [])
        for graph_id in members:
            graph = cache.get(graph_id)
            if graph.edge2d.shape[0]:
                two.append(graph.edge2d)
            both = graph.both3d
            if both.any():
                three.append(graph.edge3d[both])
        if two:
            stacked = np.concatenate(two, 0)
            mean2d.append(stacked.mean(0))
            second2d.append((stacked * stacked).mean(0))
        if three:
            stacked = np.concatenate(three, 0)
            mean3d.append(stacked.mean(0))
            second3d.append((stacked * stacked).mean(0))
    if mean2d:
        m2 = np.mean(np.stack(mean2d, 0), 0)
        s2 = np.sqrt(np.maximum(np.mean(np.stack(second2d, 0), 0) - np.square(m2), 0.0))
    else:
        m2 = np.zeros(EDGE_2D, np.float32)
        s2 = np.ones(EDGE_2D, np.float32)
    if mean3d:
        m3 = np.mean(np.stack(mean3d, 0), 0)
        s3 = np.sqrt(np.maximum(np.mean(np.stack(second3d, 0), 0) - np.square(m3), 0.0))
    else:
        m3 = np.zeros(EDGE_3D, np.float32)
        s3 = np.ones(EDGE_3D, np.float32)
    return (m2.astype(np.float32), np.maximum(s2.astype(np.float32), 0.001), m3.astype(np.float32), np.maximum(s3.astype(np.float32), 0.001))

def standardize_graph(graph: SceneGraph, mean: np.ndarray, std: np.ndarray, fill3d: np.ndarray, edge_mean2d: np.ndarray, edge_std2d: np.ndarray, edge_mean3d: np.ndarray, edge_std3d: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = graph.node.copy()
    invalid = ~graph.valid3d
    if invalid.any():
        node[invalid, NODE_2D:] = fill3d[NODE_2D:]
    node = (node - mean) / std
    node[~graph.include] = 0.0
    edge2d = (graph.edge2d - edge_mean2d) / edge_std2d
    edge3d = np.zeros_like(graph.edge3d, np.float32)
    both = graph.both3d
    if both.any():
        edge3d[both] = (graph.edge3d[both] - edge_mean3d) / edge_std3d
    edge = np.concatenate((edge2d, edge3d), axis=1)
    node, edge = apply_arm(node, edge, arm, copy=False)
    np.nan_to_num(node, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(edge, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(node, -8.0, 8.0, out=node)
    np.clip(edge, -8.0, 8.0, out=edge)
    return (node.astype(np.float32, copy=False), edge.astype(np.float32, copy=False), graph.include)

def prepare_standardization(cache: SceneCache, **kwargs) -> None:
    packed = {}
    for graph_id, graph in cache.graphs.items():
        packed[graph_id] = standardize_graph(graph, **{k: kwargs[k] for k in ('mean', 'std', 'fill3d', 'edge_mean2d', 'edge_std2d', 'edge_mean3d', 'edge_std3d', 'arm')})
    cache._standardized = packed

def _standardized(cache, graph, **kwargs):
    packed = getattr(cache, '_standardized', None)
    if packed is not None and graph.graph_id in packed:
        return packed[graph.graph_id]
    return standardize_graph(graph, **kwargs)

def build_batch(cache: SceneCache, samples: Sequence[Sample], *, arm: str, mean: np.ndarray, std: np.ndarray, fill3d: np.ndarray, edge_mean2d: np.ndarray, edge_std2d: np.ndarray, edge_mean3d: np.ndarray, edge_std3d: np.ndarray, device: str, with_spatial: bool=False) -> Batch:
    import torch
    dinos, nodes, edges, includes, valids, indices = ([], [], [], [], [], [])
    xys, ntf, ctf, nc, cc = ([], [], [], [], [])
    node_tile, tile_bag = ([], [])
    labels, baselines, spatials = ([], [], [])
    tile = 0
    node_offset = 0
    kwargs = dict(mean=mean, std=std, fill3d=fill3d, edge_mean2d=edge_mean2d, edge_std2d=edge_std2d, edge_mean3d=edge_mean3d, edge_std3d=edge_std3d, arm=arm)
    for bag_index, sample in enumerate(samples):
        labels.append(sample.label_id)
        baselines.append(np.asarray(sample.baseline, np.float32))
        for graph_id in sample.graph_ids:
            graph = cache.get(graph_id)
            node, edge, include = _standardized(cache, graph, **kwargs)
            count = len(node)
            dinos.append(graph.dino)
            nodes.append(node)
            edges.append(edge)
            includes.append(include)
            valids.append(graph.valid3d)
            xys.append(graph.center_xy)
            ntf.append(graph.nucleus_transform)
            ctf.append(graph.cell_transform)
            nc.append(graph.nucleus_center)
            cc.append(graph.cell_center)
            if graph.edge_index.shape[1]:
                indices.append(graph.edge_index + node_offset)
            node_tile.append(np.full(count, tile, np.int64))
            tile_bag.append(bag_index)
            if with_spatial:
                spatials.append(graph.spatial_or_zero())
            node_offset += count
            tile += 1
    empty_tf = np.zeros((0, 3, 3), np.float32)
    empty_c = np.zeros((0, 3), np.float32)

    def _gpu(array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if dtype is not None:
            tensor = tensor.to(dtype)
        if str(device).startswith('cuda') and tensor.device.type == 'cpu':
            try:
                tensor = tensor.pin_memory()
            except RuntimeError:
                pass
        return tensor.to(device, non_blocking=True)
    result = Batch(dino=_gpu(np.concatenate(dinos, 0) if dinos else np.zeros((0, DINO_DIM), np.float32)), node=_gpu(np.concatenate(nodes, 0) if nodes else np.zeros((0, NODE_DIM), np.float32)), edge=_gpu(np.concatenate(edges, 0) if edges else np.zeros((0, EDGE_DIM), np.float32)), edge_index=_gpu(np.concatenate(indices, 1) if indices else np.zeros((2, 0), np.int64), torch.long), include=_gpu(np.concatenate(includes, 0) if includes else np.zeros((0,), bool)), valid3d=_gpu(np.concatenate(valids, 0) if valids else np.zeros((0,), bool)), center_xy=_gpu(np.concatenate(xys, 0) if xys else np.zeros((0, 2), np.float32)), nucleus_transform=_gpu(np.concatenate(ntf, 0) if ntf else empty_tf), cell_transform=_gpu(np.concatenate(ctf, 0) if ctf else empty_tf), nucleus_center=_gpu(np.concatenate(nc, 0) if nc else empty_c), cell_center=_gpu(np.concatenate(cc, 0) if cc else empty_c), node_tile=_gpu(np.concatenate(node_tile, 0) if node_tile else np.zeros((0,), np.int64), torch.long), tile_bag=_gpu(np.asarray(tile_bag, np.int64), torch.long), labels=_gpu(np.asarray(labels, np.int64), torch.long), baseline=_gpu(np.stack(baselines, 0) if baselines else np.zeros((0, 1), np.float32)), spatial=_gpu(np.stack(spatials, 0)) if spatials else None, spatial_tile=None, n_tiles=int(tile), n_bags=len(samples))
    result.dataset = cache.dataset
    return result

def group_batches(samples: Sequence[Sample], batch_size: int, rng: np.random.Generator) -> list[list[Sample]]:
    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)
    groups = list(by_group)
    for group in groups:
        rng.shuffle(by_group[group])
    pointers = {group: 0 for group in groups}
    ordered: list[Sample] = []
    while True:
        rng.shuffle(groups)
        progress = False
        for group in groups:
            members = by_group[group]
            index = pointers[group]
            if index < len(members):
                ordered.append(members[index])
                pointers[group] = index + 1
                progress = True
        if not progress:
            break
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]

def max_run_same_group(groups: Sequence[str]) -> int:
    longest = current = 0
    previous = None
    for group in groups:
        if group == previous:
            current += 1
        else:
            current = 1
            previous = group
        if current > longest:
            longest = current
    return int(longest)

def source_uniform_batches(samples: Sequence[Sample], batch_size: int, rng: np.random.Generator):
    if not samples:
        return ([], {'sampling': 'source_uniform_replacement', 'steps_per_epoch': 0, 'max_run_same_group': 0})
    by_group: dict[str, list[Sample]] = {}
    for sample in samples:
        by_group.setdefault(sample.group_id, []).append(sample)
    groups = np.asarray(list(by_group), dtype=object)
    members = [np.asarray(by_group[str(group)], dtype=object) for group in groups]
    n_groups = len(groups)
    n_steps = int(np.ceil(len(samples) / max(1, int(batch_size))))
    batches: list[list[Sample]] = []
    order: list[str] = []
    for _ in range(n_steps):
        if n_groups >= int(batch_size):
            chosen = rng.choice(n_groups, size=int(batch_size), replace=False)
        else:
            chosen = rng.integers(0, n_groups, size=int(batch_size))
        chunk: list[Sample] = []
        for index in chosen:
            pool = members[int(index)]
            pick = int(rng.integers(0, len(pool)))
            chunk.append(pool[pick])
            order.append(str(groups[int(index)]))
        batches.append(chunk)
    return (batches, {'sampling': 'source_uniform_replacement', 'steps_per_epoch': n_steps, 'max_run_same_group': max_run_same_group(order)})
