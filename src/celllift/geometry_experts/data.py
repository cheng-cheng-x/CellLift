from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.runtime import json
import numpy as np
from celllift.runtime import torch
from .dataset import cache_index_path
from .features import EDGE_DIM, NODE_DIM, apply_arm
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
    node: torch.Tensor
    edge: torch.Tensor
    edge_index: torch.Tensor
    include: torch.Tensor
    valid3d: torch.Tensor
    node_tile: torch.Tensor
    tile_bag: torch.Tensor
    labels: torch.Tensor
    baseline: torch.Tensor
    n_tiles: int
    n_bags: int

class SceneGraph:

    def __init__(self, graph_id: str, payload: Mapping[str, np.ndarray]):
        self.graph_id = graph_id
        self.node2d = np.asarray(payload['node2d'], np.float32)
        self.node3d = np.asarray(payload['node3d'], np.float32)
        self.include = np.asarray(payload['include'], bool)
        self.valid3d = np.asarray(payload['valid3d'], bool)
        self.center_xy = np.asarray(payload['center_xy'], np.float32)
        self.center_z = np.asarray(payload['center_z'], np.float32)
        self.edge_index = np.asarray(payload['edge_index'], np.int64)
        self.edge2d = np.asarray(payload['edge2d'], np.float32)
        self.edge3d = np.asarray(payload['edge3d'], np.float32)

    @property
    def node(self) -> np.ndarray:
        return np.concatenate((self.node2d, self.node3d), axis=1)

    @property
    def edge(self) -> np.ndarray:
        return np.concatenate((self.edge2d, self.edge3d), axis=1)

class SceneCache:

    def __init__(self, cfg: Mapping[str, Any], dataset: str):
        self.dataset = dataset
        self.index_path = cache_index_path(cfg, dataset)
        self.graphs: dict[str, SceneGraph] = {}
        self.entries: dict[str, dict] = {}

    def preload(self) -> None:
        payload = json.loads(self.index_path.read_text(encoding='utf-8'))
        for entry in payload['graphs']:
            arrays = load_npz(entry['path'])
            graph = SceneGraph(entry['graph_id'], arrays)
            self.graphs[graph.graph_id] = graph
            self.entries[graph.graph_id] = entry

    def get(self, graph_id: str) -> SceneGraph:
        return self.graphs[graph_id]

def group_equal_stats(cache: SceneCache, graph_ids: Sequence[str], groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                two.append(node[graph.include, :38])
            if (graph.include & graph.valid3d).any():
                three.append(node[graph.include & graph.valid3d, 38:])
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
        m3 = np.zeros(12, np.float32)
        s3 = np.ones(12, np.float32)
    mean = np.concatenate((m2, m3)).astype(np.float32)
    std = np.maximum(np.concatenate((s2, s3)).astype(np.float32), 0.001)
    return (mean, std, mean)

def standardize_graph(graph: SceneGraph, mean: np.ndarray, std: np.ndarray, fill3d: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = graph.node.copy()
    invalid = ~graph.valid3d
    if invalid.any():
        node[invalid, 38:] = fill3d[38:]
    node = (node - mean) / std
    node[~graph.include] = 0.0
    edge = graph.edge.copy()
    node, edge = apply_arm(node, edge, arm)
    node = np.clip(np.nan_to_num(node, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    edge = np.clip(np.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    return (node.astype(np.float32), edge.astype(np.float32), graph.include)

def build_batch(cache: SceneCache, samples: Sequence[Sample], *, arm: str, mean: np.ndarray, std: np.ndarray, fill3d: np.ndarray, device: str) -> Batch:
    nodes, edges, includes, valids, indices = ([], [], [], [], [])
    node_tile, tile_bag = ([], [])
    labels, baselines = ([], [])
    tile = 0
    node_offset = 0
    for bag_index, sample in enumerate(samples):
        labels.append(sample.label_id)
        baselines.append(np.asarray(sample.baseline, np.float32))
        for graph_id in sample.graph_ids:
            graph = cache.get(graph_id)
            node, edge, include = standardize_graph(graph, mean, std, fill3d, arm)
            count = len(node)
            nodes.append(node)
            edges.append(edge)
            includes.append(include)
            valids.append(graph.valid3d)
            if graph.edge_index.shape[1]:
                indices.append(graph.edge_index + node_offset)
            node_tile.append(np.full(count, tile, np.int64))
            tile_bag.append(bag_index)
            node_offset += count
            tile += 1
    node = np.concatenate(nodes, 0) if nodes else np.zeros((0, NODE_DIM), np.float32)
    edge = np.concatenate(edges, 0) if edges else np.zeros((0, EDGE_DIM), np.float32)
    include = np.concatenate(includes, 0) if includes else np.zeros((0,), bool)
    valid3d = np.concatenate(valids, 0) if valids else np.zeros((0,), bool)
    edge_index = np.concatenate(indices, 1) if indices else np.zeros((2, 0), np.int64)
    node_tile_arr = np.concatenate(node_tile, 0) if node_tile else np.zeros((0,), np.int64)
    tile_bag_arr = np.asarray(tile_bag, np.int64)
    baseline = np.stack(baselines, 0) if baselines else np.zeros((0, 1), np.float32)
    return Batch(node=torch.as_tensor(node, device=device), edge=torch.as_tensor(edge, device=device), edge_index=torch.as_tensor(edge_index, device=device, dtype=torch.long), include=torch.as_tensor(include, device=device), valid3d=torch.as_tensor(valid3d, device=device), node_tile=torch.as_tensor(node_tile_arr, device=device, dtype=torch.long), tile_bag=torch.as_tensor(tile_bag_arr, device=device, dtype=torch.long), labels=torch.as_tensor(np.asarray(labels, np.int64), device=device, dtype=torch.long), baseline=torch.as_tensor(baseline, device=device), n_tiles=int(tile), n_bags=len(samples))

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

def source_uniform_batches(samples: Sequence[Sample], batch_size: int, rng: np.random.Generator) -> tuple[list[list[Sample]], dict[str, Any]]:
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
