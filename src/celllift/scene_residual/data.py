from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass, field
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .io_utils import load_npz, read_json
NODE_DIM = 13
NCR_CHANNEL = 8
EDGE_DIM = 5
DISTANCE_SCALE_UM = 10.0

def apply_z_permutation(center_z: np.ndarray, order: np.ndarray) -> np.ndarray:
    center_z = np.asarray(center_z)
    order = np.asarray(order)
    if order.shape != (len(center_z),):
        raise ValueError(f'permutation length {tuple(order.shape)} does not match {len(center_z)} nodes')
    return center_z[order]

@dataclass
class SceneGraph:
    graph_id: str
    node13: np.ndarray
    center_xy: np.ndarray
    center_z: np.ndarray
    edge_index: np.ndarray
    edge_base: np.ndarray
    delta_z: np.ndarray
    include: np.ndarray
    valid_ncr: np.ndarray
    metadata: dict

@dataclass
class Sample:
    bag_id: str
    graph_ids: list[str]
    label_id: int
    fold: int
    group_id: str
    baseline: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    tile_rgb: np.ndarray | None = None
    'Per-tile frozen baseline decision used by the CRC tile route: [probability, hard vote].'

def load_graph(path: str | Path, metadata: Mapping[str, Any] | None=None, graph_id: str | None=None) -> SceneGraph:
    arrays = load_npz(path)
    return SceneGraph(graph_id=str(graph_id or (metadata or {}).get('graph_id') or Path(path).stem), node13=arrays['node13'], center_xy=arrays['center_xy'], center_z=arrays['center_z'], edge_index=arrays['edge_index'].astype(np.int64), edge_base=arrays['edge_base'], delta_z=arrays['delta_z'], include=arrays['include'], valid_ncr=arrays['valid_ncr'], metadata=dict(metadata or {}))

class SceneCache:

    def __init__(self, cfg: Mapping[str, Any], dataset: str):
        self.root = Path(cfg['paths']['data_root']) / dataset
        index = read_json(self.root / 'scene_index.json')
        if index.get('status') != 'PASS':
            raise RuntimeError(f'scene index not PASS for {dataset}')
        self.metadata = {entry['graph_id']: dict(entry.get('metadata') or {}) for entry in index['graphs']}
        self.entries = {entry['graph_id']: entry['path'] for entry in index['graphs']}
        self.graphs: dict[str, SceneGraph] = {}

    def get(self, graph_id: str) -> SceneGraph:
        graph = self.graphs.get(graph_id)
        if graph is None:
            graph = load_graph(self.entries[graph_id], self.metadata.get(graph_id), graph_id)
            self.graphs[graph_id] = graph
        return graph

    def preload(self) -> int:
        for graph_id in self.entries:
            self.get(graph_id)
        return len(self.graphs)

def ncr_fill_value(cache: SceneCache, graph_ids: Sequence[str]) -> float:
    values = []
    for graph_id in graph_ids:
        graph = cache.get(graph_id)
        column = graph.node13[graph.include, NCR_CHANNEL]
        finite = column[np.isfinite(column)]
        if len(finite):
            values.append(finite)
    return float(np.median(np.concatenate(values))) if values else 0.0

@dataclass
class Batch:
    node: Any
    edge_index: Any
    edge: Any
    node_graph: Any
    node_tile: Any
    edge_graph: Any
    tile_ptr: Any
    tile_count: Any
    valid_nodes: Any
    bag_ptr: Any
    bag_count: Any
    tile_to_bag: Any
    tile_total: int
    tile_rgb: Any
    baseline: Any
    labels: Any
    samples: list[Sample]
    node_include: Any = None

def build_batch(cache: SceneCache, samples: Sequence[Sample], *, device: str, arm: str, ncr_fill: float, z_offset: Mapping[str, np.ndarray] | None=None) -> Batch:
    import torch
    node_chunks: list[np.ndarray] = []
    edge_index_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    node_graph_chunks: list[np.ndarray] = []
    node_tile_chunks: list[np.ndarray] = []
    node_include_chunks: list[np.ndarray] = []
    edge_graph_chunks: list[np.ndarray] = []
    tile_ptr = [0]
    tile_count: list[int] = []
    valid_nodes: list[int] = []
    bag_ptr = [0]
    bag_count: list[int] = []
    labels: list[int] = []
    baseline: list[np.ndarray] = []
    tile_lookup: dict[str, int] = {}
    node_offset = 0
    tile_index = 0
    for bag_position, sample in enumerate(samples):
        bag_start = tile_index
        for graph_id in sample.graph_ids:
            graph = cache.get(graph_id)
            node = graph.node13.astype(np.float32).copy()
            include = graph.include
            node[~include, :] = 0.0
            column = node[:, NCR_CHANNEL]
            node[:, NCR_CHANNEL] = np.where(np.isfinite(column), column, ncr_fill)
            if z_offset is None or graph_id not in z_offset:
                offset = graph.center_z
            else:
                offset = apply_z_permutation(graph.center_z, z_offset[graph_id])
            count = len(node)
            node_chunks.append(node)
            valid_nodes.append(int(include.sum()))
            tile_count.append(count)
            node_graph_chunks.append(np.full(count, bag_position, np.int64))
            node_tile_chunks.append(np.full(count, tile_index, np.int64))
            node_include_chunks.append(np.asarray(include, bool))
            tile_lookup[graph_id] = tile_index
            edges = graph.edge_index
            if len(edges):
                index = edges + node_offset
                delta_z = (offset[edges[0]] - offset[edges[1]]).astype(np.float32)
                base = graph.edge_base.astype(np.float32)
                if arm == 'XY':
                    base = np.stack((base[:, 0], np.zeros(len(base), np.float32), np.zeros(len(base), np.float32)), -1)
                    delta_z = np.zeros_like(delta_z)
                distance = np.sqrt((base[:, 0] * DISTANCE_SCALE_UM) ** 2 + delta_z.astype(np.float64) ** 2) / DISTANCE_SCALE_UM
                edge_chunks.append(np.stack((base[:, 0], delta_z / DISTANCE_SCALE_UM, distance.astype(np.float32), base[:, 1], base[:, 2]), -1))
                edge_index_chunks.append(index)
                edge_graph_chunks.append(np.full(index.shape[1], bag_position, np.int64))
            node_offset += count
            tile_index += 1
        tile_ptr.append(tile_index)
        bag_ptr.append(tile_index)
        bag_count.append(tile_index - bag_start)
        labels.append(int(sample.label_id))
        baseline.append(np.asarray(sample.baseline, np.float32))

    def stack(chunks, shape, dtype):
        if chunks:
            return torch.as_tensor(np.concatenate(chunks, 0), dtype=dtype, device=device)
        return torch.zeros(shape, dtype=dtype, device=device)
    if edge_index_chunks:
        edge_index = torch.as_tensor(np.concatenate(edge_index_chunks, 1), dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    tile_to_bag = np.zeros(max(tile_index, 1), np.int64)
    for index in range(len(bag_ptr) - 1):
        tile_to_bag[bag_ptr[index]:bag_ptr[index + 1]] = index
    tile_rgb = np.zeros((max(tile_index, 1), 2), np.float32)
    for sample in samples:
        entries = sample.tile_rgb
        if entries is None:
            continue
        for graph_id, entry in zip(sample.graph_ids, entries):
            tile_rgb[tile_lookup[graph_id]] = entry
    return Batch(node=stack(node_chunks, (0, NODE_DIM), torch.float32), edge_index=edge_index, edge=stack(edge_chunks, (0, EDGE_DIM), torch.float32), node_graph=stack(node_graph_chunks, (0,), torch.long), node_tile=stack(node_tile_chunks, (0,), torch.long), node_include=stack(node_include_chunks, (0,), torch.bool), edge_graph=stack(edge_graph_chunks, (0,), torch.long), tile_ptr=torch.as_tensor(np.asarray(tile_ptr, np.int64), device=device), tile_count=torch.as_tensor(np.asarray(tile_count, np.int64), device=device), valid_nodes=torch.as_tensor(np.asarray(valid_nodes, np.int64), device=device), bag_ptr=torch.as_tensor(np.asarray(bag_ptr, np.int64), device=device), bag_count=torch.as_tensor(np.asarray(bag_count, np.int64), device=device), tile_to_bag=torch.as_tensor(tile_to_bag, dtype=torch.long, device=device), tile_total=int(tile_index), tile_rgb=torch.as_tensor(tile_rgb, dtype=torch.float32, device=device), baseline=torch.as_tensor(np.stack(baseline, 0), dtype=torch.float32, device=device), labels=torch.as_tensor(np.asarray(labels, np.int64), device=device), samples=list(samples))

def bag_batches(samples: Sequence[Sample], max_bags: int, max_tiles: int) -> list[list[Sample]]:
    ordered = sorted(samples, key=lambda row: (len(row.graph_ids), row.bag_id))
    output: list[list[Sample]] = []
    current: list[Sample] = []
    tiles = 0
    for sample in ordered:
        size = len(sample.graph_ids)
        if current and (len(current) >= max_bags or tiles + size > max_tiles):
            output.append(current)
            current = []
            tiles = 0
        current.append(sample)
        tiles += size
    if current:
        output.append(current)
    return output
