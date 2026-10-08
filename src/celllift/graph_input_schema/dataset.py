from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import math
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .cache_io import ShardedLMDBReader
from .schemas import GraphRecord, ObjectType, ObservationRecord, TargetRecord

@dataclass
class GraphSample:
    graph: GraphRecord
    positive: ObservationRecord
    empty: ObservationRecord
    targets: dict[str, TargetRecord]
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None

@dataclass
class GraphInferenceSample:
    graph: GraphRecord
    nucleus_observation_count: np.ndarray
    cell_observation_count: np.ndarray
    gold_observation_count: np.ndarray
    silver_observation_count: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

class GraphObservationDataset(Dataset):

    def __init__(self, graph_index_path: str | Path, graph_cache_dir: str | Path, observation_cache_dir: str | Path, target_cache_dir: str | Path | None=None, positive_cache_dir: str | Path | None=None, *, split: str | None=None, graph_shards: int=32, observation_shards: int=32, feature_stats_path: str | Path | None=None):
        import pyarrow.parquet as pq
        table = pq.read_table(graph_index_path, partitioning=None)
        rows = table.to_pylist()
        self.rows = [row for row in rows if split is None or row['split'] == split]
        self.graphs = ShardedLMDBReader(graph_cache_dir, 'graph_cache', graph_shards)
        self.positive = ShardedLMDBReader(positive_cache_dir or observation_cache_dir, 'positive', observation_shards)
        self.empty = ShardedLMDBReader(observation_cache_dir, 'empty', observation_shards)
        self.targets = ShardedLMDBReader(target_cache_dir or observation_cache_dir, 'target_cache', observation_shards)
        stats_path = Path(feature_stats_path or Path(graph_cache_dir) / 'feature_stats.json')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        self.mean = np.asarray(stats['mean'], np.float32)
        self.std = np.asarray(stats['std'], np.float32)
        if self.mean.shape != (36,) or self.std.shape != (36,) or np.any(self.std <= 0):
            raise RuntimeError('invalid frozen train feature statistics')

    def __len__(self) -> int:
        return len(self.rows)

    def sample_cost(self, index: int) -> dict[str, int]:
        row = self.rows[index]
        return {'nodes': int(row['node_count']), 'edges': int(row['edge_count']), 'projection_voxels': int(row.get('projection_voxels') or int(row.get('positive_count', 0)) * 8192)}

    @property
    def costs(self) -> list[dict[str, int]]:
        return [self.sample_cost(i) for i in range(len(self))]

    def __getitem__(self, index: int) -> GraphSample:
        graph_id = str(self.rows[index]['graph_id'])
        graph = GraphRecord.from_bytes(self.graphs.get(graph_id))
        positive = ObservationRecord.from_bytes(self.positive.get(graph_id))
        empty = ObservationRecord.from_bytes(self.empty.get(graph_id))
        targets = {str(uid): TargetRecord.from_bytes(self.targets.get(str(uid))) for uid in np.unique(positive.target_uid)}
        return GraphSample(graph, positive, empty, targets, self.mean, self.std)

    def close_cache_handles(self) -> None:
        for reader in (self.graphs, self.positive, self.empty, self.targets):
            reader.close()

class GraphInferenceDataset(Dataset):

    def __init__(self, graph_index_path: str | Path, graph_cache_dir: str | Path, observation_cache_dir: str | Path, positive_cache_dir: str | Path | None=None, *, graph_shards: int=32, observation_shards: int=32, feature_stats_path: str | Path | None=None):
        import pyarrow.parquet as pq
        self.rows = pq.read_table(graph_index_path, partitioning=None).to_pylist()
        self.graphs = ShardedLMDBReader(graph_cache_dir, 'graph_cache', graph_shards)
        self.positive = ShardedLMDBReader(positive_cache_dir or observation_cache_dir, 'positive', observation_shards)
        self.empty = ShardedLMDBReader(observation_cache_dir, 'empty', observation_shards)
        stats_path = Path(feature_stats_path or Path(graph_cache_dir) / 'feature_stats.json')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        self.mean = np.asarray(stats['mean'], np.float32)
        self.std = np.asarray(stats['std'], np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def sample_cost(self, index: int) -> dict[str, int]:
        row = self.rows[index]
        return {'nodes': int(row['node_count']), 'edges': int(row['edge_count']), 'projection_voxels': 0}

    def __getitem__(self, index: int) -> GraphInferenceSample:
        graph_id = str(self.rows[index]['graph_id'])
        graph = GraphRecord.from_bytes(self.graphs.get(graph_id))
        counts = {ObjectType.NUCLEUS: np.zeros(len(graph.nucleus_id), np.int64), ObjectType.CELL: np.zeros(len(graph.nucleus_id), np.int64)}
        confidence_counts = {1: np.zeros(len(graph.nucleus_id), np.int64), 2: np.zeros(len(graph.nucleus_id), np.int64)}
        for reader in (self.positive, self.empty):
            observations = ObservationRecord.from_bytes(reader.get(graph_id))
            for object_type in (ObjectType.NUCLEUS, ObjectType.CELL):
                selected = observations.object_type == object_type
                np.add.at(counts[object_type], observations.anchor_node_idx[selected], 1)
            for confidence_code in (1, 2):
                selected = observations.confidence_code == confidence_code
                np.add.at(confidence_counts[confidence_code], observations.anchor_node_idx[selected], 1)
        return GraphInferenceSample(graph, counts[ObjectType.NUCLEUS], counts[ObjectType.CELL], confidence_counts[1], confidence_counts[2], self.mean, self.std)

def _concat(values: list[np.ndarray], *, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    return np.concatenate(values, axis=0) if values else np.empty(shape, dtype=dtype)

def _torch(array: np.ndarray, dtype: torch.dtype | None=None) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(array))
    return value.to(dtype=dtype) if dtype is not None else value

def _pixel_stream(observations: list[tuple[ObservationRecord, int, dict[str, TargetRecord]]], obj: int, mpp: float) -> dict[str, torch.Tensor]:
    anchor, plane, weight, confidence, cycle, triplet, target_uids = ([], [], [], [], [], [], [])
    targets, pixel_xy, pixel_obs, target_bbox, target_shape, roi_bounds = ([], [], [], [], [], [])
    obs_index = 0
    for record, offset, target_map in observations:
        selected = np.flatnonzero(record.object_type == obj)
        for idx in selected:
            target = target_map[str(record.target_uid[idx])]
            mask = target.decode_mask()
            x0, y0, x1, y1 = map(int, target.bbox_xyxy)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            targets.append(mask.reshape(-1).astype(np.float32))
            pixel_xy.append((np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32) + 0.5) * mpp)
            pixel_obs.append(np.full(mask.size, obs_index, np.int64))
            target_bbox.append(target.bbox_xyxy.astype(np.float32))
            target_shape.append(target.crop_shape.astype(np.int64))
            roi_bounds.append(target.support_xyxy.astype(np.float32))
            anchor.append(int(record.anchor_node_idx[idx]) + offset)
            plane.append(int(record.plane_idx[idx]))
            weight.append(float(record.weight[idx]))
            confidence.append(int(record.confidence_code[idx]))
            cycle.append(int(record.cycle_code[idx]))
            obs_index += 1
            triplet.append(int(record.triplet_group_id[idx]))
            target_uids.append(str(record.target_uid[idx]))
    return {'anchor_node_index': _torch(np.asarray(anchor, np.int64)), 'plane_index': _torch(np.asarray(plane, np.int64)), 'weight': _torch(np.asarray(weight, np.float32)), 'confidence_code': _torch(np.asarray(confidence, np.int64)), 'cycle_code': _torch(np.asarray(cycle, np.int64)), 'triplet_group_id': _torch(np.asarray(triplet, np.int64)), 'target_uid': target_uids, 'target': _torch(_concat(targets, shape=(0,), dtype=np.float32)), 'pixel_xy_um': _torch(_concat(pixel_xy, shape=(0, 2), dtype=np.float32)), 'pixel_observation_index': _torch(_concat(pixel_obs, shape=(0,), dtype=np.int64)), 'target_bbox_px': _torch(np.asarray(target_bbox, np.float32).reshape(-1, 4)), 'target_shape': _torch(np.asarray(target_shape, np.int64).reshape(-1, 2)), 'roi_bounds_px': _torch(np.asarray(roi_bounds, np.float32).reshape(-1, 4))}

def _observed_cell_radius_um(samples: Sequence[GraphSample], ptr: Sequence[int], mpp: float) -> np.ndarray:
    node_count = int(ptr[-1]) if ptr else 0
    radii = np.zeros(node_count, dtype=np.float32)
    for sample, offset in zip(samples, ptr[:-1]):
        graph = sample.graph
        by_cell_id: dict[int, TargetRecord] = {}
        for target in sample.targets.values():
            if int(target.object_type) == int(ObjectType.CELL):
                by_cell_id[int(target.instance_id)] = target
        for node_idx, cell_id in enumerate(graph.anchor_cell_id):
            cid = int(cell_id)
            if cid <= 0:
                continue
            target = by_cell_id.get(cid)
            if target is None or int(target.area_px) <= 0:
                continue
            radii[int(offset) + node_idx] = math.sqrt(float(target.area_px) / math.pi) * float(mpp)
    return radii

def _empty_stream(observations: list[tuple[ObservationRecord, int]], obj: int) -> dict[str, torch.Tensor]:
    fields: dict[str, list[np.ndarray]] = {key: [] for key in ('anchor', 'plane', 'weight', 'confidence', 'cycle', 'triplet', 'evidence')}
    for record, offset in observations:
        selected = np.flatnonzero(record.object_type == obj)
        fields['anchor'].append(record.anchor_node_idx[selected] + offset)
        fields['plane'].append(record.plane_idx[selected])
        fields['weight'].append(record.weight[selected])
        fields['confidence'].append(record.confidence_code[selected])
        fields['cycle'].append(record.cycle_code[selected])
        fields['evidence'].append(record.evidence_code[selected])
        fields['triplet'].append(record.triplet_group_id[selected])
    return {'anchor_node_index': _torch(_concat(fields['anchor'], shape=(0,), dtype=np.int64), torch.int64), 'plane_index': _torch(_concat(fields['plane'], shape=(0,), dtype=np.int8), torch.int64), 'weight': _torch(_concat(fields['weight'], shape=(0,), dtype=np.float32)), 'confidence_code': _torch(_concat(fields['confidence'], shape=(0,), dtype=np.uint8), torch.int64), 'cycle_code': _torch(_concat(fields['cycle'], shape=(0,), dtype=np.uint8), torch.int64), 'triplet_group_id': _torch(_concat(fields['triplet'], shape=(0,), dtype=np.int64), torch.int64), 'evidence_code': _torch(_concat(fields['evidence'], shape=(0,), dtype=np.uint8), torch.int64)}

def collate_graph_samples(samples: Sequence[GraphSample], *, mpp: float=0.46, feature_mean: np.ndarray | None=None, feature_std: np.ndarray | None=None) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot collate an empty batch')
    inherited_mean = samples[0].feature_mean
    inherited_std = samples[0].feature_std
    mean = np.asarray(feature_mean if feature_mean is not None else inherited_mean if inherited_mean is not None else np.zeros(36), np.float32)
    std = np.asarray(feature_std if feature_std is not None else inherited_std if inherited_std is not None else np.ones(36), np.float32)
    ptr = [0]
    edge_index, undirected, metadata = ([], [], [])
    positive_with_offset, empty_with_offset = ([], [])
    for sample in samples:
        graph = sample.graph
        offset = ptr[-1]
        ptr.append(offset + len(graph.nucleus_id))
        edge_index.append(np.stack([graph.edge_src + offset, graph.edge_dst + offset]))
        undirected.append(np.stack([graph.undirected_edge_src + offset, graph.undirected_edge_dst + offset]))
        metadata.append({'graph_id': graph.graph_id, 'layer_idx': graph.layer_idx, 'split': graph.split, 'component_id': graph.component_id, 'track_id': graph.track_id, 'section_id': graph.section_id})
        positive_with_offset.append((sample.positive, offset, sample.targets))
        empty_with_offset.append((sample.empty, offset))
    batch = {'graph_ids': [item['graph_id'] for item in metadata], 'metadata': metadata, 'num_graphs': len(samples), 'split': [item['split'] for item in metadata], 'track_id': [item['track_id'] for item in metadata], 'component_id': [item['component_id'] for item in metadata], 'layer_idx': [item['layer_idx'] for item in metadata], 'section_id': [item['section_id'] for item in metadata], 'graph_ptr': _torch(np.asarray(ptr, np.int64)), 'node_features': _torch(np.concatenate([(s.graph.rho_um - mean) / std for s in samples]).astype(np.float32)), 'rho_um': _torch(np.concatenate([s.graph.rho_um for s in samples])), 'nucleus_rays_um': _torch(np.concatenate([s.graph.rho_um for s in samples]).astype(np.float32)), 'xy_px': _torch(np.concatenate([s.graph.xy_px for s in samples])), 'xy_um': _torch(np.concatenate([s.graph.xy_um for s in samples])), 'nucleus_id': _torch(np.concatenate([s.graph.nucleus_id for s in samples]), torch.int64), 'anchor_cell_id': _torch(np.concatenate([s.graph.anchor_cell_id for s in samples]), torch.int64), 'observed_cell_radius_um': _torch(_observed_cell_radius_um(samples, ptr, mpp)), 'biological_entity_id': _torch(np.concatenate([s.graph.biological_entity_id for s in samples]).view(np.int64), torch.int64), 'border_flag': _torch(np.concatenate([s.graph.border_flag for s in samples])), 'edge_index': _torch(np.concatenate(edge_index, axis=1), torch.int64), 'edge_feat': _torch(np.concatenate([s.graph.edge_feat for s in samples])), 'undirected_edge_index': _torch(np.concatenate(undirected, axis=1), torch.int64)}
    left, right = batch['undirected_edge_index']
    same_cell = (batch['anchor_cell_id'][left] > 0) & (batch['anchor_cell_id'][left] == batch['anchor_cell_id'][right])
    same_entity = (batch['biological_entity_id'][left] > 0) & (batch['biological_entity_id'][left] == batch['biological_entity_id'][right])
    batch['non_overlap_edge_mask'] = ~(same_cell | same_entity)
    batch['nucleus_positive'] = _pixel_stream(positive_with_offset, ObjectType.NUCLEUS, mpp)
    batch['cell_positive'] = _pixel_stream(positive_with_offset, ObjectType.CELL, mpp)
    batch['nucleus_empty'] = _empty_stream(empty_with_offset, ObjectType.NUCLEUS)
    batch['cell_empty'] = _empty_stream(empty_with_offset, ObjectType.CELL)
    node_count = int(ptr[-1])
    for name in ('nucleus', 'cell'):
        counts = torch.zeros(node_count, dtype=torch.int64)
        for stream_name in (f'{name}_positive', f'{name}_empty'):
            anchors = batch[stream_name]['anchor_node_index']
            if anchors.numel():
                counts.index_add_(0, anchors, torch.ones_like(anchors, dtype=torch.int64))
        batch[f'{name}_observation_count'] = counts
    return batch

def make_collate_fn(dataset: GraphObservationDataset, mpp: float=0.46):

    def collate(samples: Sequence[GraphSample]):
        return collate_graph_samples(samples, mpp=mpp, feature_mean=dataset.mean, feature_std=dataset.std)
    return collate

def collate_inference_samples(samples: Sequence[GraphInferenceSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot collate an empty inference batch')
    ptr = [0]
    edges: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for sample in samples:
        graph = sample.graph
        offset = ptr[-1]
        ptr.append(offset + len(graph.nucleus_id))
        edges.append(np.stack([graph.edge_src + offset, graph.edge_dst + offset]))
        metadata.append({'graph_id': graph.graph_id, 'layer_idx': graph.layer_idx, 'split': graph.split, 'component_id': graph.component_id, 'track_id': graph.track_id, 'section_id': graph.section_id})
    mean, std = (samples[0].feature_mean, samples[0].feature_std)
    return {'graph_ids': [item['graph_id'] for item in metadata], 'metadata': metadata, 'num_graphs': len(samples), 'split': [item['split'] for item in metadata], 'track_id': [item['track_id'] for item in metadata], 'component_id': [item['component_id'] for item in metadata], 'layer_idx': [item['layer_idx'] for item in metadata], 'section_id': [item['section_id'] for item in metadata], 'graph_ptr': _torch(np.asarray(ptr, np.int64)), 'node_features': _torch(np.concatenate([(s.graph.rho_um - mean) / std for s in samples]).astype(np.float32)), 'node_features_raw_um': _torch(np.concatenate([s.graph.rho_um for s in samples]).astype(np.float32)), 'nucleus_rays_um': _torch(np.concatenate([s.graph.rho_um for s in samples]).astype(np.float32)), 'xy_um': _torch(np.concatenate([s.graph.xy_um for s in samples])), 'nucleus_id': _torch(np.concatenate([s.graph.nucleus_id for s in samples]), torch.int64), 'edge_index': _torch(np.concatenate(edges, axis=1), torch.int64), 'edge_feat': _torch(np.concatenate([s.graph.edge_feat for s in samples])), 'nucleus_observation_count': _torch(np.concatenate([s.nucleus_observation_count for s in samples]), torch.int64), 'cell_observation_count': _torch(np.concatenate([s.cell_observation_count for s in samples]), torch.int64), 'gold_observation_count': _torch(np.concatenate([s.gold_observation_count for s in samples]), torch.int64), 'silver_observation_count': _torch(np.concatenate([s.silver_observation_count for s in samples]), torch.int64)}

def make_inference_collate_fn(dataset: GraphInferenceDataset):

    def collate(samples: Sequence[GraphInferenceSample]):
        return collate_inference_samples(samples)
    return collate
