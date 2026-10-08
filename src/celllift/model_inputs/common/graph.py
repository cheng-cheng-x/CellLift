from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import gzip
import io
from celllift.runtime import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .utils import atomic_json, atomic_parquet, read_parquet_rows, sha256_file, stable_shard

@dataclass
class GraphRecord:
    graph_id: str
    layer_idx: int
    split: str
    component_id: str
    track_id: str
    section_id: int
    nucleus_id: np.ndarray
    xy_px: np.ndarray
    xy_um: np.ndarray
    rho_um: np.ndarray
    border_flag: np.ndarray
    anchor_cell_id: np.ndarray
    biological_entity_id: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_feat: np.ndarray
    undirected_edge_src: np.ndarray
    undirected_edge_dst: np.ndarray

    def to_bytes(self) -> bytes:
        stream = io.BytesIO()
        np.savez(stream, metadata=np.asarray([self.graph_id, self.split, self.component_id, self.track_id], dtype='U128'), scalar=np.asarray([self.layer_idx, self.section_id], dtype=np.int64), nucleus_id=self.nucleus_id, xy_px=self.xy_px, xy_um=self.xy_um, rho_um=self.rho_um, border_flag=self.border_flag, anchor_cell_id=self.anchor_cell_id, biological_entity_id=self.biological_entity_id, edge_src=self.edge_src, edge_dst=self.edge_dst, edge_feat=self.edge_feat, undirected_edge_src=self.undirected_edge_src, undirected_edge_dst=self.undirected_edge_dst)
        return stream.getvalue()

    @staticmethod
    def from_bytes(payload: bytes) -> 'GraphRecord':
        data = np.load(io.BytesIO(payload), allow_pickle=False)
        meta = data['metadata']
        scalar = data['scalar']
        return GraphRecord(str(meta[0]), int(scalar[0]), str(meta[1]), str(meta[2]), str(meta[3]), int(scalar[1]), np.asarray(data['nucleus_id']), np.asarray(data['xy_px']), np.asarray(data['xy_um']), np.asarray(data['rho_um']), np.asarray(data['border_flag']), np.asarray(data['anchor_cell_id']), np.asarray(data['biological_entity_id']), np.asarray(data['edge_src']), np.asarray(data['edge_dst']), np.asarray(data['edge_feat']), np.asarray(data['undirected_edge_src']), np.asarray(data['undirected_edge_dst']))

class ShardedWriter:

    def __init__(self, root: Path, shards: int, map_size: int):
        import lmdb
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.shards = (root, int(shards))
        self.envs = [lmdb.open(str(root / f'graph_cache_{index:02d}.lmdb'), subdir=True, map_size=map_size, max_dbs=1, lock=True, sync=False, metasync=False) for index in range(self.shards)]

    def put(self, key: str, value: bytes) -> int:
        import lmdb
        shard = stable_shard(key, self.shards)
        env = self.envs[shard]
        while True:
            try:
                with env.begin(write=True) as txn:
                    existing = txn.get(key.encode())
                    if existing is None:
                        txn.put(key.encode(), value, overwrite=False)
                    elif bytes(existing) != value:
                        raise RuntimeError(f'LMDB key exists with different payload: {key}')
                return shard
            except lmdb.MapFullError:
                env.set_mapsize(env.info()['map_size'] * 2)

    def close(self) -> None:
        for env in self.envs:
            env.sync()
            env.close()

def radial_rays_from_mask(mask: np.ndarray, instance_id: int, centroid_xy: tuple[float, float], count: int, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox_xyxy)
    crop = mask[y0:y1, x0:x1]
    y, x = np.nonzero(crop == instance_id)
    if not len(x):
        raise RuntimeError(f'instance {instance_id} absent from nucleus mask')
    cx, cy = (float(centroid_xy[0]) - x0, float(centroid_xy[1]) - y0)
    rx, ry = (int(round(cx)), int(round(cy)))
    if not (0 <= ry < crop.shape[0] and 0 <= rx < crop.shape[1] and (crop[ry, rx] == instance_id)):
        nearest = int(np.argmin(np.square(x - cx) + np.square(y - cy)))
        cx, cy = (float(x[nearest]), float(y[nearest]))
    maximum = math.hypot(max(cx, crop.shape[1] - 1 - cx), max(cy, crop.shape[0] - 1 - cy)) + 2
    distance = np.arange(0.0, maximum + 0.25, 0.25, dtype=np.float32)
    angles = np.arange(count, dtype=np.float32) * (2 * np.pi / count)
    sample_x = np.rint(cx + np.cos(angles)[:, None] * distance).astype(np.int32)
    sample_y = np.rint(cy + np.sin(angles)[:, None] * distance).astype(np.int32)
    valid = (sample_x >= 0) & (sample_x < crop.shape[1]) & (sample_y >= 0) & (sample_y < crop.shape[0])
    inside = np.zeros_like(valid)
    inside[valid] = crop[sample_y[valid], sample_x[valid]] == instance_id
    rays = np.cumprod(inside.astype(np.uint8), axis=1).sum(axis=1).astype(np.float32) * 0.25
    if np.any(rays <= 0) or not np.all(np.isfinite(rays)):
        raise RuntimeError(f'invalid radial rays for instance {instance_id}')
    return rays

def build_knn_physical(xy_um: np.ndarray, k: int=12, radius_um: float=60.0, normalizer_um: float=471.04) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(xy_um)
    empty = np.empty(0, np.int64)
    if count <= 1:
        return (empty, empty, np.empty((0, 3), np.float32), empty, empty)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(xy_um)
        distances, indices = tree.query(xy_um, k=min(k + 1, count), distance_upper_bound=radius_um)
    except ModuleNotFoundError:
        pairwise = np.linalg.norm(xy_um[:, None, :] - xy_um[None, :, :], axis=2)
        order = np.argsort(pairwise, axis=1)[:, :min(k + 1, count)]
        distances = np.take_along_axis(pairwise, order, axis=1)
        indices = order
        indices = np.where(distances <= radius_um, indices, count)
        distances = np.where(distances <= radius_um, distances, np.inf)
    source, target = ([], [])
    for index in range(count):
        candidates = [(float(distance), int(neighbor)) for distance, neighbor in zip(np.atleast_1d(distances[index]), np.atleast_1d(indices[index])) if neighbor < count and neighbor != index and (distance <= radius_um)]
        for _, neighbor in sorted(candidates)[:k]:
            source.append(index)
            target.append(neighbor)
    src, dst = (np.asarray(source, np.int64), np.asarray(target, np.int64))
    if len(src):
        delta = (xy_um[dst] - xy_um[src]) / float(normalizer_um)
        features = np.column_stack([delta, np.linalg.norm(delta, axis=1)]).astype(np.float32)
    else:
        features = np.empty((0, 3), np.float32)
    undirected = sorted({(min(a, b), max(a, b)) for a, b in zip(source, target)})
    us = np.asarray([item[0] for item in undirected], np.int64)
    ud = np.asarray([item[1] for item in undirected], np.int64)
    return (src, dst, features, us, ud)

def _read_gzip(path: str | Path) -> dict[str, Any]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def build_record(row: dict[str, Any], layer_idx: int, *, nucleus_only: bool=False) -> GraphRecord:
    mask = np.load(row['nucleus_mask_path'], mmap_mode='r')
    nuclei = sorted(_read_gzip(row['nucleus_instances_path'])['instances'], key=lambda item: int(item['instance_id']))
    if not nuclei:
        raise RuntimeError('zero_nuclei')
    accepted: dict[int, int] = {}
    if not nucleus_only:
        pairs = _read_gzip(row['pairs_path'])
        accepted = {int(item['nucleus_id']): int(item['cell_id']) for item in pairs['nucleus_relations'] if item.get('pair_status') == 'accepted_one_to_one' and item.get('cell_id')}
    mpp = float(row['actual_target_mpp'])
    nucleus_id = np.asarray([int(item['instance_id']) for item in nuclei], np.int64)
    xy_px = np.asarray([item['centroid_xy'] for item in nuclei], np.float32)
    xy_um = xy_px * mpp
    rho_um = np.stack([radial_rays_from_mask(mask, int(item['instance_id']), tuple(item['centroid_xy']), 36, tuple(item['bbox_xyxy'])) for item in nuclei]).astype(np.float32) * mpp
    border = np.asarray([bool(item['border_flag']) for item in nuclei])
    cells = np.asarray([accepted.get(int(value), 0) for value in nucleus_id], np.int64)
    edge_src, edge_dst, edge_feat, undirected_src, undirected_dst = build_knn_physical(xy_um)
    return GraphRecord(str(row['graph_id']), layer_idx, str(row['official_split']), str(row['patient_id']), '', 0, nucleus_id, xy_px, xy_um, rho_um, border, cells, np.zeros(len(nucleus_id), np.uint64), edge_src, edge_dst, edge_feat, undirected_src, undirected_dst)

def _build_record_job(args: tuple[dict[str, Any], int, bool]) -> tuple[GraphRecord | None, str | None]:
    row, layer_idx, nucleus_only = args
    try:
        return (build_record(row, layer_idx, nucleus_only=nucleus_only), None)
    except Exception as exc:
        return (None, f'{type(exc).__name__}: {exc}')

def run_graph(cfg: dict[str, Any], dataset: str, shard_id: int, num_shards: int, *, pilot: bool=False) -> dict[str, Any]:
    data_root = Path(cfg['paths']['data_root'])
    all_rows = read_parquet_rows(data_root / '00_manifest/patch_manifest.parquet')
    layer_lookup = {str(row['patch_id']): index for index, row in enumerate(all_rows)}
    if pilot:
        selected = {str(row['patch_id']) for row in read_parquet_rows(data_root / '00_manifest/pilot_manifest.parquet')}
        all_rows = [row for row in all_rows if str(row['patch_id']) in selected]
    rows = [row for row in all_rows if stable_shard(str(row['patch_id']), num_shards) == shard_id]
    nucleus_only = cfg.get('input', {}).get('mode') == 'nucleus_only'
    graph_workers = max(1, int(cfg.get('runtime', {}).get('graph_workers', 1)))
    pilot_cache = '05_qc/nucleus_only_pilot_graph_cache' if nucleus_only else '05_qc/pilot_graph_cache'
    graph_root = data_root / (pilot_cache if pilot else '03_graph_cache')
    writer = ShardedWriter(graph_root, 64, int(cfg['graph'].get('lmdb_map_size_bytes', 4 << 30)))
    index_rows, excluded = ([], [])
    try:
        jobs = ((row, layer_lookup[str(row['patch_id'])], nucleus_only) for row in rows)
        executor = ProcessPoolExecutor(max_workers=graph_workers) if graph_workers > 1 else None
        results = executor.map(_build_record_job, jobs, chunksize=4) if executor else map(_build_record_job, jobs)
        for row, (record, error) in zip(rows, results):
            try:
                if error or record is None:
                    raise RuntimeError(error or 'unknown graph build failure')
                lmdb_shard = writer.put(record.graph_id, record.to_bytes())
                index_rows.append({'graph_id': record.graph_id, 'patch_id': row['patch_id'], 'dataset_id': row['dataset_id'], 'layer_idx': record.layer_idx, 'split': record.split, 'patient_id': row['patient_id'], 'slide_id': row['slide_id'], 'label_id': row['label_id'], 'label_name': row['label_name'], 'label_scope': row['label_scope'], 'validation_fold': row['validation_fold'], 'shard': lmdb_shard, 'node_count': len(record.nucleus_id), 'edge_count': len(record.edge_src), 'undirected_edge_count': len(record.undirected_edge_src), 'center_pair_count': int(np.count_nonzero(record.anchor_cell_id)), 'output_width_px': row['output_width_px'], 'output_height_px': row['output_height_px'], 'actual_target_mpp': row['actual_target_mpp']})
            except Exception as exc:
                excluded.append({'patch_id': row['patch_id'], 'graph_id': row['graph_id'], 'label_name': row['label_name'], 'official_split': row['official_split'], 'exclusion_reason': f'{type(exc).__name__}: {exc}'})
        if executor:
            executor.shutdown(wait=True)
    finally:
        writer.close()
    tag = 'pilot' if pilot else 'full'
    index_path = graph_root / f'graph_index_{tag}_workshard_{shard_id:03d}_of_{num_shards:03d}.parquet'
    excluded_path = graph_root / f'excluded_{tag}_workshard_{shard_id:03d}_of_{num_shards:03d}.parquet'
    atomic_parquet(index_path, index_rows)
    atomic_parquet(excluded_path, excluded)
    stats_source = Path(cfg['graph']['training_feature_stats'])
    if not stats_source.is_file():
        raise FileNotFoundError(stats_source)
    digest = sha256_file(stats_source)
    stats = json.loads(stats_source.read_text(encoding='utf-8'))
    if len(stats.get('mean', [])) != 36 or len(stats.get('std', [])) != 36 or min(stats['std']) <= 0:
        raise RuntimeError('invalid frozen training feature statistics')
    atomic_json(graph_root / 'inference_feature_stats.json', {**stats, 'source_path': str(stats_source), 'source_sha256': digest, 'policy': 'frozen_training_only'})
    return {'status': 'PASS' if not excluded else 'PARTIAL', 'graphs': len(index_rows), 'excluded': len(excluded), 'index': str(index_path)}
