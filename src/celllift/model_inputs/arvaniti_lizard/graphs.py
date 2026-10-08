from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import gzip
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
_MODEL_INPUT = Path(__file__).resolve().parents[1]
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
from celllift.model_inputs.common.graph import GraphRecord, ShardedWriter, build_knn_physical, radial_rays_from_mask
from celllift.model_inputs.common.segment import instance_payload
from celllift.model_inputs.common.utils import atomic_json, sha256_file, stable_shard
from .constants import EDGE_NORMALIZER_UM, FEATURE_STATS, K_NEIGHBORS, RADIUS_UM, RAY_COUNT, TARGET_MPP
from .io_utils import write_parquet

def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.tmp.{os.getpid()}.npy')
    with temporary.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)

def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as handle:
        json.dump(value, handle, separators=(',', ':'), sort_keys=True)
    os.replace(temporary, path)

def write_nucleus_artifacts(mask: np.ndarray, mask_path: Path, instances_path: Path) -> dict[str, Any]:
    payload = instance_payload(mask)
    _atomic_npy(mask_path, np.asarray(mask))
    _atomic_gzip_json(instances_path, payload)
    return payload

def crop_instances(core_mask: np.ndarray, row0: int, col0: int, size: int) -> tuple[np.ndarray, dict[str, Any]]:
    crop = np.asarray(core_mask[row0:row0 + size, col0:col0 + size]).copy()
    payload = instance_payload(crop)
    for item in payload['instances']:
        item['source_row0'] = int(row0)
        item['source_col0'] = int(col0)
    return (crop, payload)

def build_record_from_payload(graph_id: str, split: str, patient_id: str, mask: np.ndarray, instances: list[dict[str, Any]], *, mpp: float=TARGET_MPP, layer_idx: int=0) -> GraphRecord:
    if not instances:
        raise RuntimeError('zero_nuclei')
    nuclei = sorted(instances, key=lambda item: int(item['instance_id']))
    nucleus_id = np.asarray([int(item['instance_id']) for item in nuclei], np.int64)
    xy_px = np.asarray([item['centroid_xy'] for item in nuclei], np.float32)
    xy_um = xy_px * float(mpp)
    rho_um = np.stack([radial_rays_from_mask(mask, int(item['instance_id']), tuple(item['centroid_xy']), RAY_COUNT, tuple(item['bbox_xyxy'])) for item in nuclei]).astype(np.float32) * float(mpp)
    border = np.asarray([bool(item['border_flag']) for item in nuclei])
    edge_src, edge_dst, edge_feat, undirected_src, undirected_dst = build_knn_physical(xy_um, k=K_NEIGHBORS, radius_um=RADIUS_UM, normalizer_um=EDGE_NORMALIZER_UM)
    return GraphRecord(graph_id, layer_idx, split, str(patient_id), '', 0, nucleus_id, xy_px, xy_um, rho_um, border, np.zeros(len(nucleus_id), np.int64), np.zeros(len(nucleus_id), np.uint64), edge_src, edge_dst, edge_feat, undirected_src, undirected_dst)
_READER_ENVS: dict[str, list] = {}

class GraphReader:

    def __init__(self, root: Path, shards: int=64, prefix: str='graph_cache'):
        import lmdb
        self.root = Path(root)
        self.shards = int(shards)
        cache_key = f'{self.root}:{prefix}:{shards}'
        cached = _READER_ENVS.get(cache_key)
        if cached is not None:
            self.envs = cached
            return
        self.envs = [lmdb.open(str(self.root / f'{prefix}_{index:02d}.lmdb'), subdir=True, readonly=True, lock=False, readahead=False, max_readers=2048) for index in range(self.shards)]
        _READER_ENVS[cache_key] = self.envs

    def get(self, key: str) -> bytes:
        shard = stable_shard(key, self.shards)
        with self.envs[shard].begin(buffers=False) as txn:
            payload = txn.get(key.encode())
        if payload is None:
            raise KeyError(key)
        return bytes(payload)

    def close(self) -> None:
        return

def write_feature_stats(graph_root: Path) -> dict[str, Any]:
    source = Path(FEATURE_STATS)
    stats = json.loads(source.read_text(encoding='utf-8'))
    payload = {**stats, 'source_path': str(source), 'source_sha256': sha256_file(source), 'policy': 'frozen_training_only'}
    atomic_json(graph_root / 'inference_feature_stats.json', payload)
    return payload

def _graph_job(row: dict[str, Any]) -> tuple[bytes | None, dict[str, Any], str | None]:
    try:
        mask = np.load(row['nucleus_mask_path'], mmap_mode='r')
        with gzip.open(row['nucleus_instances_path'], 'rt', encoding='utf-8') as handle:
            payload = json.load(handle)
        record = build_record_from_payload(row['graph_id'], row['official_split'], row['patient_id'], mask, payload['instances'])
        meta = {**{key: row.get(key) for key in ('graph_id', 'core_id', 'roi_id', 'patient_id', 'role', 'official_split', 'label_id', 'label_name', 'rgb_path', 'nucleus_mask_path', 'dino_rgb_path')}, 'node_count': len(record.nucleus_id), 'edge_count': len(record.edge_src), 'output_height_px': int(mask.shape[0]), 'output_width_px': int(mask.shape[1]), 'actual_target_mpp': TARGET_MPP}
        return (record.to_bytes(), meta, None)
    except Exception as exc:
        return (None, {'graph_id': row.get('graph_id'), 'role': row.get('role')}, f'{type(exc).__name__}: {exc}')

def write_graphs(rows: list[dict[str, Any]], graph_root: Path, *, map_size: int=4 << 30, workers: int=8) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    graph_root.mkdir(parents=True, exist_ok=True)
    writer = ShardedWriter(graph_root, 64, map_size)
    index_rows = []
    excluded = []
    try:
        if workers <= 1:
            results = map(_graph_job, rows)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            results = pool.map(_graph_job, rows, chunksize=4)
        for payload, meta, error in results:
            if error or payload is None:
                excluded.append({**meta, 'exclusion_reason': error or 'unknown'})
                continue
            shard = writer.put(str(meta['graph_id']), payload)
            meta['shard'] = shard
            index_rows.append(meta)
        if workers > 1:
            pool.shutdown(wait=True)
    finally:
        writer.close()
    write_parquet(graph_root / 'graph_index.parquet', index_rows)
    write_parquet(graph_root / 'excluded.parquet', excluded)
    write_feature_stats(graph_root)
    return {'graphs': len(index_rows), 'excluded': len(excluded), 'root': str(graph_root)}
