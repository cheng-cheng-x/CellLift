from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import io
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.model_inputs.common.graph import GraphRecord, build_knn_physical, radial_rays_from_mask
from ..io_utils import atomic_json, atomic_npz, sha256_file
from ..protocol import GRAPH_K, GRAPH_NORMALIZER_UM, GRAPH_RADIUS_UM, RAY_COUNT, TARGET_MPP, TRAINING_FEATURE_STATS
from . import layout
from .segment import load_slide_instances, load_slide_masks

def empty_record(graph_id: str, split: str, patient_id: str) -> GraphRecord:
    empty_i = np.empty(0, np.int64)
    empty_f2 = np.empty((0, 2), np.float32)
    empty_r = np.empty((0, RAY_COUNT), np.float32)
    empty_e = np.empty((0, 3), np.float32)
    return GraphRecord(graph_id, 0, split, patient_id, '', 0, empty_i, empty_f2, empty_f2, empty_r, np.empty(0, np.bool_), empty_i, np.empty(0, np.uint64), empty_i, empty_i, empty_e, empty_i, empty_i)

def record_from_mask(mask: np.ndarray, instances: list[dict[str, Any]], *, graph_id: str, split: str, patient_id: str, mpp: float=TARGET_MPP) -> GraphRecord:
    nuclei = sorted(instances, key=lambda item: int(item['instance_id']))
    if not nuclei:
        return empty_record(graph_id, split, patient_id)
    nucleus_id = np.asarray([int(item['instance_id']) for item in nuclei], np.int64)
    if mask.dtype != np.int32:
        raise RuntimeError(f'mask must be int32, got {mask.dtype}')
    mask_ids = {int(value) for value in np.unique(mask) if int(value) > 0}
    if set(map(int, nucleus_id.tolist())) != mask_ids:
        raise RuntimeError(f'nucleus_id mismatch for {graph_id}')
    xy_px = np.asarray([item['centroid_xy'] for item in nuclei], np.float32)
    xy_um = xy_px * float(mpp)
    rho_um = np.stack([radial_rays_from_mask(mask, int(item['instance_id']), tuple(item['centroid_xy']), RAY_COUNT, tuple(item['bbox_xyxy'])) for item in nuclei]).astype(np.float32) * float(mpp)
    border = np.asarray([bool(item['border_flag']) for item in nuclei])
    edge_src, edge_dst, edge_feat, undirected_src, undirected_dst = build_knn_physical(xy_um, k=GRAPH_K, radius_um=GRAPH_RADIUS_UM, normalizer_um=GRAPH_NORMALIZER_UM)
    return GraphRecord(graph_id, 0, split, patient_id, '', 0, nucleus_id, xy_px, xy_um, rho_um, border, np.zeros(len(nucleus_id), np.int64), np.zeros(len(nucleus_id), np.uint64), edge_src, edge_dst, edge_feat, undirected_src, undirected_dst)

def graph_from_bytes(payload: bytes) -> GraphRecord:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        meta = data['metadata'].tolist()
        scalar = data['scalar']
        return GraphRecord(str(meta[0]), int(scalar[0]), str(meta[1]), str(meta[2]), str(meta[3]), int(scalar[1]), data['nucleus_id'], data['xy_px'], data['xy_um'], data['rho_um'], data['border_flag'], data['anchor_cell_id'], data['biological_entity_id'], data['edge_src'], data['edge_dst'], data['edge_feat'], data['undirected_edge_src'], data['undirected_edge_dst'])

def freeze_feature_stats(base: Path) -> None:
    source = Path(TRAINING_FEATURE_STATS)
    stats = json.loads(source.read_text(encoding='utf-8'))
    if len(stats.get('mean', [])) != RAY_COUNT or len(stats.get('std', [])) != RAY_COUNT or min(stats['std']) <= 0:
        raise RuntimeError('invalid frozen training feature statistics')
    atomic_json(base / 'graphs' / 'inference_feature_stats.json', {**stats, 'source_path': str(source), 'source_sha256': sha256_file(source), 'policy': 'frozen_training_only'})

def build_slide_graphs(job: dict[str, Any]) -> dict[str, Any]:
    base = Path(job['model_input_root'])
    slide_id = job['slide_id']
    freeze_feature_stats(base)
    masks = load_slide_masks(layout.mask_npz_path(base, slide_id))
    instances = load_slide_instances(layout.instances_path(base, slide_id))['tiles']
    tile_ids = []
    payloads = []
    empty = 0
    for row in job['tiles']:
        tile_id = str(row['tile_id'])
        graph_id = str(row['graph_id'])
        mask = masks[tile_id]
        record = record_from_mask(mask, instances[tile_id]['instances'], graph_id=graph_id, split=str(row.get('split') or ''), patient_id=str(row['patient_id']))
        if len(record.nucleus_id) == 0:
            empty += 1
        tile_ids.append(tile_id)
        payloads.append(np.frombuffer(record.to_bytes(), dtype=np.uint8))
    max_len = max((len(item) for item in payloads), default=0)
    packed = np.zeros((len(payloads), max_len), dtype=np.uint8)
    lengths = np.zeros(len(payloads), dtype=np.int64)
    for index, item in enumerate(payloads):
        packed[index, :len(item)] = item
        lengths[index] = len(item)
    dest = layout.graph_path(base, slide_id)
    atomic_npz(dest, compressed=True, tile_ids=np.asarray(tile_ids, dtype='U128'), graph_ids=np.asarray([str(row['graph_id']) for row in job['tiles']], dtype='U160'), lengths=lengths, payloads=packed)
    summary = {'slide_id': slide_id, 'stage': 'graph', 'status': 'done', 'n_tiles': len(tile_ids), 'empty_nuclei': empty, 'bytes': dest.stat().st_size}
    atomic_json(base / 'logs' / 'status' / f'{dest.stem}.graph.json', summary)
    return summary

def load_slide_graphs(path: Path) -> dict[str, GraphRecord]:
    with np.load(path, allow_pickle=False) as data:
        tile_ids = [str(item) for item in data['tile_ids'].tolist()]
        lengths = data['lengths']
        payloads = data['payloads']
    out = {}
    for index, tile_id in enumerate(tile_ids):
        raw = bytes(payloads[index, :int(lengths[index])].tobytes())
        out[tile_id] = graph_from_bytes(raw)
    return out
