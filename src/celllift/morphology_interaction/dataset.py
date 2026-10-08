from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
DATASETS = ('sicapv2', 'bracs', 'tcga_crc_msi')
ROUTES = ('a', 'b', 'c', 'd')
ROUTE_ARMS = {'a': ('A2', 'AS', 'AR'), 'b': ('B2', 'B3'), 'c': ('C2', 'CX', 'C3'), 'd': ('D2', 'D3')}
ROUTE_PAIRS = {'a': (('AR', 'B'), ('AR', 'A2'), ('AS', 'A2'), ('AR', 'AS'), ('A2', 'B'), ('AS', 'B')), 'b': (('B3', 'B'), ('B3', 'B2'), ('B2', 'B')), 'c': (('C3', 'B'), ('C3', 'C2'), ('C3', 'CX'), ('CX', 'C2'), ('C2', 'B')), 'd': (('D3', 'D2'), ('D3', 'B'), ('D2', 'B'))}
ROUTE_PRIMARY = {'a': ('AR-B', 'AR-A2'), 'b': ('B3-B', 'B3-B2'), 'c': ('C3-B', 'C3-C2'), 'd': ('D3-D2', 'D3-B')}
ARM_NODE_3D = {'A2': False, 'AS': True, 'AR': True, 'B2': False, 'B3': True, 'C2': False, 'CX': False, 'C3': True, 'D2': False, 'D3': True}
ARM_EDGE_3D = {'A2': False, 'AS': False, 'AR': True, 'B2': False, 'B3': True, 'C2': False, 'CX': False, 'C3': True, 'D2': False, 'D3': True}
DATASET_SPEC: dict[str, dict[str, Any]] = {'sicapv2': {'scene_shards': (0, 1), 'group_key': 'patient_id', 'folds': 4, 'classes': 4, 'metric': 'patch_qwk', 'mpp': 0.46, 'image_size': 1024, 'crc_pad': 0}, 'bracs': {'scene_shards': (0, 1), 'group_key': 'wsi_id', 'folds': 5, 'classes': 7, 'metric': 'roi_macro_f1', 'mpp': 0.46, 'image_size': 1024, 'crc_pad': 0}, 'tcga_crc_msi': {'scene_shards': (0, 1, 2, 3), 'group_key': 'patient_id', 'folds': 5, 'classes': 1, 'metric': 'mean_fold_patient_auroc', 'tile_route': True, 'min_tiles': 10, 'mpp': 0.459605, 'image_size': 557, 'crc_pad': 233}}
TRAIN_SPLITS = ('train',)
DEFAULT_CACHE_BATCH = 'parallel-set_encoding'
DEFAULT_RESULT_BATCH = 'parallel-a'
DINO_DIM = 384
SPATIAL_HW = 74
DINO_PAD = 6
DINO_ENCODED = 1036
PREDICTION_FORM = 'full'
B_ROLE = 'published_comparator'

def load_config(path: str | Path) -> dict:
    from celllift.runtime import yaml
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def cache_batch(cfg: Mapping[str, Any] | None=None) -> str:
    if cfg is None:
        return DEFAULT_CACHE_BATCH
    return str(cfg.get('cache_batch') or DEFAULT_CACHE_BATCH)

def result_batch(cfg: Mapping[str, Any] | None=None, route: str | None=None) -> str:
    official = None if cfg is None else cfg.get('official_result_subdir')
    if official:
        return str(official)
    if route:
        return f'parallel-{route}'
    if cfg is None:
        return DEFAULT_RESULT_BATCH
    return str(cfg.get('batch') or DEFAULT_RESULT_BATCH)

def cache_tag(cfg: Mapping[str, Any] | None=None) -> str:
    return cache_batch(cfg).replace('-', '_')

def cache_root(cfg: Mapping[str, Any], dataset: str) -> Path:
    return Path(cfg['paths']['data_root']) / dataset / cache_tag(cfg)

def batch_result_root(cfg: Mapping[str, Any], route: str | None=None) -> Path:
    return Path(cfg['paths']['result_root']) / result_batch(cfg, route)

def cache_index_path(cfg: Mapping[str, Any], dataset: str) -> Path:
    return cache_root(cfg, dataset) / 'scene_index.json'

def _row(graph_id, split, fold, label_id, group_id, group_key, wsi_id=None, roi_id=None, patient_id=None):
    return {'graph_id': str(graph_id), 'split': str(split), 'fold': None if fold is None else int(fold), 'label_id': None if label_id is None else int(label_id), 'group_id': str(group_id), 'group_key': group_key, 'wsi_id': None if wsi_id is None else str(wsi_id), 'roi_id': None if roi_id is None else str(roi_id), 'patient_id': None if patient_id is None else str(patient_id)}

def _row_from_metadata(meta: Mapping[str, Any], dataset: str, fallback_id: str) -> dict:
    group_key = DATASET_SPEC[dataset]['group_key']
    return _row(meta.get('graph_id', fallback_id), meta.get('split', 'train'), meta.get('fold'), meta.get('label_id'), meta.get(group_key) or meta.get('patient_id') or fallback_id, group_key, wsi_id=meta.get('wsi_id'), roi_id=meta.get('roi_id'), patient_id=meta.get('patient_id'))

def dev_index_rows(dataset: str, cfg: Mapping[str, Any]) -> list[dict]:
    index = cache_index_path(cfg, dataset)
    if not index.is_file():
        raise FileNotFoundError(f'scene index missing at {index}; run build-cache first')
    payload = json.loads(index.read_text(encoding='utf-8'))
    rows = []
    for entry in payload['graphs']:
        meta = entry.get('metadata') or {}
        row = _row_from_metadata(meta, dataset, entry['graph_id'])
        if row['fold'] is None and meta.get('final_fold') is not None:
            row['fold'] = int(meta['final_fold'])
        row['rgb_path'] = entry.get('rgb_path') or meta.get('rgb_path')
        row['mask_path'] = entry.get('mask_path') or meta.get('nucleus_mask_path')
        rows.append(row)
    return rows

def train_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row['split'] in TRAIN_SPLITS and row['fold'] is not None]

def route_arms(route: str) -> tuple[str, ...]:
    if route not in ROUTE_ARMS:
        raise ValueError(route)
    return ROUTE_ARMS[route]
