from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
DATASETS = ('sicapv2', 'bracs', 'tcga_crc_msi')
EXPERT_ARMS = ('E2', 'ES', 'ER')
FUSED_ARMS = ('F2', 'FS', 'FR')
ARM_TO_EXPERT = {'F2': 'E2', 'FS': 'ES', 'FR': 'ER'}
EXPERT_TO_FUSED = {'E2': 'F2', 'ES': 'FS', 'ER': 'FR'}
DATASET_SPEC: dict[str, dict[str, Any]] = {'sicapv2': {'scene_shards': (0, 1), 'group_key': 'patient_id', 'folds': 4, 'classes': 4, 'metric': 'patch_qwk', 'inner_folds': 3}, 'bracs': {'scene_shards': (0, 1), 'group_key': 'wsi_id', 'folds': 5, 'classes': 7, 'metric': 'roi_macro_f1', 'inner_folds': 3}, 'tcga_crc_msi': {'scene_shards': (0, 1, 2, 3), 'group_key': 'patient_id', 'folds': 5, 'classes': 1, 'metric': 'mean_fold_patient_auroc', 'tile_route': True, 'min_tiles': 10, 'inner_folds': 3}}
TRAIN_SPLITS = ('train',)
DEFAULT_BATCH = 'expert-conditional_geometry'

def batch_name(cfg: Mapping[str, Any] | None=None) -> str:
    if cfg is None:
        return DEFAULT_BATCH
    return str(cfg.get('batch') or DEFAULT_BATCH)

def cache_tag(cfg: Mapping[str, Any] | None=None) -> str:
    return batch_name(cfg).replace('-', '_')

def cache_root(cfg: Mapping[str, Any], dataset: str) -> Path:
    return Path(cfg['paths']['data_root']) / dataset / cache_tag(cfg)

def batch_result_root(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg['paths']['result_root']) / batch_name(cfg)

def load_config(path: str | Path) -> dict:
    from celllift.runtime import yaml
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def _row(graph_id, split, fold, label_id, group_id, group_key, wsi_id=None, roi_id=None, patient_id=None):
    return {'graph_id': str(graph_id), 'split': str(split), 'fold': None if fold is None else int(fold), 'label_id': None if label_id is None else int(label_id), 'group_id': str(group_id), 'group_key': group_key, 'wsi_id': None if wsi_id is None else str(wsi_id), 'roi_id': None if roi_id is None else str(roi_id), 'patient_id': None if patient_id is None else str(patient_id)}

def _row_from_metadata(meta: Mapping[str, Any], dataset: str, fallback_id: str) -> dict:
    group_key = DATASET_SPEC[dataset]['group_key']
    return _row(meta.get('graph_id', fallback_id), meta.get('split', 'train'), meta.get('fold'), meta.get('label_id'), meta.get(group_key) or meta.get('patient_id') or fallback_id, group_key, wsi_id=meta.get('wsi_id'), roi_id=meta.get('roi_id'), patient_id=meta.get('patient_id'))

def cache_index_path(cfg: Mapping[str, Any], dataset: str) -> Path:
    return cache_root(cfg, dataset) / 'scene_index.json'

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
        rows.append(row)
    return rows

def train_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row['split'] in TRAIN_SPLITS and row['fold'] is not None]
