from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
DATASETS = ('sicapv2', 'bracs', 'tcga_crc_msi')
DATASET_SPEC: dict[str, dict[str, Any]] = {'sicapv2': {'scene_shards': (0, 1), 'group_key': 'patient_id', 'folds': 4, 'classes': 4, 'metric': 'patch_qwk'}, 'bracs': {'scene_shards': (0, 1), 'group_key': 'wsi_id', 'folds': 5, 'classes': 7, 'metric': 'roi_macro_f1'}, 'tcga_crc_msi': {'scene_shards': (0, 1, 2, 3), 'group_key': 'patient_id', 'folds': 5, 'classes': 1, 'metric': 'mean_fold_patient_auroc', 'tile_route': True}}
TRAIN_SPLITS = ('train',)
EXTERNAL_SPLITS = ('val', 'validation')
TEST_SPLITS = ('test',)

def load_config(path: str | Path) -> dict:
    from celllift.runtime import yaml
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def config_path(config_dir: str | Path, dataset: str) -> Path:
    return Path(config_dir) / f'{dataset}.yaml'

def to_float(value):
    return None if value is None else float(value)

def _row(graph_id, split, fold, label_id, group_id, group_key, wsi_id=None, roi_id=None, patient_id=None):
    return {'graph_id': str(graph_id), 'split': str(split), 'fold': None if fold is None else int(fold), 'label_id': None if label_id is None else int(label_id), 'group_id': str(group_id), 'group_key': group_key, 'wsi_id': None if wsi_id is None else str(wsi_id), 'roi_id': None if roi_id is None else str(roi_id), 'patient_id': None if patient_id is None else str(patient_id)}

def _row_from_metadata(meta: Mapping[str, Any], dataset: str, fallback_id: str) -> dict:
    group_key = DATASET_SPEC[dataset]['group_key']
    return _row(meta.get('graph_id', fallback_id), meta.get('split', 'train'), meta.get('fold'), meta.get('label_id'), meta.get(group_key) or meta.get('patient_id') or fallback_id, group_key, wsi_id=meta.get('wsi_id'), roi_id=meta.get('roi_id'), patient_id=meta.get('patient_id'))

def dev_index_rows(dataset: str, cfg: Mapping[str, Any]) -> list[dict]:
    index = Path(cfg['paths']['data_root']) / dataset / 'scene_index.json'
    if not index.is_file():
        raise FileNotFoundError(f'scene index missing at {index}; run `build-cache` for {dataset} first')
    payload = json.loads(index.read_text(encoding='utf-8'))
    rows = []
    for entry in payload['graphs']:
        meta = entry.get('metadata') or {}
        row = _row_from_metadata(meta, dataset, entry['graph_id'])
        if row['fold'] is None:
            row['fold'] = None if meta.get('final_fold') is None else int(meta['final_fold'])
        rows.append(row)
    return rows

def train_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row['split'] in TRAIN_SPLITS and row['fold'] is not None]

def group_partition(rows: list[dict], validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    import numpy as np
    unique = sorted({row['group_id'] for row in rows})
    generator = np.random.default_rng(seed)
    order = np.asarray(unique, object)
    generator.shuffle(order)
    count = max(1, int(round(len(unique) * float(validation_fraction))))
    count = min(count, max(1, len(unique) - 1))
    validation_groups = set(order[:count].tolist())
    train = [row for row in rows if row['group_id'] not in validation_groups]
    validation = [row for row in rows if row['group_id'] in validation_groups]
    if not train or not validation:
        raise RuntimeError('group partition produced an empty side')
    return (train, validation)

def bag_index(rows: list[dict], dataset: str) -> dict[str, list[dict]]:
    from collections import defaultdict
    spec = DATASET_SPEC[dataset]
    if dataset == 'sicapv2':
        return {row['graph_id']: [row] for row in rows}
    key = 'roi_id' if dataset == 'bracs' else 'patient_id'
    output: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        output[str(row[key])].append(row)
    if dataset == 'tcga_crc_msi':
        output = defaultdict(list, {k: v for k, v in output.items() if len(v) >= int(spec.get('min_tiles', 10))})
    return dict(output)
