from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .feature_store import read_index
from .fusion import crossfit
from .io_utils import atomic_json, atomic_npz, atomic_parquet
from .metrics import fold_stratified_patient_bootstrap, macro_f1, paired_cluster_bootstrap, qwk, registered_interpretation, t7_to_t3
from .protocol import DEEPSETS_SEEDS, MEANPOOL_SEEDS

def _rows(path: str | Path):
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _softmax(logits):
    value = np.asarray(logits, float)
    value = np.exp(value - value.max(-1, keepdims=True))
    return value / value.sum(-1, keepdims=True)

def _logit(probability, eps=1e-06):
    value = np.clip(np.asarray(probability, float), eps, 1 - eps)
    return np.log(value / (1 - value))

def _seed_set(encoder: str):
    return MEANPOOL_SEEDS if encoder == 'meanpool' else DEEPSETS_SEEDS

def _validate_seed_bundle(key, values, expected):
    seeds = {int(row['seed']) for row in values}
    if seeds != set(expected) or len(values) != len(expected):
        raise RuntimeError(f'seed bundle mismatch for {key}: {sorted(seeds)} != {list(expected)}')

def load_sicap_baseline(root: Path, encoder: str):
    seeds = _seed_set(encoder)
    grouped = defaultdict(list)
    for fold in range(4):
        for seed in seeds:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in _rows(path):
                grouped[fold, str(row['graph_id'])].append(row)
    output = {}
    for key, values in grouped.items():
        _validate_seed_bundle(key, values, seeds)
        output[key] = {'label': int(values[0]['label_id']), 'group': str(values[0]['patient_id']), 'logits': np.mean([np.asarray(row['logits'], float) for row in values], 0)}
    return output

def load_crc_baseline(root: Path, encoder: str, eligible_graphs: set[str]):
    seeds = _seed_set(encoder)
    tiles = defaultdict(list)
    for fold in range(5):
        by_graph = defaultdict(list)
        for seed in seeds:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in _rows(path):
                if str(row['graph_id']) in eligible_graphs:
                    by_graph[str(row['graph_id'])].append(row)
        for graph_id, values in by_graph.items():
            _validate_seed_bundle((fold, graph_id), values, seeds)
            probability = float(np.mean([row['probabilities'][1] for row in values]))
            first = values[0]
            tiles[fold, str(first['patient_id'])].append((probability, int(first['label_id'])))
    output = {}
    for key, values in tiles.items():
        if len(values) < 10:
            continue
        labels = {label for _, label in values}
        if len(labels) != 1:
            raise RuntimeError(f'CRC patient label mismatch: {key}')
        positive = sum((probability >= 0.5 for probability, _ in values))
        total = len(values)
        continuity = float(np.log((positive + 0.5) / (total - positive + 0.5)))
        output[key] = {'label': values[0][1], 'group': key[1], 'logit': continuity, 'tiles': total}
    return output

def load_bracs_baseline(root: Path, encoder: str):
    seeds = _seed_set(encoder)
    grouped = defaultdict(list)
    for fold in range(5):
        for seed in seeds:
            path = root / f'experts/development/t7/{encoder}/E0_RGB_MASK2D/fold_{fold:02d}/seed_{seed}/predictions.parquet'
            for row in _rows(path):
                if row['role'] == 'heldout':
                    grouped[fold, str(row['roi_id'])].append(row)
    output = {}
    for key, values in grouped.items():
        _validate_seed_bundle(key, values, seeds)
        probability = np.mean([np.asarray(row['probability'], float) for row in values], 0)
        output[key] = {'label': int(values[0]['label']), 'group': str(values[0]['wsi_id']), 'logits': np.log(np.clip(probability, 1e-07, 1.0))}
    return output

def load_geometry(result_root: Path, encoder: str, arm: str, folds: int):
    seeds = _seed_set(encoder)
    grouped = defaultdict(list)
    for fold in range(folds):
        for seed in seeds:
            path = result_root / f'06_experts/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}/predictions.parquet'
            for row in _rows(path):
                grouped[fold, str(row['sample_id'])].append(row)
    output = {}
    for key, values in grouped.items():
        _validate_seed_bundle(key, values, seeds)
        labels = {int(row['label_id']) for row in values}
        if len(labels) != 1:
            raise RuntimeError(f'geometry label mismatch: {key}')
        output[key] = {'label': int(values[0]['label_id']), 'logits': np.mean([np.asarray(row['logits'], float) for row in values], 0)}
    return output

def evaluate_oof(cfg: Mapping[str, Any], encoder: str) -> dict:
    dataset = cfg['dataset']
    result_root = Path(cfg['paths']['result_root'])
    baseline_root = Path(cfg['paths']['baseline_predictions'])
    modal = read_index(Path(cfg['paths']['data_root']) / '03_modal_features/index.parquet')
    if dataset == 'sicapv2':
        baseline = load_sicap_baseline(baseline_root, encoder)
        folds_count = 4
    elif dataset == 'tcga_crc_msi':
        baseline = load_crc_baseline(baseline_root, encoder, {row.graph_id for row in modal})
        folds_count = 5
    else:
        baseline = load_bracs_baseline(baseline_root, encoder)
        folds_count = 5
    keys = sorted(baseline)
    if not keys:
        raise RuntimeError(f'empty baseline OOF predictions for {dataset}/{encoder}')
    labels = np.asarray([baseline[key]['label'] for key in keys])
    folds = np.asarray([key[0] for key in keys])
    groups = np.asarray([baseline[key]['group'] for key in keys], object)
    if dataset == 'tcga_crc_msi':
        baseline_logits = np.asarray([baseline[key]['logit'] for key in keys])
    else:
        baseline_logits = np.stack([baseline[key]['logits'] for key in keys])
    scores = {}
    predictions = []
    scores['B0'] = 1 / (1 + np.exp(-baseline_logits)) if dataset == 'tcga_crc_msi' else _softmax(baseline_logits)
    fusion_payload = {}
    for arm in ('D', 'DS', 'R', 'RS'):
        geometry = load_geometry(result_root, encoder, arm, folds_count)
        if set(geometry) != set(keys):
            missing = sorted(set(keys) - set(geometry))[:5]
            extra = sorted(set(geometry) - set(keys))[:5]
            raise RuntimeError(f'baseline/geometry OOF join mismatch for {dataset}/{encoder}/{arm}: missing={missing}, extra={extra}')
        if any((geometry[key]['label'] != baseline[key]['label'] for key in keys)):
            raise RuntimeError(f'baseline/geometry label mismatch for {dataset}/{encoder}/{arm}')
        geometry_logits = np.asarray([geometry[key]['logits'][0] for key in keys]) if dataset == 'tcga_crc_msi' else np.stack([geometry[key]['logits'] for key in keys])
        fused, parameters = crossfit(folds, baseline_logits, geometry_logits, labels, crc=dataset == 'tcga_crc_msi')
        scores[arm] = 1 / (1 + np.exp(-fused)) if dataset == 'tcga_crc_msi' else _softmax(fused)
        fusion_payload[arm] = {str(fold): value.__dict__ for fold, value in parameters.items()}
        for key, label, value in zip(keys, labels, scores[arm]):
            predictions.append({'sample_id': key[1], 'fold': key[0], 'label_id': int(label), 'group_id': baseline[key]['group'], 'arm': arm, 'score': float(value) if np.ndim(value) == 0 else None, 'probabilities': None if np.ndim(value) == 0 else np.asarray(value).tolist(), 'calibration': parameters[key[0]].__dict__})
    for key, label, value in zip(keys, labels, scores['B0']):
        predictions.append({'sample_id': key[1], 'fold': key[0], 'label_id': int(label), 'group_id': baseline[key]['group'], 'arm': 'B0', 'score': float(value) if np.ndim(value) == 0 else None, 'probabilities': None if np.ndim(value) == 0 else np.asarray(value).tolist(), 'calibration': None})
    replicates = int(cfg.get('runtime', {}).get('bootstrap_replicates', 10000))
    if dataset == 'sicapv2':
        observed, distributions = paired_cluster_bootstrap(labels=labels, groups=groups, scores=scores, metric=qwk, replicates=replicates)
    elif dataset == 'bracs':
        observed, distributions = paired_cluster_bootstrap(labels=labels, groups=groups, scores=scores, metric=macro_f1, replicates=replicates)
    else:
        observed, distributions = fold_stratified_patient_bootstrap(labels=labels, folds=folds, scores=scores, replicates=replicates)
    interpretation = registered_interpretation(observed, distributions)
    output_root = result_root / f'08_metrics/{encoder}'
    atomic_parquet(output_root / 'oof_predictions.parquet', predictions)
    atomic_npz(output_root / 'bootstrap_replicates.npz', **distributions)
    atomic_json(result_root / f'07_fusion/{encoder}/parameters.json', {'status': 'PASS', 'arms': fusion_payload})
    payload = {'status': 'PASS', 'dataset': dataset, 'encoder': encoder, 'samples': len(keys), 'bootstrap_replicates': replicates, 'observed': observed, **interpretation}
    if dataset == 'bracs':
        labels_t3 = np.where(labels <= 2, 0, np.where(labels <= 4, 1, 2))
        payload['descriptive_t3_macro_f1'] = {arm: macro_f1(labels_t3, t7_to_t3(value)) for arm, value in scores.items()}
    atomic_json(output_root / 'summary.json', payload)
    return payload
