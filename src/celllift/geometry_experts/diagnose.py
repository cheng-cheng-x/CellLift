from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .dataset import DATASET_SPEC, EXPERT_ARMS, batch_result_root
from .io_utils import atomic_json, read_json, read_parquet
from .train_expert import job_dir

def _jobs(cfg: Mapping[str, Any], dataset: str, seed: int=42) -> list[dict]:
    rows = []
    for arm in EXPERT_ARMS:
        for outer in range(int(DATASET_SPEC[dataset]['folds'])):
            for inner in range(int(DATASET_SPEC[dataset]['inner_folds'])):
                path = job_dir(cfg, dataset, arm, seed, outer, inner) / 'job.json'
                if path.is_file():
                    rows.append(read_json(path))
    return rows

def _oof(cfg: Mapping[str, Any], dataset: str, seed: int=42) -> list[dict]:
    path = batch_result_root(cfg) / dataset / 'metrics' / f'oof_seed{seed}.parquet'
    return read_parquet(path)

def _confusion(labels: np.ndarray, pred: np.ndarray, classes: int) -> list[list[int]]:
    table = np.zeros((classes, classes), np.int64)
    for truth, guess in zip(labels, pred):
        table[int(truth), int(guess)] += 1
    return table.tolist()

def _sicap(cfg: Mapping[str, Any], seed: int=42) -> dict[str, Any]:
    rows = _oof(cfg, 'sicapv2', seed)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[str(row['arm'])].append(row)
    experts = {}
    for arm in EXPERT_ARMS:
        labels = np.asarray([int(row['label_id']) for row in by_arm[arm]], np.int64)
        pred = np.asarray([np.asarray(row['probability'], float).argmax() for row in by_arm[arm]], np.int64)
        experts[arm] = {'n': int(len(labels)), 'pred_nc': int((pred == 0).sum()), 'pred_g5': int((pred == 3).sum()), 'true_nc': int((labels == 0).sum()), 'recall': [float(((pred == k) & (labels == k)).sum() / max(1, (labels == k).sum())) for k in range(4)], 'confusion': _confusion(labels, pred, 4)}
    jobs = _jobs(cfg, 'sicapv2', seed)
    nll = [float(job['best_nll']) for job in jobs if np.isfinite(job.get('best_nll', np.nan))]
    runs = [int(job['max_run_same_group']) for job in jobs if job.get('max_run_same_group') is not None]
    return {'experts': experts, 'median_best_val_nll': float(np.median(nll)) if nll else None, 'max_run_same_group': {'max': max(runs) if runs else None, 'median': float(np.median(runs)) if runs else None}, 'class_recovery': all((experts[arm]['pred_nc'] / max(1, experts[arm]['n']) < 0.7 and experts[arm]['recall'][3] > 0 for arm in EXPERT_ARMS))}

def _bracs(cfg: Mapping[str, Any], seed: int=42) -> dict[str, Any]:
    summary = read_json(batch_result_root(cfg) / 'bracs' / 'metrics' / f'summary_seed{seed}.json')
    wanted = {row['comparison']: row for row in summary.get('comparisons', [])}
    return {'observed': summary.get('observed'), 'ES-E2': wanted.get('ES-E2'), 'ER-ES': wanted.get('ER-ES'), 'FS': summary.get('observed', {}).get('FS'), 'FR': summary.get('observed', {}).get('FR')}

def _pairs(labels: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        return (0, 0)
    left = scores[pos][:, None]
    right = scores[neg][None, :]
    return (int((left > right).sum()), int((left < right).sum()))

def _crc(cfg: Mapping[str, Any], seed: int=42) -> dict[str, Any]:
    rows = _oof(cfg, 'tcga_crc_msi', seed)
    by_arm: dict[str, dict[str, float]] = defaultdict(dict)
    labels = {}
    for row in rows:
        key = str(row['sample_id'])
        score = float(np.asarray(row['probability'], float).reshape(-1)[0])
        by_arm[str(row['arm'])][key] = score
        labels[key] = int(row['label_id'])
    keys = sorted(labels)
    y = np.asarray([labels[key] for key in keys], np.int64)
    b = np.asarray([by_arm['B'][key] for key in keys], np.float64)
    result = {}
    b_correct, b_wrong = _pairs(y, b)
    for arm in EXPERT_ARMS:
        other = np.asarray([by_arm[arm][key] for key in keys], np.float64)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if not len(pos) or not len(neg):
            result[arm] = {'corrected': 0, 'introduced': 0, 'net': 0}
            continue
        b_order = b[pos][:, None] - b[neg][None, :]
        e_order = other[pos][:, None] - other[neg][None, :]
        corrected = int(((b_order < 0) & (e_order > 0)).sum())
        introduced = int(((b_order > 0) & (e_order < 0)).sum())
        result[arm] = {'corrected': corrected, 'introduced': introduced, 'net': corrected - introduced, 'pairs': int(len(pos) * len(neg))}
    return {'b_correct_pairs': b_correct, 'b_wrong_pairs': b_wrong, 'experts': result}

def diagnose_dataset(cfg: Mapping[str, Any], dataset: str, seed: int=42) -> dict[str, Any]:
    if dataset == 'sicapv2':
        payload = _sicap(cfg, seed)
    elif dataset == 'bracs':
        payload = _bracs(cfg, seed)
    else:
        payload = _crc(cfg, seed)
    payload = {'status': 'PASS', 'dataset': dataset, 'seed': seed, **payload}
    destination = batch_result_root(cfg) / dataset / 'metrics' / f'diagnose_seed{seed}.json'
    atomic_json(destination, payload)
    return payload
