from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from typing import Any, Mapping
import numpy as np
from .dataset import DATASET_SPEC, ROUTE_PRIMARY, batch_result_root, route_arms
from .foundation.loop import job_dir
from .io_utils import atomic_json, read_json, read_parquet

def _jobs(cfg: Mapping[str, Any], dataset: str, route: str, seed: int=42) -> list[dict]:
    rows = []
    for arm in route_arms(route):
        for outer in range(int(DATASET_SPEC[dataset]['folds'])):
            path = job_dir(cfg, dataset, route, arm, seed, outer) / 'job.json'
            if path.is_file():
                rows.append(read_json(path))
    return rows

def _oof(cfg: Mapping[str, Any], dataset: str, route: str, seed: int=42) -> list[dict]:
    return read_parquet(batch_result_root(cfg, route) / dataset / 'metrics' / f'oof_seed{seed}.parquet')

def _confusion(labels: np.ndarray, pred: np.ndarray, classes: int) -> list[list[int]]:
    table = np.zeros((classes, classes), np.int64)
    for truth, guess in zip(labels, pred):
        table[int(truth), int(guess)] += 1
    return table.tolist()

def _sicap(cfg, route, seed):
    rows = _oof(cfg, 'sicapv2', route, seed)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[str(row['arm'])].append(row)
    arms = {}
    for arm in ['B', *route_arms(route)]:
        if arm not in by_arm:
            continue
        labels = np.asarray([int(row['label_id']) for row in by_arm[arm]], np.int64)
        pred = np.asarray([np.asarray(row['probability'], float).argmax() for row in by_arm[arm]], np.int64)
        arms[arm] = {'n': int(len(labels)), 'pred_nc': int((pred == 0).sum()), 'pred_g5': int((pred == 3).sum()), 'true_nc': int((labels == 0).sum()), 'recall': [float(((pred == k) & (labels == k)).sum() / max(1, (labels == k).sum())) for k in range(4)], 'confusion': _confusion(labels, pred, 4)}
    jobs = _jobs(cfg, 'sicapv2', route, seed)
    used = [str(job.get('early_stop_metric')) for job in jobs]
    runs = [int(job['max_run_same_group']) for job in jobs if job.get('max_run_same_group') is not None]
    return {'arms': arms, 'early_stop_metric_counts': {name: used.count(name) for name in sorted(set(used))}, 'max_run_same_group': {'max': max(runs) if runs else None, 'median': float(np.median(runs)) if runs else None}}

def _pairs(labels: np.ndarray, scores: np.ndarray) -> tuple[int, int]:
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if not len(pos) or not len(neg):
        return (0, 0)
    left = scores[pos][:, None]
    right = scores[neg][None, :]
    return (int((left > right).sum()), int((left < right).sum()))

def _crc(cfg, route, seed):
    rows = _oof(cfg, 'tcga_crc_msi', route, seed)
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
    for arm in route_arms(route):
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
    return {'b_correct_pairs': b_correct, 'b_wrong_pairs': b_wrong, 'arms': result, 'crc_vote_order': 'mean_then_hard_vote', 'positive_rate_not_reported': True}

def diagnose_dataset(cfg: Mapping[str, Any], dataset: str, route: str, seed: int=42) -> dict[str, Any]:
    if dataset == 'sicapv2':
        payload = _sicap(cfg, route, seed)
    elif dataset == 'bracs':
        summary = read_json(batch_result_root(cfg, route) / 'bracs' / 'metrics' / f'summary_seed{seed}.json')
        wanted = {row['comparison']: row for row in summary.get('comparisons', [])}
        payload = {'observed': summary.get('observed') or {}, 'comparisons': wanted, 'primary': ROUTE_PRIMARY[route]}
    else:
        payload = _crc(cfg, route, seed)
    payload = {'status': 'PASS', 'dataset': dataset, 'route': route, 'seed': seed, **payload}
    atomic_json(batch_result_root(cfg, route) / dataset / 'metrics' / f'diagnose_seed{seed}.json', payload)
    return payload
