from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .io import job_dir, reused, write_pass
from .metrics import macro_f1, patient_auroc, qwk
from .paths import baseline_result_root, comparison_result_root, result_root

def _as_probability(row: Mapping[str, Any]) -> list[float]:
    raw = row.get('probability')
    if raw is None:
        raw = row.get('final_probability')
    arr = np.asarray(raw, np.float64).reshape(-1)
    if arr.size == 1:
        score = float(np.clip(arr[0], 1e-06, 1 - 1e-06))
        return [1.0 - score, score]
    arr = np.clip(arr, 1e-06, 1)
    arr = arr / arr.sum()
    return arr.tolist()

def _load_pred(path: Path) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    out = {}
    for row in rows:
        item = dict(row)
        item['probability'] = _as_probability(item)
        out[str(item['sample_id'])] = item
    return out

def pred_path(dataset: str, family: str, arm: str, unit: str, seed: int=42, fold: int=0) -> Path:
    relative = Path(dataset) / family / arm / unit / f'seed_{int(seed)}' / f'fold_{int(fold):02d}' / 'predictions.parquet'
    for root in (comparison_result_root(), baseline_result_root(), result_root()):
        path = root / relative
        if path.is_file():
            return path
    return comparison_result_root() / relative

def _metric(dataset: str, labels, probs):
    probs = np.asarray(probs, np.float64)
    labels = np.asarray(labels)
    if dataset == 'sicapv2':
        return qwk(labels, probs.argmax(1))
    if dataset == 'bracs':
        return macro_f1(labels, probs.argmax(1), probs.shape[1])
    scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.reshape(-1)
    return patient_auroc(labels, scores)

def fuse_from_job(job: Mapping[str, Any], device: str='cpu') -> dict[str, Any]:
    del device
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    dataset = job['dataset']
    unit = job.get('unit') or 'official'
    fold = int(job.get('fold') or 0)
    seed = int(job.get('seed') or 42)
    b_path = pred_path(dataset, job['partner_family'], job['partner_arm'], unit, seed, fold)
    e_path = pred_path(dataset, job['expert_family'], job['expert_arm'], unit, seed, fold)
    if not b_path.is_file() or not e_path.is_file():
        return {'status': 'BLOCKED', 'reason': 'missing_predictions', 'b': str(b_path), 'e': str(e_path)}
    try:
        b_rows, e_rows = (_load_pred(b_path), _load_pred(e_path))
    except FileNotFoundError:
        return {'status': 'BLOCKED', 'reason': 'missing_predictions', 'b': str(b_path), 'e': str(e_path)}
    keys = sorted(set(b_rows) & set(e_rows))
    labels = np.asarray([int(b_rows[k]['label_id']) for k in keys])
    b = np.stack([np.asarray(b_rows[k]['probability'], np.float64) for k in keys], 0)
    e = np.stack([np.asarray(e_rows[k]['probability'], np.float64) for k in keys], 0)
    if not np.isfinite(b).all() or not np.isfinite(e).all():
        return {'status': 'BLOCKED', 'reason': 'non_finite_predictions', 'n': len(keys)}
    b = np.clip(b, 1e-06, 1)
    e = np.clip(e, 1e-06, 1)
    b = b / b.sum(1, keepdims=True)
    e = e / e.sum(1, keepdims=True)
    if job.get('identity'):
        weights = {'a': 0.0, 'temperature': 1.0}
        predictions = [{'sample_id': key, 'label_id': int(label), 'probability': prob.tolist(), 'arm': job['arm']} for key, label, prob in zip(keys, labels, b)]
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'weights.json').write_text(json.dumps(weights, indent=2), encoding='utf-8')
        return write_pass(destination, {'dataset': dataset, 'arm': job['arm'], 'best': _metric(dataset, labels, b), 'weights': weights, 'fit_split': 'identity_copy', 'test_used': False, 'n': len(keys)}, predictions)
    temperature = False if job.get('baseline_qualification') or job.get('r3') or job.get('no_temperature') else bool(job.get('temperature'))
    if b.shape[1] == 2:
        logit_b = np.log(b[:, 1]) - np.log(b[:, 0])
        logit_e = np.log(e[:, 1]) - np.log(e[:, 0])
        best_a, best_t, best = (0.0, 1.0, -1000000000.0)
        temps = np.linspace(0.25, 4.0, 16) if temperature else np.asarray([1.0])
        for t in temps:
            expert = logit_e / t
            if temperature:
                expert = expert - expert.mean()
            for a in np.linspace(0.0, 4.0, 41):
                fused = 1 / (1 + np.exp(-(logit_b + a * expert)))
                metric = _metric(dataset, labels, np.stack([1 - fused, fused], 1))
                better = metric > best + 1e-12
                tie = abs(metric - best) <= 1e-12 and float(a) < best_a
                if better or tie:
                    best, best_a, best_t = (metric, float(a), float(t))
        expert = logit_e / best_t
        if temperature:
            expert = expert - expert.mean()
        fused = 1 / (1 + np.exp(-(logit_b + best_a * expert)))
        probs = np.stack([1 - fused, fused], 1)
        weights = {'a': best_a, 'temperature': best_t}
    else:
        logit_b = np.log(b)
        logit_e = np.log(e)
        logit_b = logit_b - logit_b.mean(1, keepdims=True)
        logit_e = logit_e - logit_e.mean(1, keepdims=True)
        best_a, best = (0.0, -1000000000.0)
        for a in np.linspace(0.0, 4.0, 41):
            z = logit_b + a * logit_e
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p = p / p.sum(1, keepdims=True)
            metric = _metric(dataset, labels, p)
            better = metric > best + 1e-12
            tie_smaller_a = abs(metric - best) <= 1e-12 and float(a) < best_a
            if better or tie_smaller_a:
                best, best_a = (metric, float(a))
        weights = {'a': best_a, 'temperature': 1.0}
        z = logit_b + best_a * logit_e
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        probs = p / p.sum(1, keepdims=True)
    predictions = [{'sample_id': key, 'label_id': int(label), 'probability': prob.tolist(), 'arm': job['arm']} for key, label, prob in zip(keys, labels, probs)]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'weights.json').write_text(json.dumps(weights, indent=2), encoding='utf-8')
    return write_pass(destination, {'dataset': dataset, 'arm': job['arm'], 'best': best, 'weights': weights, 'fit_split': 'val_or_oof_only', 'test_used': False, 'n': len(keys)}, predictions)
