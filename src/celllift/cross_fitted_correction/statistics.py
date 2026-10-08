from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.geometry_baselines.io_utils import atomic_json, atomic_parquet
from .data import read_parquet
from .protocol import PROTOCOL_ID

def _load(cfg: Mapping[str, Any], pooler: str, fusion: str, arm: str, seed: int) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'screen' / pooler / fusion / arm / f'seed_{seed}'
    rows = []
    for fold in range(int(cfg['split']['validation_folds'])):
        rows.extend(read_parquet(root / f'fold_{fold}' / 'validation_predictions.parquet'))
    return rows

def _qwk(labels: np.ndarray, prediction: np.ndarray) -> float:
    classes = 4
    matrix = np.bincount(labels * classes + prediction, minlength=classes ** 2).reshape(classes, classes).astype(float)
    n = matrix.sum()
    expected = np.outer(matrix.sum(1), matrix.sum(0)) / n
    weight = (np.arange(classes)[:, None] - np.arange(classes)[None, :]) ** 2 / 9.0
    denominator = float((weight * expected).sum())
    return 1.0 - float((weight * matrix).sum()) / denominator if denominator > 0 else 0.0

def _auc(labels: np.ndarray, score: np.ndarray) -> float:
    from scipy.stats import rankdata
    rank = rankdata(score)
    positive = labels == 1
    n1, n0 = (int(positive.sum()), int((~positive).sum()))
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((rank[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def _fold_mean_auc(labels: np.ndarray, score: np.ndarray, folds: np.ndarray) -> float:
    return float(np.mean([_auc(labels[folds == fold], score[folds == fold]) for fold in np.unique(folds)]))

def _crc_patients(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row['patient_id'])].append(row)
    output = {}
    for patient, values in groups.items():
        if len(values) < 10:
            continue
        if key == 'rgb':
            score = np.mean([float(row['rgb_logits']) >= 0 for row in values])
        else:
            score = np.mean([float(row['final_logits']) >= 0 for row in values])
        output[patient] = {'label': int(values[0]['label_id']), 'fold': int(values[0]['fold']), 'score': float(score)}
    return output

def _holm(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]['p_one_sided'])
    running = 0.0
    m = len(rows)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[index]['p_one_sided']))
        rows[index]['holm_p'] = running

def run_statistics(cfg: Mapping[str, Any], *, pooler: str, fusion: str, seed: int, comparisons: Sequence[str], replicates: int=10000, family: str='residual') -> dict[str, Any]:
    needed = sorted({part for value in comparisons for part in value.split('-') if part != 'RGB'})
    loaded = {arm: _load(cfg, pooler, fusion, arm, seed) for arm in needed}
    rng = np.random.default_rng(20260831)
    output = []
    if cfg['dataset'] == 'sicapv2':
        reference = loaded[needed[0]]
        keys = [str(row['graph_id']) for row in reference]
        labels = np.asarray([int(row['label_id']) for row in reference])
        patients = np.asarray([str(row['patient_id']) for row in reference])
        prediction: dict[str, np.ndarray] = {'RGB': np.asarray([np.argmax(row['rgb_logits']) for row in reference])}
        for arm, rows in loaded.items():
            mapping = {str(row['graph_id']): row for row in rows}
            if set(mapping) != set(keys):
                raise RuntimeError(f'SICAP comparison join mismatch: {arm}')
            prediction[arm] = np.asarray([np.argmax(mapping[key]['final_logits']) for key in keys])
        unique = np.unique(patients)
        groups = {patient: np.flatnonzero(patients == patient) for patient in unique}
        for comparison in comparisons:
            left, right = comparison.split('-')
            observed = _qwk(labels, prediction[left]) - _qwk(labels, prediction[right])
            values = []
            for _ in range(replicates):
                sampled = rng.choice(unique, len(unique), replace=True)
                index = np.concatenate([groups[p] for p in sampled])
                values.append(_qwk(labels[index], prediction[left][index]) - _qwk(labels[index], prediction[right][index]))
            values = np.asarray(values)
            output.append({'comparison': comparison, 'delta': observed, 'ci_low': float(np.quantile(values, 0.025)), 'ci_high': float(np.quantile(values, 0.975)), 'p_one_sided': float((1 + (values <= 0).sum()) / (replicates + 1))})
    else:
        patient = {arm: _crc_patients(rows, 'final') for arm, rows in loaded.items()}
        patient['RGB'] = _crc_patients(loaded[needed[0]], 'rgb')
        keys = sorted(patient['RGB'])
        if any((set(patient[arm]) != set(keys) for arm in patient)):
            raise RuntimeError('CRC comparison patient join mismatch')
        labels = np.asarray([patient['RGB'][key]['label'] for key in keys])
        folds = np.asarray([patient['RGB'][key]['fold'] for key in keys])
        scores = {arm: np.asarray([patient[arm][key]['score'] for key in keys]) for arm in patient}
        fold_groups = {fold: np.flatnonzero(folds == fold) for fold in np.unique(folds)}
        for comparison in comparisons:
            left, right = comparison.split('-')
            observed = _fold_mean_auc(labels, scores[left], folds) - _fold_mean_auc(labels, scores[right], folds)
            values = []
            for _ in range(replicates):
                index = np.concatenate([rng.choice(group, len(group), replace=True) for group in fold_groups.values()])
                value = _fold_mean_auc(labels[index], scores[left][index], folds[index]) - _fold_mean_auc(labels[index], scores[right][index], folds[index])
                if np.isfinite(value):
                    values.append(value)
            values = np.asarray(values)
            output.append({'comparison': comparison, 'delta': observed, 'ci_low': float(np.quantile(values, 0.025)), 'ci_high': float(np.quantile(values, 0.975)), 'p_one_sided': float((1 + (values <= 0).sum()) / (len(values) + 1))})
    _holm(output)
    root = Path(cfg['paths']['result_root']) / 'statistics' / pooler / fusion / f'seed_{seed}' / family
    atomic_parquet(root / 'paired_bootstrap.parquet', output)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'family': family, 'replicates': replicates, 'cluster': 'patient', 'validation_only': True, 'official_test_touched': False, 'comparisons': output}
    atomic_json(root / 'summary.json', payload)
    return payload
