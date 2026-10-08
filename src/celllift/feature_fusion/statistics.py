from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.geometry_baselines.io_utils import atomic_json, atomic_parquet
from celllift.cross_fitted_correction.statistics import _auc, _fold_mean_auc, _holm, _qwk
from .evaluation import load_arm
from .protocol import PROTOCOL_ID

def _crc_patients(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row['patient_id'])].append(row)
    output = {}
    for patient, values in groups.items():
        if len(values) < 10:
            continue
        if key == 'paper':
            score = float(np.mean([int(np.argmax(row['paper_logits'])) == 1 for row in values]))
        else:
            score = float(np.mean([float(row['final_logits']) >= 0 for row in values]))
        output[patient] = {'label': int(values[0]['label_id']), 'fold': int(values[0]['fold']), 'score': score}
    return output

def run_statistics(cfg: Mapping[str, Any], *, seed: int, comparisons: Sequence[str], replicates: int=10000, family: str='residual') -> dict[str, Any]:
    needed = sorted({part for value in comparisons for part in value.split('-') if part != 'R0'})
    loaded = {arm: load_arm(cfg, arm, seed) for arm in needed}
    rng = np.random.default_rng(20260901)
    output = []
    if cfg['dataset'] == 'sicapv2':
        reference = loaded[needed[0]]
        keys = [str(row['graph_id']) for row in reference]
        labels = np.asarray([int(row['label_id']) for row in reference])
        patients = np.asarray([str(row['patient_id']) for row in reference])
        prediction = {'R0': np.asarray([np.argmax(row['paper_logits']) for row in reference])}
        for arm, rows in loaded.items():
            mapping = {str(row['graph_id']): row for row in rows}
            if set(mapping) != set(keys):
                raise RuntimeError('SICAP join mismatch')
            prediction[arm] = np.asarray([np.argmax(mapping[key]['final_logits']) for key in keys])
        unique = np.unique(patients)
        groups = {p: np.flatnonzero(patients == p) for p in unique}
        for comparison in comparisons:
            left, right = comparison.split('-')
            observed = _qwk(labels, prediction[left]) - _qwk(labels, prediction[right])
            values = []
            for _ in range(replicates):
                index = np.concatenate([groups[p] for p in rng.choice(unique, len(unique), replace=True)])
                values.append(_qwk(labels[index], prediction[left][index]) - _qwk(labels[index], prediction[right][index]))
            values = np.asarray(values)
            output.append({'comparison': comparison, 'delta': observed, 'ci_low': float(np.quantile(values, 0.025)), 'ci_high': float(np.quantile(values, 0.975)), 'p_one_sided': float((1 + (values <= 0).sum()) / (replicates + 1))})
    else:
        patient = {arm: _crc_patients(rows, 'final') for arm, rows in loaded.items()}
        patient['R0'] = _crc_patients(loaded[needed[0]], 'paper')
        keys = sorted(patient['R0'])
        if any((set(values) != set(keys) for values in patient.values())):
            raise RuntimeError('CRC patient join mismatch')
        labels = np.asarray([patient['R0'][key]['label'] for key in keys])
        folds = np.asarray([patient['R0'][key]['fold'] for key in keys])
        scores = {arm: np.asarray([values[key]['score'] for key in keys]) for arm, values in patient.items()}
        fold_groups = {f: np.flatnonzero(folds == f) for f in np.unique(folds)}
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
    root = Path(cfg['paths']['result_root']) / 'statistics' / f'seed_{seed}' / family
    atomic_parquet(root / 'paired_bootstrap.parquet', output)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'family': family, 'replicates': replicates, 'comparisons': output, 'validation_only': True, 'official_test_touched': False}
    atomic_json(root / 'summary.json', payload)
    return payload
