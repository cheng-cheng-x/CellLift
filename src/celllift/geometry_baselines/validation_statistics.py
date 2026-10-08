from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
import math
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .io_utils import atomic_json, atomic_parquet
from .protocol import ARMS, ENCODERS, PROTOCOL_ID
from .statistics import holm_adjust
PAIR_FAMILIES: dict[str, tuple[tuple[str, str], ...]] = {'primary_residual': (('R5', 'R1'), ('R5', 'R5S'), ('R4', 'R0'), ('R4', 'R4S')), 'direct_3d_vs_2d': (('R4', 'R1'), ('R2', 'R1'), ('O4', 'O1'), ('O2', 'O1')), 'raw_3d': (('R3', 'R1'), ('R3', 'R3S'), ('R2', 'R0'), ('R2', 'R2S')), 'mask2d_usefulness': (('R1', 'R0'), ('R1', 'R1S'), ('O1', 'O1S')), 'two_d_given_3d': (('R5', 'R4'), ('R3', 'R2')), 'geometry_real_vs_shuffle': (('O1', 'O1S'), ('O2', 'O2S'), ('O3', 'O3S'), ('O4', 'O4S'), ('O5', 'O5S'), ('R1', 'R1S'), ('R2', 'R2S'), ('R3', 'R3S'), ('R4', 'R4S'), ('R5', 'R5S'))}
INTERACTIONS = {'I_res': (('R5', 1.0), ('R1', -1.0), ('R4', -1.0), ('R0', 1.0)), 'I_raw': (('R3', 1.0), ('R1', -1.0), ('R2', -1.0), ('R0', 1.0))}

def _qwk_from_confusion(confusion: np.ndarray) -> np.ndarray:
    confusion = np.asarray(confusion, np.float64)
    if confusion.ndim == 2:
        confusion = confusion[None]
    classes = confusion.shape[-1]
    indices = np.arange(classes, dtype=np.float64)
    penalty = ((indices[:, None] - indices[None, :]) / max(1, classes - 1)) ** 2
    row = confusion.sum(2)
    column = confusion.sum(1)
    total = confusion.sum((1, 2))
    expected = row[:, :, None] * column[:, None, :] / total[:, None, None].clip(min=1.0)
    numerator = (confusion * penalty).sum((1, 2))
    denominator = (expected * penalty).sum((1, 2))
    return 1.0 - numerator / np.clip(denominator, 1e-15, None)

def _sicap_distributions(labels: np.ndarray, patients: np.ndarray, scores_by_arm: Mapping[str, np.ndarray], replicates: int, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    unique = np.unique(patients)
    patient_index = {patient: index for index, patient in enumerate(unique)}
    row_patient = np.asarray([patient_index[value] for value in patients], np.int64)
    rng = np.random.default_rng(seed)
    bootstrap_weight = rng.multinomial(len(unique), np.full(len(unique), 1.0 / len(unique)), size=replicates)
    observed, distributions = ({}, {})
    for arm, probability in scores_by_arm.items():
        predicted = probability.argmax(1)
        contribution = np.zeros((len(unique), 4, 4), np.int64)
        np.add.at(contribution, (row_patient, labels, predicted), 1)
        observed[arm] = float(_qwk_from_confusion(contribution.sum(0))[0])
        values = np.empty(replicates, np.float64)
        for start in range(0, replicates, 1000):
            stop = min(replicates, start + 1000)
            confusion = np.einsum('rp,pij->rij', bootstrap_weight[start:stop], contribution, optimize=True)
            values[start:stop] = _qwk_from_confusion(confusion)
        distributions[arm] = values
    return (observed, distributions)

def _weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, np.float64)
    if weights.ndim == 1:
        weights = weights[None]
    order = np.argsort(scores, kind='stable')
    labels, scores, weights = (labels[order], scores[order], weights[:, order])
    numerator = np.zeros(len(weights), np.float64)
    cumulative_negative = np.zeros(len(weights), np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        group = weights[:, start:stop]
        group_labels = labels[start:stop]
        positive = group[:, group_labels == 1].sum(1)
        negative = group[:, group_labels == 0].sum(1)
        numerator += positive * (cumulative_negative + 0.5 * negative)
        cumulative_negative += negative
        start = stop
    positive_total = weights[:, labels == 1].sum(1)
    negative_total = weights[:, labels == 0].sum(1)
    denominator = positive_total * negative_total
    result = numerator / np.where(denominator > 0, denominator, np.nan)
    return result

def _crc_distributions(labels: np.ndarray, folds: np.ndarray, scores_by_arm: Mapping[str, np.ndarray], replicates: int, seed: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    fold_values = sorted(set(map(int, folds)))
    fold_weights: dict[int, np.ndarray] = {}
    for fold in fold_values:
        count = int((folds == fold).sum())
        fold_weights[fold] = rng.multinomial(count, np.full(count, 1.0 / count), size=replicates)
    observed, distributions = ({}, {})
    for arm, scores in scores_by_arm.items():
        observed_fold, bootstrap_fold = ([], [])
        for fold in fold_values:
            selected = folds == fold
            observed_fold.append(float(_weighted_auc(labels[selected], scores[selected], np.ones(selected.sum()))[0]))
            bootstrap_fold.append(_weighted_auc(labels[selected], scores[selected], fold_weights[fold]))
        observed[arm] = float(np.mean(observed_fold))
        distributions[arm] = np.nanmean(np.stack(bootstrap_fold), axis=0)
    return (observed, distributions)

def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    output = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        output[start:stop] = 0.5 * (start + stop - 1) + 1
        start = stop
    result = np.empty(len(values), np.float64)
    result[order] = output
    return result

def _delong_fold(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-labels)
    truth = labels[order]
    predictions = np.stack((first[order], second[order]))
    positives = int(truth.sum())
    negatives = len(truth) - positives
    tx = np.stack([_midrank(row[:positives]) for row in predictions])
    ty = np.stack([_midrank(row[positives:]) for row in predictions])
    tz = np.stack([_midrank(row) for row in predictions])
    auc = tz[:, :positives].sum(1) / (positives * negatives) - (positives + 1) / (2 * negatives)
    auxiliary_938db8 = (tz[:, :positives] - tx) / negatives
    shape_warmstart = 1.0 - (tz[:, positives:] - ty) / positives
    covariance = np.cov(auxiliary_938db8) / positives + np.cov(shape_warmstart) / negatives
    variance = float(np.array([1.0, -1.0]) @ covariance @ np.array([1.0, -1.0]))
    z = float((auc[0] - auc[1]) / math.sqrt(max(variance, 1e-15)))
    try:
        from scipy.stats import norm
        p = float(norm.sf(z))
    except ImportError:
        p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return (z, p)

def _fold_stratified_delong(labels: np.ndarray, folds: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    values = {}
    z_values = []
    for fold in sorted(set(map(int, folds))):
        selected = folds == fold
        z, p = _delong_fold(labels[selected], first[selected], second[selected])
        values[str(fold)] = {'z': z, 'p_one_sided': p}
        z_values.append(z)
    combined_z = float(np.sum(z_values) / math.sqrt(len(z_values)))
    try:
        from scipy.stats import norm
        combined_p = float(norm.sf(combined_z))
    except ImportError:
        combined_p = 0.5 * math.erfc(combined_z / math.sqrt(2.0))
    return {'folds': values, 'stouffer_z': combined_z, 'stouffer_p_one_sided': combined_p}

def _result_from_distribution(observed: float, values: np.ndarray) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    if len(valid) < 0.95 * len(values):
        raise RuntimeError('too many invalid registered bootstrap replicates')
    return {'delta': float(observed), 'ci_low': float(np.quantile(valid, 0.025)), 'ci_high': float(np.quantile(valid, 0.975)), 'p_one_sided': float((1 + np.sum(valid <= 0)) / (1 + len(valid))), 'bootstrap_replicates': int(len(valid))}

def compute_validation_statistics(cfg: Mapping[str, Any], prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    replicates = int(cfg.get('runtime', {}).get('bootstrap_replicates', 10000))
    output_rows: list[dict[str, Any]] = []
    coverage = {}
    paper = [row for row in prediction_rows if row['arm_id'] == 'R0' and row['encoder'] == 'paper']
    for encoder in ENCODERS:
        by_arm: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
        for arm in ARMS:
            selected = paper if arm == 'R0' else [row for row in prediction_rows if row['encoder'] == encoder and row['arm_id'] == arm]
            key_name = 'graph_id' if dataset == 'sicapv2' else 'patient_id'
            by_arm[arm] = {(int(row['fold']), str(row[key_name])): row for row in selected}
        common_keys = set.intersection(*(set(values) for values in by_arm.values()))
        if not common_keys:
            raise RuntimeError(f'empty 21-arm comparison universe for {dataset}/{encoder}')
        coverage[encoder] = {'common_samples': len(common_keys), 'per_arm': {arm: len(values) for arm, values in by_arm.items()}}
        keys = sorted(common_keys)
        reference = [by_arm['R0'][key] for key in keys]
        labels = np.asarray([int(row['label_id']) for row in reference], np.int64)
        folds = np.asarray([int(row['fold']) for row in reference], np.int64)
        patients = np.asarray([str(row['patient_id']) for row in reference], object)
        if dataset == 'sicapv2':
            scores = {arm: np.stack([np.asarray(by_arm[arm][key]['probabilities'], float) for key in keys]) for arm in ARMS}
            observed, distributions = _sicap_distributions(labels, patients, scores, replicates, 20260831)
        else:
            scores = {arm: np.asarray([float(by_arm[arm][key]['score']) for key in keys]) for arm in ARMS}
            observed, distributions = _crc_distributions(labels, folds, scores, replicates, 20260831)
        for family, comparisons in PAIR_FAMILIES.items():
            family_rows = []
            for first, second in comparisons:
                name = f'{first}-{second}'
                result = _result_from_distribution(observed[first] - observed[second], distributions[first] - distributions[second])
                row = {'encoder': encoder, 'family': family, 'comparison': name, **result}
                if dataset == 'tcga_crc_msi':
                    row['delong'] = _fold_stratified_delong(labels, folds, scores[first], scores[second])
                family_rows.append(row)
            adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in family_rows})
            for row in family_rows:
                row['holm_p'] = adjusted[row['comparison']]
                row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
                output_rows.append(row)
        interaction_rows = []
        for name, terms in INTERACTIONS.items():
            point = sum((weight * observed[arm] for arm, weight in terms))
            distribution = sum((weight * distributions[arm] for arm, weight in terms))
            interaction_rows.append({'encoder': encoder, 'family': 'interaction', 'comparison': name, **_result_from_distribution(point, distribution)})
        adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in interaction_rows})
        for row in interaction_rows:
            row['holm_p'] = adjusted[row['comparison']]
            row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
            output_rows.append(row)
    root = Path(cfg['paths']['result_root']) / 'statistics' / 'validation'
    parquet_rows = []
    for row in output_rows:
        flat = {key: value for key, value in row.items() if key != 'delong'}
        if 'delong' in row:
            flat['delong_stouffer_z'] = row['delong']['stouffer_z']
            flat['delong_stouffer_p_one_sided'] = row['delong']['stouffer_p_one_sided']
        parquet_rows.append(flat)
    atomic_parquet(root / 'comparisons.parquet', parquet_rows)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'metric': 'QWK' if dataset == 'sicapv2' else 'mean_fold_patient_AUROC', 'bootstrap': 'patient_cluster' if dataset == 'sicapv2' else 'fold_stratified_patient', 'replicates': replicates, 'coverage': coverage, 'comparisons': output_rows}
    atomic_json(root / 'summary.json', payload)
    return payload

def compute_crc_tile_validation_statistics(cfg: Mapping[str, Any], prediction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if str(cfg['dataset']) != 'tcga_crc_msi':
        raise RuntimeError('tile validation statistics are CRC-only')
    replicates = int(cfg.get('runtime', {}).get('bootstrap_replicates', 10000))
    available = ('R0',) + tuple((arm for arm in ARMS if arm.startswith('R') and arm != 'R0'))
    paper = [row for row in prediction_rows if row['encoder'] == 'paper' and row['arm_id'] == 'R0']
    output_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    for encoder in ENCODERS:
        by_arm = {}
        for arm in available:
            selected = paper if arm == 'R0' else [row for row in prediction_rows if row['encoder'] == encoder and row['arm_id'] == arm]
            by_arm[arm] = {(int(row['fold']), str(row['patient_id'])): row for row in selected}
        keys = sorted(set.intersection(*(set(value) for value in by_arm.values())))
        if not keys:
            raise RuntimeError(f'empty tile-fusion comparison universe: {encoder}')
        coverage[encoder] = {'common_patients': len(keys), 'per_arm': {arm: len(values) for arm, values in by_arm.items()}}
        labels = np.asarray([int(by_arm['R0'][key]['label_id']) for key in keys], np.int64)
        folds = np.asarray([key[0] for key in keys], np.int64)
        scores = {arm: np.asarray([float(by_arm[arm][key]['score']) for key in keys]) for arm in available}
        observed, distributions = _crc_distributions(labels, folds, scores, replicates, 20260831)
        for family, comparisons in PAIR_FAMILIES.items():
            registered = [(first, second) for first, second in comparisons if first in scores and second in scores]
            if not registered:
                continue
            family_rows = []
            for first, second in registered:
                row = {'encoder': encoder, 'family': family, 'comparison': f'{first}-{second}', **_result_from_distribution(observed[first] - observed[second], distributions[first] - distributions[second]), 'delong': _fold_stratified_delong(labels, folds, scores[first], scores[second])}
                family_rows.append(row)
            adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in family_rows})
            for row in family_rows:
                row['holm_p'] = adjusted[row['comparison']]
                row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
                output_rows.append(row)
        interaction_rows = []
        for name, terms in INTERACTIONS.items():
            if not all((arm in scores for arm, _ in terms)):
                continue
            interaction_rows.append({'encoder': encoder, 'family': 'interaction', 'comparison': name, **_result_from_distribution(sum((weight * observed[arm] for arm, weight in terms)), sum((weight * distributions[arm] for arm, weight in terms)))})
        adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in interaction_rows})
        for row in interaction_rows:
            row['holm_p'] = adjusted[row['comparison']]
            row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
            output_rows.append(row)
    root = Path(cfg['paths']['result_root']) / 'statistics' / 'validation_tile'
    parquet_rows = []
    for row in output_rows:
        flat = {key: value for key, value in row.items() if key != 'delong'}
        if 'delong' in row:
            flat['delong_stouffer_z'] = row['delong']['stouffer_z']
            flat['delong_stouffer_p_one_sided'] = row['delong']['stouffer_p_one_sided']
        parquet_rows.append(flat)
    atomic_parquet(root / 'comparisons.parquet', parquet_rows)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': 'tcga_crc_msi', 'route': 'tile_temperature_scalar_then_patient_hard_vote', 'metric': 'mean_fold_patient_AUROC', 'bootstrap': 'fold_stratified_patient', 'replicates': replicates, 'coverage': coverage, 'comparisons': output_rows}
    atomic_json(root / 'summary.json', payload)
    return payload
