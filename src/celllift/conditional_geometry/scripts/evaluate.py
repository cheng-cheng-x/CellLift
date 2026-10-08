from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.conditional_geometry.protocol import PROTOCOL_ID, TARGET_COLUMNS

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _qwk(truth: np.ndarray, prediction: np.ndarray) -> float:
    classes = 4
    observed = np.zeros((classes, classes), np.float64)
    np.add.at(observed, (truth, prediction), 1)
    expected = np.outer(np.bincount(truth, minlength=classes), np.bincount(prediction, minlength=classes)) / len(truth)
    index = np.arange(classes)
    weight = np.square((index[:, None] - index[None, :]) / (classes - 1))
    return float(1 - (weight * observed).sum() / max((weight * expected).sum(), 1e-12))

def _auroc(truth: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    positive = truth == 1
    n1, n0 = (int(positive.sum()), int((~positive).sum()))
    if not n1 or not n0:
        return float('nan')
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def _metric(dataset: str, truth: np.ndarray, values: np.ndarray) -> float:
    return _qwk(truth, values.argmax(1)) if dataset == 'sicapv2' else _auroc(truth, values)

def _ensemble_predictions(root: Path, dataset: str, encoder: str, arm: str, seeds: Sequence[int], folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    by_seed: dict[int, dict[str, tuple[str, int, np.ndarray | float]]] = {}
    seed_fold_metrics: dict[str, float] = {}
    for seed in seeds:
        samples: dict[str, tuple[str, int, np.ndarray | float]] = {}
        for fold in range(folds):
            path = root / 'screen' / encoder / arm / f'seed_{seed}' / f'fold_{fold}' / 'validation_predictions.parquet'
            if not path.is_file():
                raise FileNotFoundError(path)
            fold_rows = _rows(path)
            truth = np.asarray([int(row['y_true']) for row in fold_rows])
            if dataset == 'sicapv2':
                value = np.asarray([[row[f'prob_{index}'] for index in range(4)] for row in fold_rows], np.float64)
                identifiers = [str(row['graph_id']) for row in fold_rows]
            else:
                value = np.asarray([row['score'] for row in fold_rows], np.float64)
                identifiers = [str(row['patient_id']) for row in fold_rows]
            seed_fold_metrics[f'seed_{seed}/fold_{fold}'] = _metric(dataset, truth, value)
            for index, row in enumerate(fold_rows):
                identifier = identifiers[index]
                if identifier in samples:
                    raise RuntimeError(f'duplicate OOF prediction {identifier}')
                samples[identifier] = (str(row['patient_id']), int(row['y_true']), value[index])
        by_seed[int(seed)] = samples
    identifiers = sorted(next(iter(by_seed.values())))
    if any((sorted(values) != identifiers for values in by_seed.values())):
        raise RuntimeError('seed prediction sets differ')
    patient = np.asarray([by_seed[seeds[0]][key][0] for key in identifiers], object)
    truth = np.asarray([by_seed[seeds[0]][key][1] for key in identifiers], np.int64)
    if any((any((values[key][1] != truth[index] for index, key in enumerate(identifiers))) for values in by_seed.values())):
        raise RuntimeError('labels differ across seeds')
    stack = np.stack([[by_seed[seed][key][2] for key in identifiers] for seed in seeds])
    return (patient, truth, stack.mean(0), seed_fold_metrics)

def _ensemble_crc_predictions_by_fold(root: Path, encoder: str, arm: str, seeds: Sequence[int], folds: int) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, float]]:
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seed_fold_metrics: dict[str, float] = {}
    for fold in range(folds):
        by_seed: dict[int, dict[str, tuple[int, float]]] = {}
        for seed in seeds:
            path = root / 'screen' / encoder / arm / f'seed_{seed}' / f'fold_{fold}' / 'validation_predictions.parquet'
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = _rows(path)
            identifiers = [str(row['patient_id']) for row in rows]
            truth = np.asarray([int(row['y_true']) for row in rows], np.int64)
            scores = np.asarray([float(row['score']) for row in rows], np.float64)
            if len(set(identifiers)) != len(identifiers):
                raise RuntimeError(f'duplicate patient OOF prediction in fold {fold}')
            by_seed[int(seed)] = {identifier: (int(truth[index]), float(scores[index])) for index, identifier in enumerate(identifiers)}
            seed_fold_metrics[f'seed_{seed}/fold_{fold}'] = _auroc(truth, scores)
        identifiers = sorted(by_seed[int(seeds[0])])
        if any((sorted(values) != identifiers for values in by_seed.values())):
            raise RuntimeError(f'seed patient sets differ in fold {fold}')
        truth = np.asarray([by_seed[int(seeds[0])][key][0] for key in identifiers], np.int64)
        if any((any((values[key][0] != truth[index] for index, key in enumerate(identifiers))) for values in by_seed.values())):
            raise RuntimeError(f'labels differ across seeds in fold {fold}')
        stack = np.stack([[by_seed[int(seed)][key][1] for key in identifiers] for seed in seeds])
        result.append((np.asarray(identifiers, object), truth, stack.mean(axis=0)))
    return (result, seed_fold_metrics)

def _paired_bootstrap(dataset: str, patient: np.ndarray, truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, replicates: int) -> dict[str, Any]:
    clusters = np.asarray(sorted(set(map(str, patient))), object)
    indices = {cluster: np.flatnonzero(patient == cluster) for cluster in clusters}
    rng = np.random.default_rng(20260822)
    deltas = np.empty(replicates, np.float64)
    for iteration in range(replicates):
        drawn = clusters[rng.integers(0, len(clusters), size=len(clusters))]
        selected = np.concatenate([indices[str(cluster)] for cluster in drawn])
        deltas[iteration] = _metric(dataset, truth[selected], candidate[selected]) - _metric(dataset, truth[selected], baseline[selected])
    finite = deltas[np.isfinite(deltas)]
    return {'delta': float(_metric(dataset, truth, candidate) - _metric(dataset, truth, baseline)), 'ci95': [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))], 'one_sided_p': float((1 + np.sum(finite <= 0)) / (1 + len(finite))), 'replicates': int(len(finite)), 'clusters': int(len(clusters)), 'positive_ci_lower_gt_zero': bool(np.quantile(finite, 0.025) > 0)}

def _paired_fold_stratified_crc_bootstrap(candidate_folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], baseline_folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], replicates: int) -> dict[str, Any]:
    if len(candidate_folds) != len(baseline_folds) or not candidate_folds:
        raise ValueError('candidate/baseline fold sets differ')
    prepared = []
    for candidate, baseline in zip(candidate_folds, baseline_folds):
        cp, cy, cs = candidate
        bp, by, bs = baseline
        if not np.array_equal(cp, bp) or not np.array_equal(cy, by):
            raise RuntimeError('candidate/baseline patients differ within fold')
        prepared.append((cp, cy, cs, bs))
    candidate_metric = float(np.mean([_auroc(y, c) for _, y, c, _ in prepared]))
    baseline_metric = float(np.mean([_auroc(y, b) for _, y, _, b in prepared]))
    rng = np.random.default_rng(20260822)
    deltas = np.empty(replicates, np.float64)
    for iteration in range(replicates):
        fold_deltas = []
        for patient, truth, candidate, baseline in prepared:
            clusters = np.asarray(sorted(set(map(str, patient))), object)
            indices = {cluster: np.flatnonzero(patient == cluster) for cluster in clusters}
            drawn = clusters[rng.integers(0, len(clusters), size=len(clusters))]
            selected = np.concatenate([indices[str(cluster)] for cluster in drawn])
            fold_deltas.append(_auroc(truth[selected], candidate[selected]) - _auroc(truth[selected], baseline[selected]))
        deltas[iteration] = np.mean(fold_deltas)
    finite = deltas[np.isfinite(deltas)]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {'delta': float(candidate_metric - baseline_metric), 'ci95': [float(lower), float(upper)], 'one_sided_p': float((1 + np.sum(finite <= 0)) / (1 + len(finite))), 'replicates': int(len(finite)), 'clusters': int(sum((len(set(map(str, patient))) for patient, _, _, _ in prepared))), 'folds': int(len(prepared)), 'positive_ci_lower_gt_zero': bool(lower > 0), 'aggregation': 'mean fold AUROC; patient bootstrap stratified within fold'}

def evaluate_validation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    folds = int(cfg['split']['validation_folds'])
    probe_folds = []
    for fold in range(folds):
        path = result_root / 'probe' / f'fold_{fold:02d}' / 'metrics.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS' or payload.get('official_test_touched') is not False:
            raise RuntimeError(f'probe fold gate failed: {path}')
        probe_folds.append(payload)
    probe_summary = {}
    for target in TARGET_COLUMNS:
        anchor = [row['validation_r2']['targets'][target]['r2_anchor_weighted'] for row in probe_folds]
        graph = [row['validation_r2']['targets'][target]['r2_graph_balanced'] for row in probe_folds]
        probe_summary[target] = {'r2_anchor_weighted_by_fold': anchor, 'r2_anchor_weighted_mean': float(np.mean(anchor)), 'r2_anchor_weighted_std': float(np.std(anchor, ddof=1)), 'r2_graph_balanced_by_fold': graph, 'r2_graph_balanced_mean': float(np.mean(graph)), 'all_folds_r2_gt_0_9': bool(np.all(np.asarray(anchor) > 0.9))}
    seeds = tuple(map(int, cfg['downstream']['seeds']))
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    downstream: dict[str, Any] = {}
    for encoder in cfg['downstream']['encoders']:
        if cfg['dataset'] == 'tcga_crc_msi':
            fold_ensemble = {}
            fold_metrics = {}
            for arm in ('conditional_geometry_2D', 'conditional_geometry_RES3D', 'conditional_geometry_SHUF_RES3D'):
                values, metrics = _ensemble_crc_predictions_by_fold(result_root, encoder, arm, seeds, folds)
                fold_ensemble[arm] = values
                fold_metrics[arm] = metrics
            ensemble_metrics = {arm: float(np.mean([_auroc(truth, score) for _, truth, score in values])) for arm, values in fold_ensemble.items()}
            downstream[encoder] = {'metric': 'mean_fold_patient_AUROC', 'ensemble_metrics': ensemble_metrics, 'seed_fold_metrics': fold_metrics, 'residual_vs_2d': _paired_fold_stratified_crc_bootstrap(fold_ensemble['conditional_geometry_RES3D'], fold_ensemble['conditional_geometry_2D'], replicates), 'residual_vs_shuffled': _paired_fold_stratified_crc_bootstrap(fold_ensemble['conditional_geometry_RES3D'], fold_ensemble['conditional_geometry_SHUF_RES3D'], replicates)}
            continue
        ensemble = {}
        fold_metrics = {}
        reference_patient = reference_truth = None
        for arm in ('conditional_geometry_2D', 'conditional_geometry_RES3D', 'conditional_geometry_SHUF_RES3D'):
            patient, truth, values, metrics = _ensemble_predictions(result_root, cfg['dataset'], encoder, arm, seeds, folds)
            if reference_patient is not None and (not np.array_equal(patient, reference_patient) or not np.array_equal(truth, reference_truth)):
                raise RuntimeError('arm OOF samples differ')
            reference_patient, reference_truth = (patient, truth)
            ensemble[arm] = values
            fold_metrics[arm] = metrics
        downstream[encoder] = {'metric': 'QWK' if cfg['dataset'] == 'sicapv2' else 'patient_AUROC', 'ensemble_metrics': {arm: _metric(cfg['dataset'], reference_truth, values) for arm, values in ensemble.items()}, 'seed_fold_metrics': fold_metrics, 'residual_vs_2d': _paired_bootstrap(cfg['dataset'], reference_patient, reference_truth, ensemble['conditional_geometry_RES3D'], ensemble['conditional_geometry_2D'], replicates), 'residual_vs_shuffled': _paired_bootstrap(cfg['dataset'], reference_patient, reference_truth, ensemble['conditional_geometry_RES3D'], ensemble['conditional_geometry_SHUF_RES3D'], replicates)}
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'official_test_touched': False, 'probe': probe_summary, 'downstream': downstream}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
