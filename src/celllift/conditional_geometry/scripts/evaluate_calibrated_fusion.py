from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.conditional_geometry.calibrated_fusion_protocol import ABLATION_ARM, ALPHA_BOUNDS, BASE_ARM, BOOTSTRAP_SEED, EPSILON, GEOMETRY_ARMS, OFFICIAL_TEST_ALLOWED, OPTIMIZER_MAX_ITERATIONS, OPTIMIZER_TOLERANCE, PRIMARY_COMPARISONS, PRIMARY_ENCODER, PROTOCOL_ID, SECONDARY_ENCODERS, TEMPERATURE_BOUNDS
from celllift.conditional_geometry.late_fusion_protocol import GEOMETRY_PROTOCOL_ID, RGB_MASK_PROTOCOL_ID
from celllift.conditional_geometry.scripts.evaluate import _auroc, _metric
from celllift.conditional_geometry.scripts.evaluate_late_fusion import PredictionBatch, _aligned, _atomic_json, _atomic_parquet, _holm_adjust, _prediction_rows, _read_source

@dataclass(frozen=True)
class FittedGate:
    base_temperature: float
    geometry_temperature: float
    alpha: np.ndarray
    base_temperature_success: bool
    geometry_temperature_success: bool
    alpha_success: bool
    base_temperature_nll: float
    geometry_temperature_nll: float
    fused_nll: float

def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, np.float64), EPSILON, 1.0 - EPSILON)

def probabilities_to_logits(dataset: str, values: np.ndarray) -> np.ndarray:
    values = _clip_probability(values)
    if dataset == 'sicapv2':
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError('SICAP fusion requires N x 4 probabilities')
        log_values = np.log(values)
        return log_values - log_values.mean(axis=1, keepdims=True)
    if dataset == 'tcga_crc_msi':
        if values.ndim != 1:
            raise ValueError('CRC fusion requires one probability per patient')
        return np.log(values) - np.log1p(-values)
    raise ValueError(f'unsupported dataset: {dataset}')

def logits_to_probabilities(dataset: str, logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, np.float64)
    if dataset == 'sicapv2':
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)
    if dataset == 'tcga_crc_msi':
        output = np.empty_like(logits)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        output[~positive] = exponent / (1.0 + exponent)
        return output
    raise ValueError(f'unsupported dataset: {dataset}')

def _nll(dataset: str, truth: np.ndarray, logits: np.ndarray) -> float:
    truth = np.asarray(truth, np.int64)
    probabilities = _clip_probability(logits_to_probabilities(dataset, logits))
    if dataset == 'sicapv2':
        return float(-np.log(probabilities[np.arange(len(truth)), truth]).mean())
    return float(-(truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities)).mean())

def fit_temperature(dataset: str, truth: np.ndarray, logits: np.ndarray) -> tuple[float, bool, float]:
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda temperature: _nll(dataset, truth, logits / float(temperature)), bounds=TEMPERATURE_BOUNDS, method='bounded', options={'xatol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    temperature = float(result.x)
    if not np.isfinite(temperature) or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]:
        raise RuntimeError('temperature optimizer returned an invalid value')
    return (temperature, bool(result.success), float(result.fun))

def fit_nonnegative_gate(dataset: str, truth: np.ndarray, base_logits: np.ndarray, geometry_logits: np.ndarray) -> tuple[np.ndarray, bool, float]:
    from scipy.optimize import minimize
    dimensions = 4 if dataset == 'sicapv2' else 1

    def objective(alpha: np.ndarray) -> float:
        if dataset == 'sicapv2':
            fused = base_logits + geometry_logits * alpha.reshape(1, 4)
        else:
            fused = base_logits + geometry_logits * float(alpha[0])
        return _nll(dataset, truth, fused)
    result = minimize(objective, x0=np.full(dimensions, 0.25, np.float64), method='L-BFGS-B', bounds=[ALPHA_BOUNDS] * dimensions, options={'ftol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    alpha = np.asarray(result.x, np.float64)
    if alpha.shape != (dimensions,) or not np.all(np.isfinite(alpha)) or np.any(alpha < ALPHA_BOUNDS[0]) or np.any(alpha > ALPHA_BOUNDS[1]):
        raise RuntimeError('gate optimizer returned invalid geometry weights')
    return (alpha, bool(result.success), float(result.fun))

def fit_gate(dataset: str, truth: np.ndarray, base_values: np.ndarray, geometry_values: np.ndarray, *, calibrate_temperatures: bool) -> FittedGate:
    base_logits = probabilities_to_logits(dataset, base_values)
    geometry_logits = probabilities_to_logits(dataset, geometry_values)
    if calibrate_temperatures:
        base_temperature, base_success, base_nll = fit_temperature(dataset, truth, base_logits)
        geometry_temperature, geometry_success, geometry_nll = fit_temperature(dataset, truth, geometry_logits)
    else:
        base_temperature = geometry_temperature = 1.0
        base_success = geometry_success = True
        base_nll = _nll(dataset, truth, base_logits)
        geometry_nll = _nll(dataset, truth, geometry_logits)
    alpha, alpha_success, fused_nll = fit_nonnegative_gate(dataset, truth, base_logits / base_temperature, geometry_logits / geometry_temperature)
    return FittedGate(base_temperature=base_temperature, geometry_temperature=geometry_temperature, alpha=alpha, base_temperature_success=base_success, geometry_temperature_success=geometry_success, alpha_success=alpha_success, base_temperature_nll=base_nll, geometry_temperature_nll=geometry_nll, fused_nll=fused_nll)

def apply_gate(dataset: str, base_values: np.ndarray, geometry_values: np.ndarray, gate: FittedGate) -> np.ndarray:
    base_logits = probabilities_to_logits(dataset, base_values) / gate.base_temperature
    geometry_logits = probabilities_to_logits(dataset, geometry_values) / gate.geometry_temperature
    if dataset == 'sicapv2':
        fused = base_logits + geometry_logits * gate.alpha.reshape(1, 4)
    else:
        fused = base_logits + geometry_logits * float(gate.alpha[0])
    output = logits_to_probabilities(dataset, fused)
    if not np.all(np.isfinite(output)) or np.any((output < 0) | (output > 1)):
        raise RuntimeError('calibrated fusion produced invalid probabilities')
    return output

def _gate_payload(gate: FittedGate) -> dict[str, Any]:
    alpha = [float(value) for value in gate.alpha]
    tolerance = 1e-06
    return {'base_temperature': gate.base_temperature, 'geometry_temperature': gate.geometry_temperature, 'alpha': alpha, 'optimizer_success': {'base_temperature': gate.base_temperature_success, 'geometry_temperature': gate.geometry_temperature_success, 'alpha': gate.alpha_success}, 'fit_nll': {'base': gate.base_temperature_nll, 'geometry': gate.geometry_temperature_nll, 'fused': gate.fused_nll}, 'boundary_hits': {'base_temperature': bool(abs(gate.base_temperature - TEMPERATURE_BOUNDS[0]) <= tolerance or abs(gate.base_temperature - TEMPERATURE_BOUNDS[1]) <= tolerance), 'geometry_temperature': bool(abs(gate.geometry_temperature - TEMPERATURE_BOUNDS[0]) <= tolerance or abs(gate.geometry_temperature - TEMPERATURE_BOUNDS[1]) <= tolerance), 'alpha': [bool(abs(value - ALPHA_BOUNDS[0]) <= tolerance or abs(value - ALPHA_BOUNDS[1]) <= tolerance) for value in alpha]}}

def validate_patient_disjoint(fold_batches: Sequence[PredictionBatch]) -> None:
    patient_sets = [set(map(str, batch.patients)) for batch in fold_batches]
    for left in range(len(patient_sets)):
        for right in range(left + 1, len(patient_sets)):
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise RuntimeError(f'patient leakage between folds {left} and {right}: {sorted(overlap)[:3]}')

def _mean_seed_batch(batches: Sequence[PredictionBatch]) -> PredictionBatch:
    if not batches:
        raise ValueError('cannot ensemble an empty seed set')
    reference = batches[0]
    for batch in batches[1:]:
        _aligned(reference, batch)
    values = np.stack([batch.values for batch in batches], axis=0).mean(axis=0)
    return PredictionBatch(reference.identifiers.copy(), reference.patients.copy(), reference.truth.copy(), values)

def _concat_batches(batches: Sequence[PredictionBatch]) -> PredictionBatch:
    if not batches:
        raise ValueError('cannot concatenate an empty fold set')
    identifiers = np.concatenate([batch.identifiers for batch in batches])
    if len(set(map(str, identifiers))) != len(identifiers):
        raise RuntimeError('duplicate sample identifiers across OOF folds')
    return PredictionBatch(identifiers=identifiers, patients=np.concatenate([batch.patients for batch in batches]), truth=np.concatenate([batch.truth for batch in batches]), values=np.concatenate([batch.values for batch in batches], axis=0))

def _paired_bootstrap(dataset: str, patient: np.ndarray, truth: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, replicates: int) -> dict[str, Any]:
    if dataset != 'sicapv2':
        raise ValueError('the cluster-confusion bootstrap is specific to SICAP')
    clusters = np.asarray(sorted(set(map(str, patient))), object)
    indices = {cluster: np.flatnonzero(patient == cluster) for cluster in clusters}
    classes = 4
    candidate_confusion = np.zeros((len(clusters), classes, classes), np.int64)
    baseline_confusion = np.zeros_like(candidate_confusion)
    candidate_class = np.asarray(candidate).argmax(axis=1)
    baseline_class = np.asarray(baseline).argmax(axis=1)
    for cluster_index, cluster in enumerate(clusters):
        selected = indices[str(cluster)]
        np.add.at(candidate_confusion[cluster_index], (truth[selected], candidate_class[selected]), 1)
        np.add.at(baseline_confusion[cluster_index], (truth[selected], baseline_class[selected]), 1)
    weight_index = np.arange(classes, dtype=np.float64)
    weights = np.square((weight_index[:, None] - weight_index[None, :]) / (classes - 1))

    def qwk_from_confusion(confusion: np.ndarray) -> np.ndarray:
        row = confusion.sum(axis=2)
        column = confusion.sum(axis=1)
        total = confusion.sum(axis=(1, 2))
        expected = row[:, :, None] * column[:, None, :] / total[:, None, None]
        numerator = (confusion * weights[None]).sum(axis=(1, 2))
        denominator = (expected * weights[None]).sum(axis=(1, 2))
        return 1.0 - numerator / np.maximum(denominator, 1e-12)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    deltas = np.empty(replicates, np.float64)
    for start in range(0, replicates, 500):
        stop = min(start + 500, replicates)
        candidate_draw = candidate_confusion[draws[start:stop]].sum(axis=1)
        baseline_draw = baseline_confusion[draws[start:stop]].sum(axis=1)
        deltas[start:stop] = qwk_from_confusion(candidate_draw) - qwk_from_confusion(baseline_draw)
    finite = deltas[np.isfinite(deltas)]
    lower, upper = np.quantile(finite, [0.025, 0.975])
    return {'delta': float(_metric(dataset, truth, candidate) - _metric(dataset, truth, baseline)), 'ci95': [float(lower), float(upper)], 'one_sided_p': float((1 + np.sum(finite <= 0)) / (1 + len(finite))), 'replicates': int(len(finite)), 'clusters': int(len(clusters)), 'positive_ci_lower_gt_zero': bool(lower > 0)}

def _paired_crc_bootstrap(candidate_folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], baseline_folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], replicates: int) -> dict[str, Any]:
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
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
    return {'delta': float(candidate_metric - baseline_metric), 'ci95': [float(lower), float(upper)], 'one_sided_p': float((1 + np.sum(finite <= 0)) / (1 + len(finite))), 'replicates': int(len(finite)), 'clusters': int(sum((len(set(map(str, row[0]))) for row in prepared))), 'folds': int(len(prepared)), 'positive_ci_lower_gt_zero': bool(lower > 0), 'aggregation': 'mean fold AUROC; patient bootstrap stratified within fold'}

def _config_gate(cfg: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...], int, int]:
    if cfg.get('protocol_id') != PROTOCOL_ID or OFFICIAL_TEST_ALLOWED:
        raise RuntimeError('calibrated fusion protocol is not locked')
    fusion = cfg['fusion']
    seeds = tuple(map(int, fusion['seeds']))
    encoders = tuple(map(str, fusion['encoders']))
    if not seeds or encoders != (PRIMARY_ENCODER,) + SECONDARY_ENCODERS:
        raise ValueError('calibrated_fusion encoder order must be meanpool primary, deepsets secondary')
    if tuple(map(float, fusion['temperature_bounds'])) != TEMPERATURE_BOUNDS:
        raise ValueError('temperature bounds differ from the registered protocol')
    if tuple(map(float, fusion['alpha_bounds'])) != ALPHA_BOUNDS:
        raise ValueError('alpha bounds differ from the registered protocol')
    folds = int(cfg['split']['validation_folds'])
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    if folds < 3 or replicates != 10000:
        raise ValueError('calibrated_fusion requires at least three folds and exactly 10,000 bootstraps')
    return (seeds, encoders, folds, replicates)

def evaluate_calibrated_fusion(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    seeds, encoders, folds, replicates = _config_gate(cfg)
    base_root = Path(cfg['paths']['rgb_mask_result_root'])
    geometry_root = Path(cfg['paths']['geometry_result_root'])
    result_root = Path(cfg['paths']['result_root'])
    all_results: dict[str, Any] = {}
    source_provenance: dict[str, Any] = {}
    for encoder in encoders:
        fold_base: list[PredictionBatch] = []
        fold_geometry: dict[str, list[PredictionBatch]] = {arm_id: [] for arm_id in GEOMETRY_ARMS}
        for fold in range(folds):
            seed_base = []
            seed_geometry = {arm_id: [] for arm_id in GEOMETRY_ARMS}
            for seed in seeds:
                base, base_provenance = _read_source(base_root, dataset, encoder, BASE_ARM, seed, fold, protocol_id=RGB_MASK_PROTOCOL_ID, use_rgb=True)
                seed_base.append(base)
                provenance_key = f'{encoder}/seed_{seed}/fold_{fold}'
                source_provenance.setdefault(provenance_key, {})['rgb_mask'] = base_provenance
                for arm_id, source_arm in GEOMETRY_ARMS.items():
                    geometry, geometry_provenance = _read_source(geometry_root, dataset, encoder, source_arm, seed, fold, protocol_id=GEOMETRY_PROTOCOL_ID, use_rgb=False)
                    _aligned(base, geometry)
                    seed_geometry[arm_id].append(geometry)
                    source_provenance[provenance_key][arm_id] = geometry_provenance
            fold_base.append(_mean_seed_batch(seed_base))
            for arm_id in GEOMETRY_ARMS:
                fold_geometry[arm_id].append(_mean_seed_batch(seed_geometry[arm_id]))
        validate_patient_disjoint(fold_base)
        for arm_id in GEOMETRY_ARMS:
            for fold in range(folds):
                _aligned(fold_base[fold], fold_geometry[arm_id][fold])
        fold_predictions: dict[str, list[np.ndarray]] = {'F0_RGB_MASK': [batch.values.copy() for batch in fold_base], **{arm_id: [] for arm_id in GEOMETRY_ARMS}, ABLATION_ARM: []}
        fold_parameters: dict[str, Any] = {}
        fold_metrics: dict[str, dict[str, float]] = {arm: {} for arm in fold_predictions}
        for held_out in range(folds):
            fit_folds = [fold for fold in range(folds) if fold != held_out]
            train_base = _concat_batches([fold_base[fold] for fold in fit_folds])
            target_base = fold_base[held_out]
            if set(map(str, train_base.patients)) & set(map(str, target_base.patients)):
                raise RuntimeError('held-out patient entered fusion calibration')
            fold_parameters[f'fold_{held_out}'] = {'held_out_fold': held_out, 'fit_folds': fit_folds, 'fit_samples': len(train_base.truth), 'fit_patients': len(set(map(str, train_base.patients))), 'held_out_samples': len(target_base.truth), 'held_out_patients': len(set(map(str, target_base.patients))), 'arms': {}}
            for arm_id in GEOMETRY_ARMS:
                train_geometry = _concat_batches([fold_geometry[arm_id][fold] for fold in fit_folds])
                target_geometry = fold_geometry[arm_id][held_out]
                _aligned(train_base, train_geometry)
                gate = fit_gate(dataset, train_base.truth, train_base.values, train_geometry.values, calibrate_temperatures=True)
                fused = apply_gate(dataset, target_base.values, target_geometry.values, gate)
                fold_predictions[arm_id].append(fused)
                fold_parameters[f'fold_{held_out}']['arms'][arm_id] = _gate_payload(gate)
            train_real = _concat_batches([fold_geometry['F2_REAL3D'][fold] for fold in fit_folds])
            target_real = fold_geometry['F2_REAL3D'][held_out]
            ablation_gate = fit_gate(dataset, train_base.truth, train_base.values, train_real.values, calibrate_temperatures=False)
            ablation_prediction = apply_gate(dataset, target_base.values, target_real.values, ablation_gate)
            fold_predictions[ABLATION_ARM].append(ablation_prediction)
            fold_parameters[f'fold_{held_out}']['arms'][ABLATION_ARM] = _gate_payload(ablation_gate)
            for arm_id, values_by_fold in fold_predictions.items():
                values = values_by_fold[held_out]
                fold_metrics[arm_id][f'fold_{held_out}'] = _metric(dataset, target_base.truth, values)
                prediction_path = result_root / 'predictions' / encoder / arm_id / f'fold_{held_out}.parquet'
                checksum = _atomic_parquet(prediction_path, _prediction_rows(dataset, target_base, values))
                fold_parameters[f'fold_{held_out}'].setdefault('prediction_sha256', {})[arm_id] = checksum
        if dataset == 'sicapv2':
            combined_base = _concat_batches(fold_base)
            combined_predictions = {arm: np.concatenate(values_by_fold, axis=0) for arm, values_by_fold in fold_predictions.items()}
            metrics = {arm: _metric(dataset, combined_base.truth, values) for arm, values in combined_predictions.items()}

            def compare(candidate: str, baseline: str) -> dict[str, Any]:
                return _paired_bootstrap(dataset, combined_base.patients, combined_base.truth, combined_predictions[candidate], combined_predictions[baseline], replicates)
        else:
            crc_predictions = {arm: [(fold_base[fold].patients, fold_base[fold].truth, values_by_fold[fold]) for fold in range(folds)] for arm, values_by_fold in fold_predictions.items()}
            metrics = {arm: float(np.mean([_auroc(truth, score) for _, truth, score in rows])) for arm, rows in crc_predictions.items()}

            def compare(candidate: str, baseline: str) -> dict[str, Any]:
                return _paired_crc_bootstrap(crc_predictions[candidate], crc_predictions[baseline], replicates)
        comparisons = {'real3d_vs_2d': compare('F2_REAL3D', 'F1_2D'), 'real3d_vs_shuffled3d': compare('F2_REAL3D', 'F3_SHUF3D'), 'real3d_vs_rgb_mask_base': compare('F2_REAL3D', 'F0_RGB_MASK'), 'real3d_vs_uncalibrated_gate': compare('F2_REAL3D', ABLATION_ARM), 'geometry2d_vs_rgb_mask_base': compare('F1_2D', 'F0_RGB_MASK')}
        adjusted = _holm_adjust({key: comparisons[key]['one_sided_p'] for key in PRIMARY_COMPARISONS})
        for key in PRIMARY_COMPARISONS:
            comparisons[key]['holm_adjusted_one_sided_p'] = adjusted[key]
            comparisons[key]['holm_positive'] = bool(comparisons[key]['positive_ci_lower_gt_zero'] and adjusted[key] < 0.05)
        all_results[encoder] = {'analysis_role': 'primary' if encoder == PRIMARY_ENCODER else 'secondary_reproduction', 'metric': 'QWK' if dataset == 'sicapv2' else 'mean_fold_patient_AUROC', 'metrics': {key: float(value) for key, value in metrics.items()}, 'fold_metrics': fold_metrics, 'fold_parameters': fold_parameters, 'comparisons': comparisons, 'registered_3d_positive': bool(comparisons['real3d_vs_2d'].get('holm_positive', False) and comparisons['real3d_vs_shuffled3d'].get('holm_positive', False)), 'registered_rgb_increment_positive': bool(comparisons['real3d_vs_rgb_mask_base'].get('holm_positive', False))}
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'official_test_touched': False, 'fusion': {'formula': 'base_logit / T_base + nonnegative_alpha * geometry_logit / T_geometry', 'cross_fit': 'for held-out fold f, fit on OOF predictions from all folds except f', 'temperature_bounds': list(TEMPERATURE_BOUNDS), 'alpha_bounds': list(ALPHA_BOUNDS), 'primary_encoder': PRIMARY_ENCODER, 'secondary_encoders': list(SECONDARY_ENCODERS), 'primary_comparisons': list(PRIMARY_COMPARISONS), 'bootstrap_seed': BOOTSTRAP_SEED}, 'results': all_results, 'source_provenance': source_provenance}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
