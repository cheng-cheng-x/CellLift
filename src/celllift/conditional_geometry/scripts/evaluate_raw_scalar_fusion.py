from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.late_fusion_protocol import GEOMETRY_PROTOCOL_ID, RGB_MASK_PROTOCOL_ID
from celllift.conditional_geometry.raw_scalar_fusion_protocol import ALPHA_BOUNDS, BASE_ARM, DATASET, GEOMETRY_ARMS, OFFICIAL_TEST_ALLOWED, OPTIMIZER_MAX_ITERATIONS, OPTIMIZER_TOLERANCE, PRIMARY_COMPARISONS, PRIMARY_ENCODER, PROTOCOL_ID, SECONDARY_ENCODERS
from celllift.conditional_geometry.scripts.evaluate import _metric
from celllift.conditional_geometry.scripts.evaluate_calibrated_fusion import _concat_batches, _nll, _paired_bootstrap, _mean_seed_batch, logits_to_probabilities, probabilities_to_logits, validate_patient_disjoint
from celllift.conditional_geometry.scripts.evaluate_late_fusion import PredictionBatch, _aligned, _atomic_json, _atomic_parquet, _holm_adjust, _prediction_rows, _read_source

def fit_raw_scalar_gate(truth: np.ndarray, base_values: np.ndarray, geometry_values: np.ndarray) -> tuple[float, bool, float]:
    from scipy.optimize import minimize
    base_logits = probabilities_to_logits(DATASET, base_values)
    geometry_logits = probabilities_to_logits(DATASET, geometry_values)

    def objective(alpha: np.ndarray) -> float:
        return _nll(DATASET, truth, base_logits + float(alpha[0]) * geometry_logits)
    result = minimize(objective, x0=np.asarray([0.25], np.float64), method='L-BFGS-B', bounds=[ALPHA_BOUNDS], options={'ftol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    alpha = float(result.x[0])
    if not np.isfinite(alpha) or not ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]:
        raise RuntimeError('raw scalar optimizer returned an invalid alpha')
    return (alpha, bool(result.success), float(result.fun))

def apply_raw_scalar_gate(base_values: np.ndarray, geometry_values: np.ndarray, alpha: float) -> np.ndarray:
    if not ALPHA_BOUNDS[0] <= float(alpha) <= ALPHA_BOUNDS[1]:
        raise ValueError('raw scalar alpha is outside the registered bounds')
    fused = probabilities_to_logits(DATASET, base_values) + float(alpha) * probabilities_to_logits(DATASET, geometry_values)
    output = logits_to_probabilities(DATASET, fused)
    if not np.all(np.isfinite(output)) or not np.allclose(output.sum(axis=1), 1.0):
        raise RuntimeError('raw scalar fusion produced invalid probabilities')
    return output

def _config_gate(cfg: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...], int, int]:
    if cfg.get('protocol_id') != PROTOCOL_ID or cfg.get('dataset') != DATASET:
        raise RuntimeError('raw scalar fusion is locked to the SICAP scalar_fusion protocol')
    if OFFICIAL_TEST_ALLOWED or cfg['split'].get('official_test_frozen') is not True:
        raise RuntimeError('official TEST must remain frozen')
    fusion = cfg['fusion']
    seeds = tuple(map(int, fusion['seeds']))
    encoders = tuple(map(str, fusion['encoders']))
    if encoders != (PRIMARY_ENCODER,) + SECONDARY_ENCODERS:
        raise ValueError('scalar_fusion encoder order must be meanpool primary, deepsets secondary')
    if tuple(map(float, fusion['alpha_bounds'])) != ALPHA_BOUNDS:
        raise ValueError('scalar_fusion alpha bounds differ from the registered protocol')
    if fusion.get('temperature_scaling') is not False or fusion.get('classwise_alpha') is not False:
        raise ValueError('scalar_fusion requires raw logits and one shared scalar alpha')
    folds = int(cfg['split']['validation_folds'])
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    if folds != 4 or replicates != 10000 or (not seeds):
        raise ValueError('scalar_fusion requires four folds, five seeds and 10,000 bootstraps')
    return (seeds, encoders, folds, replicates)

def evaluate_raw_scalar_fusion(cfg: Mapping[str, Any]) -> dict[str, Any]:
    seeds, encoders, folds, replicates = _config_gate(cfg)
    base_root = Path(cfg['paths']['rgb_mask_result_root'])
    geometry_root = Path(cfg['paths']['geometry_result_root'])
    result_root = Path(cfg['paths']['result_root'])
    source_provenance: dict[str, Any] = {}
    all_results: dict[str, Any] = {}
    for encoder in encoders:
        fold_base: list[PredictionBatch] = []
        fold_geometry: dict[str, list[PredictionBatch]] = {arm_id: [] for arm_id in GEOMETRY_ARMS}
        for fold in range(folds):
            seed_base = []
            seed_geometry = {arm_id: [] for arm_id in GEOMETRY_ARMS}
            for seed in seeds:
                base, base_provenance = _read_source(base_root, DATASET, encoder, BASE_ARM, seed, fold, protocol_id=RGB_MASK_PROTOCOL_ID, use_rgb=True)
                seed_base.append(base)
                key = f'{encoder}/seed_{seed}/fold_{fold}'
                source_provenance.setdefault(key, {})['rgb_mask'] = base_provenance
                for arm_id, source_arm in GEOMETRY_ARMS.items():
                    geometry, geometry_provenance = _read_source(geometry_root, DATASET, encoder, source_arm, seed, fold, protocol_id=GEOMETRY_PROTOCOL_ID, use_rgb=False)
                    _aligned(base, geometry)
                    seed_geometry[arm_id].append(geometry)
                    source_provenance[key][arm_id] = geometry_provenance
            fold_base.append(_mean_seed_batch(seed_base))
            for arm_id in GEOMETRY_ARMS:
                fold_geometry[arm_id].append(_mean_seed_batch(seed_geometry[arm_id]))
        validate_patient_disjoint(fold_base)
        fold_predictions: dict[str, list[np.ndarray]] = {'R0_RGB_MASK': [batch.values.copy() for batch in fold_base], **{arm_id: [] for arm_id in GEOMETRY_ARMS}}
        fold_parameters: dict[str, Any] = {}
        fold_metrics: dict[str, dict[str, float]] = {arm: {} for arm in fold_predictions}
        for held_out in range(folds):
            fit_folds = [fold for fold in range(folds) if fold != held_out]
            train_base = _concat_batches([fold_base[fold] for fold in fit_folds])
            target_base = fold_base[held_out]
            if set(map(str, train_base.patients)) & set(map(str, target_base.patients)):
                raise RuntimeError('held-out patient entered raw scalar fitting')
            fold_parameters[f'fold_{held_out}'] = {'held_out_fold': held_out, 'fit_folds': fit_folds, 'fit_samples': len(train_base.truth), 'fit_patients': len(set(map(str, train_base.patients))), 'held_out_samples': len(target_base.truth), 'held_out_patients': len(set(map(str, target_base.patients))), 'arms': {}}
            for arm_id in GEOMETRY_ARMS:
                train_geometry = _concat_batches([fold_geometry[arm_id][fold] for fold in fit_folds])
                target_geometry = fold_geometry[arm_id][held_out]
                _aligned(train_base, train_geometry)
                alpha, success, fit_nll = fit_raw_scalar_gate(train_base.truth, train_base.values, train_geometry.values)
                fused = apply_raw_scalar_gate(target_base.values, target_geometry.values, alpha)
                fold_predictions[arm_id].append(fused)
                fold_parameters[f'fold_{held_out}']['arms'][arm_id] = {'alpha': alpha, 'optimizer_success': success, 'fit_nll': fit_nll, 'boundary_hit': bool(abs(alpha - ALPHA_BOUNDS[0]) <= 1e-06 or abs(alpha - ALPHA_BOUNDS[1]) <= 1e-06)}
            for arm_id, values_by_fold in fold_predictions.items():
                values = values_by_fold[held_out]
                fold_metrics[arm_id][f'fold_{held_out}'] = _metric(DATASET, target_base.truth, values)
                path = result_root / 'predictions' / encoder / arm_id / f'fold_{held_out}.parquet'
                checksum = _atomic_parquet(path, _prediction_rows(DATASET, target_base, values))
                fold_parameters[f'fold_{held_out}'].setdefault('prediction_sha256', {})[arm_id] = checksum
        combined_base = _concat_batches(fold_base)
        combined_predictions = {arm: np.concatenate(values_by_fold, axis=0) for arm, values_by_fold in fold_predictions.items()}
        metrics = {arm: _metric(DATASET, combined_base.truth, values) for arm, values in combined_predictions.items()}

        def compare(candidate: str, baseline: str) -> dict[str, Any]:
            return _paired_bootstrap(DATASET, combined_base.patients, combined_base.truth, combined_predictions[candidate], combined_predictions[baseline], replicates)
        comparisons = {'real3d_vs_2d': compare('R2_REAL3D', 'R1_2D'), 'real3d_vs_shuffled3d': compare('R2_REAL3D', 'R3_SHUF3D'), 'real3d_vs_rgb_mask_base': compare('R2_REAL3D', 'R0_RGB_MASK'), 'geometry2d_vs_rgb_mask_base': compare('R1_2D', 'R0_RGB_MASK')}
        adjusted = _holm_adjust({key: comparisons[key]['one_sided_p'] for key in PRIMARY_COMPARISONS})
        for key in PRIMARY_COMPARISONS:
            comparisons[key]['holm_adjusted_one_sided_p'] = adjusted[key]
            comparisons[key]['holm_positive'] = bool(comparisons[key]['positive_ci_lower_gt_zero'] and adjusted[key] < 0.05)
        all_results[encoder] = {'analysis_role': 'primary' if encoder == PRIMARY_ENCODER else 'secondary_reproduction', 'metric': 'QWK', 'metrics': {key: float(value) for key, value in metrics.items()}, 'fold_metrics': fold_metrics, 'fold_parameters': fold_parameters, 'comparisons': comparisons, 'registered_3d_positive': bool(comparisons['real3d_vs_2d'].get('holm_positive', False) and comparisons['real3d_vs_shuffled3d'].get('holm_positive', False)), 'registered_rgb_increment_positive': bool(comparisons['real3d_vs_rgb_mask_base'].get('holm_positive', False))}
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': DATASET, 'official_test_touched': False, 'fusion': {'formula': 'base_centered_log_probability + shared_nonnegative_alpha * geometry_centered_log_probability', 'temperature_scaling': False, 'classwise_alpha': False, 'alpha_bounds': list(ALPHA_BOUNDS), 'primary_comparisons': list(PRIMARY_COMPARISONS)}, 'results': all_results, 'source_provenance': source_provenance}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
