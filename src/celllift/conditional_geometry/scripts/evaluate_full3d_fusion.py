from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.full3d_protocol import ALPHA_BOUNDS, BASE_ARM, ENCODERS, FUSION_ARMS, OFFICIAL_TEST_ALLOWED, PRIMARY_COMPARISONS, PROTOCOL_ID, SEEDS, TEMPERATURE_BOUNDS
from celllift.conditional_geometry.late_fusion_protocol import GEOMETRY_PROTOCOL_ID, RGB_MASK_PROTOCOL_ID
from celllift.conditional_geometry.scripts.evaluate import _auroc, _metric
from celllift.conditional_geometry.scripts.evaluate_calibrated_fusion import _concat_batches, _gate_payload, _mean_seed_batch, _paired_bootstrap, _paired_crc_bootstrap, apply_gate, fit_gate, validate_patient_disjoint
from celllift.conditional_geometry.scripts.evaluate_late_fusion import PredictionBatch, _aligned, _atomic_json, _atomic_parquet, _holm_adjust, _prediction_rows, _read_source
from celllift.conditional_geometry.scripts.evaluate_raw_scalar_fusion import apply_raw_scalar_gate, fit_raw_scalar_gate

def _config_gate(cfg: Mapping[str, Any]) -> tuple[str, tuple[int, ...], tuple[str, ...], int, int]:
    dataset = str(cfg.get('dataset'))
    if cfg.get('protocol_id') != PROTOCOL_ID or OFFICIAL_TEST_ALLOWED:
        raise RuntimeError('raw_residual_comparison full-3D fusion protocol is not locked')
    if dataset not in {'sicapv2', 'tcga_crc_msi'}:
        raise ValueError(f'unsupported dataset: {dataset}')
    if cfg['split'].get('official_test_frozen') is not True:
        raise RuntimeError('official TEST must remain frozen')
    fusion = cfg['fusion']
    seeds = tuple(map(int, fusion['seeds']))
    encoders = tuple(map(str, fusion['encoders']))
    if seeds != SEEDS or encoders != ENCODERS:
        raise ValueError('raw_residual_comparison requires the locked five seeds and encoder order')
    if tuple(map(float, fusion['alpha_bounds'])) != ALPHA_BOUNDS:
        raise ValueError('raw_residual_comparison alpha bounds differ from the registered protocol')
    expected_temperature = dataset == 'tcga_crc_msi'
    if bool(fusion.get('temperature_scaling')) != expected_temperature:
        raise ValueError('SICAP uses raw logits; CRC uses temperature calibration')
    if expected_temperature and tuple(map(float, fusion['temperature_bounds'])) != TEMPERATURE_BOUNDS:
        raise ValueError('raw_residual_comparison temperature bounds differ from the registered protocol')
    folds = int(cfg['split']['validation_folds'])
    if folds != (4 if dataset == 'sicapv2' else 5):
        raise ValueError('unexpected outer-fold count')
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    if replicates != 10000:
        raise ValueError('raw_residual_comparison requires exactly 10,000 paired bootstrap replicates')
    return (dataset, seeds, encoders, folds, replicates)

def _fit_and_apply(dataset: str, train_base: PredictionBatch, train_geometry: PredictionBatch, target_base: PredictionBatch, target_geometry: PredictionBatch) -> tuple[np.ndarray, dict[str, Any]]:
    if dataset == 'sicapv2':
        alpha, success, fit_nll = fit_raw_scalar_gate(train_base.truth, train_base.values, train_geometry.values)
        values = apply_raw_scalar_gate(target_base.values, target_geometry.values, alpha)
        return (values, {'alpha': alpha, 'optimizer_success': success, 'fit_nll': fit_nll, 'temperature_scaling': False, 'boundary_hit': bool(abs(alpha - ALPHA_BOUNDS[0]) <= 1e-06 or abs(alpha - ALPHA_BOUNDS[1]) <= 1e-06)})
    gate = fit_gate(dataset, train_base.truth, train_base.values, train_geometry.values, calibrate_temperatures=True)
    return (apply_gate(dataset, target_base.values, target_geometry.values, gate), _gate_payload(gate))

def _recommendation(metrics: Mapping[str, float], comparison: Mapping[str, Any], full_positive: bool) -> str:
    delta = float(metrics['conditional_geometry_FULL3D'] - metrics['calibrated_fusion_RESIDUAL3D'])
    ci = comparison['ci95']
    if float(ci[1]) < 0:
        return 'residual3d_significantly_better'
    if full_positive and delta >= -0.002:
        return 'prefer_full3d_parsimonious_within_0.002'
    return 'inconclusive_keep_full3d_and_residual3d'

def evaluate_full3d_fusion(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset, seeds, encoders, folds, replicates = _config_gate(cfg)
    base_root = Path(cfg['paths']['rgb_mask_result_root'])
    residual_root = Path(cfg['paths']['residual_result_root'])
    full_root = Path(cfg['paths']['full3d_result_root'])
    result_root = Path(cfg['paths']['result_root'])
    source_roots = {'residual': residual_root, 'full': full_root}
    source_protocols = {'residual': GEOMETRY_PROTOCOL_ID, 'full': PROTOCOL_ID}
    results: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for encoder in encoders:
        fold_base: list[PredictionBatch] = []
        fold_geometry: dict[str, list[PredictionBatch]] = {arm: [] for arm in FUSION_ARMS}
        for fold in range(folds):
            seed_base: list[PredictionBatch] = []
            seed_geometry: dict[str, list[PredictionBatch]] = {arm: [] for arm in FUSION_ARMS}
            for seed in seeds:
                key = f'{encoder}/seed_{seed}/fold_{fold}'
                base, base_provenance = _read_source(base_root, dataset, encoder, BASE_ARM, seed, fold, protocol_id=RGB_MASK_PROTOCOL_ID, use_rgb=True)
                seed_base.append(base)
                provenance[key] = {'rgb_mask': base_provenance}
                for arm, (source_kind, source_arm) in FUSION_ARMS.items():
                    geometry, geometry_provenance = _read_source(source_roots[source_kind], dataset, encoder, source_arm, seed, fold, protocol_id=source_protocols[source_kind], use_rgb=False)
                    _aligned(base, geometry)
                    seed_geometry[arm].append(geometry)
                    provenance[key][arm] = geometry_provenance
            fold_base.append(_mean_seed_batch(seed_base))
            for arm in FUSION_ARMS:
                fold_geometry[arm].append(_mean_seed_batch(seed_geometry[arm]))
        validate_patient_disjoint(fold_base)
        fold_predictions: dict[str, list[np.ndarray]] = {'auxiliary_5feceb_RGB_MASK': [batch.values.copy() for batch in fold_base], **{arm: [] for arm in FUSION_ARMS}}
        fold_parameters: dict[str, Any] = {}
        fold_metrics: dict[str, dict[str, float]] = {arm: {} for arm in fold_predictions}
        for held_out in range(folds):
            fit_folds = [fold for fold in range(folds) if fold != held_out]
            train_base = _concat_batches([fold_base[fold] for fold in fit_folds])
            target_base = fold_base[held_out]
            if set(map(str, train_base.patients)) & set(map(str, target_base.patients)):
                raise RuntimeError('held-out patient entered raw_residual_comparison gate fitting')
            fold_parameters[f'fold_{held_out}'] = {'held_out_fold': held_out, 'fit_folds': fit_folds, 'fit_samples': len(train_base.truth), 'fit_patients': len(set(map(str, train_base.patients))), 'held_out_samples': len(target_base.truth), 'held_out_patients': len(set(map(str, target_base.patients))), 'arms': {}}
            for arm in FUSION_ARMS:
                train_geometry = _concat_batches([fold_geometry[arm][fold] for fold in fit_folds])
                target_geometry = fold_geometry[arm][held_out]
                _aligned(train_base, train_geometry)
                values, parameters = _fit_and_apply(dataset, train_base, train_geometry, target_base, target_geometry)
                fold_predictions[arm].append(values)
                fold_parameters[f'fold_{held_out}']['arms'][arm] = parameters
            for arm, values_by_fold in fold_predictions.items():
                values = values_by_fold[held_out]
                fold_metrics[arm][f'fold_{held_out}'] = _metric(dataset, target_base.truth, values)
                path = result_root / 'predictions' / encoder / arm / f'fold_{held_out}.parquet'
                checksum = _atomic_parquet(path, _prediction_rows(dataset, target_base, values))
                fold_parameters[f'fold_{held_out}'].setdefault('prediction_sha256', {})[arm] = checksum
        if dataset == 'sicapv2':
            combined = _concat_batches(fold_base)
            predictions = {arm: np.concatenate(values_by_fold, axis=0) for arm, values_by_fold in fold_predictions.items()}
            metrics = {arm: _metric(dataset, combined.truth, values) for arm, values in predictions.items()}

            def compare(candidate: str, baseline: str) -> dict[str, Any]:
                return _paired_bootstrap(dataset, combined.patients, combined.truth, predictions[candidate], predictions[baseline], replicates)
        else:
            predictions = {arm: [(fold_base[fold].patients, fold_base[fold].truth, values_by_fold[fold]) for fold in range(folds)] for arm, values_by_fold in fold_predictions.items()}
            metrics = {arm: float(np.mean([_auroc(y, score) for _, y, score in rows])) for arm, rows in predictions.items()}

            def compare(candidate: str, baseline: str) -> dict[str, Any]:
                return _paired_crc_bootstrap(predictions[candidate], predictions[baseline], replicates)
        comparisons = {'full3d_vs_2d': compare('conditional_geometry_FULL3D', 'set_encoding_2D'), 'full3d_vs_shuffled_full3d': compare('conditional_geometry_FULL3D', 'mask_conditioning_SHUF_FULL3D'), 'full3d_vs_rgb_mask_base': compare('conditional_geometry_FULL3D', 'auxiliary_5feceb_RGB_MASK'), 'residual3d_vs_2d': compare('calibrated_fusion_RESIDUAL3D', 'set_encoding_2D'), 'residual3d_vs_shuffled_residual3d': compare('calibrated_fusion_RESIDUAL3D', 'raw_residual_comparison_SHUF_RESIDUAL3D'), 'residual3d_vs_rgb_mask_base': compare('calibrated_fusion_RESIDUAL3D', 'auxiliary_5feceb_RGB_MASK'), 'full3d_vs_residual3d': compare('conditional_geometry_FULL3D', 'calibrated_fusion_RESIDUAL3D')}
        adjusted = _holm_adjust({key: comparisons[key]['one_sided_p'] for key in PRIMARY_COMPARISONS})
        for key in PRIMARY_COMPARISONS:
            comparisons[key]['holm_adjusted_one_sided_p'] = adjusted[key]
            comparisons[key]['holm_positive'] = bool(comparisons[key]['positive_ci_lower_gt_zero'] and adjusted[key] < 0.05)
        full_positive = bool(comparisons['full3d_vs_2d']['holm_positive'] and comparisons['full3d_vs_shuffled_full3d']['holm_positive'])
        results[encoder] = {'analysis_role': 'primary' if encoder == 'meanpool' else 'secondary_reproduction', 'metric': 'QWK' if dataset == 'sicapv2' else 'mean_fold_patient_AUROC', 'metrics': {key: float(value) for key, value in metrics.items()}, 'fold_metrics': fold_metrics, 'fold_parameters': fold_parameters, 'comparisons': comparisons, 'registered_full3d_positive': full_positive, 'registered_rgb_increment_positive': bool(comparisons['full3d_vs_rgb_mask_base']['holm_positive']), 'representation_recommendation': _recommendation(metrics, comparisons['full3d_vs_residual3d'], full_positive)}
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'official_test_touched': False, 'fusion': {'sicap': 'raw base centered-log-probability + shared nonnegative alpha * geometry centered-log-probability', 'crc': 'temperature-calibrated base logit + nonnegative scalar alpha * calibrated geometry logit', 'cross_fit': 'fit on all outer folds except held-out fold; apply once to held-out fold', 'alpha_bounds': list(ALPHA_BOUNDS), 'temperature_bounds': list(TEMPERATURE_BOUNDS), 'primary_comparisons': list(PRIMARY_COMPARISONS), 'full_vs_residual_role': 'descriptive model selection; excluded from confirmatory Holm family'}, 'results': results, 'source_provenance': provenance}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
