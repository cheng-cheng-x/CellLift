from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.mask_protocol import MASK_ARMS, MASK_PROTOCOL_ID, TARGET_COLUMNS
from celllift.conditional_geometry.scripts.evaluate import _atomic_json, _auroc, _ensemble_crc_predictions_by_fold, _ensemble_predictions, _metric, _paired_bootstrap, _paired_fold_stratified_crc_bootstrap

def evaluate_mask_validation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    rgb_mode = str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays'
    if rgb_mode:
        from celllift.conditional_geometry.rgb_mask_protocol import RGB_MASK_PROTOCOL_ID
        protocol_id = RGB_MASK_PROTOCOL_ID
    else:
        protocol_id = MASK_PROTOCOL_ID
    result_root = Path(cfg['paths']['result_root'])
    folds = int(cfg['split']['validation_folds'])
    probe_folds = []
    for fold in range(folds):
        path = result_root / 'probe' / f'fold_{fold:02d}' / 'metrics.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS' or payload.get('protocol_id') != protocol_id or payload.get('official_test_touched') is not False:
            raise RuntimeError(f'mask probe fold gate failed: {path}')
        probe_folds.append(payload)
    probe_summary = {}
    for target in TARGET_COLUMNS:
        anchor = [row['validation_r2']['targets'][target]['r2_anchor_weighted'] for row in probe_folds]
        graph = [row['validation_r2']['targets'][target]['r2_graph_balanced'] for row in probe_folds]
        probe_summary[target] = {'r2_anchor_weighted_by_fold': anchor, 'r2_anchor_weighted_mean': float(np.mean(anchor)), 'r2_anchor_weighted_std': float(np.std(anchor, ddof=1)), 'r2_graph_balanced_by_fold': graph, 'r2_graph_balanced_mean': float(np.mean(graph)), 'all_folds_r2_gt_0_9': bool(np.all(np.asarray(anchor) > 0.9))}
    seeds = tuple(map(int, cfg['downstream']['seeds']))
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    baseline, residual, shuffled = MASK_ARMS
    downstream: dict[str, Any] = {}
    for encoder in cfg['downstream']['encoders']:
        if cfg['dataset'] == 'tcga_crc_msi':
            fold_ensemble, fold_metrics = ({}, {})
            for arm in MASK_ARMS:
                values, metrics = _ensemble_crc_predictions_by_fold(result_root, encoder, arm, seeds, folds)
                fold_ensemble[arm], fold_metrics[arm] = (values, metrics)
            downstream[encoder] = {'metric': 'mean_fold_patient_AUROC', 'ensemble_metrics': {arm: float(np.mean([_auroc(truth, score) for _, truth, score in values])) for arm, values in fold_ensemble.items()}, 'seed_fold_metrics': fold_metrics, 'residual_vs_mask2d': _paired_fold_stratified_crc_bootstrap(fold_ensemble[residual], fold_ensemble[baseline], replicates), 'residual_vs_shuffled': _paired_fold_stratified_crc_bootstrap(fold_ensemble[residual], fold_ensemble[shuffled], replicates)}
            continue
        ensemble, fold_metrics = ({}, {})
        reference_patient = reference_truth = None
        for arm in MASK_ARMS:
            patient, truth, values, metrics = _ensemble_predictions(result_root, cfg['dataset'], encoder, arm, seeds, folds)
            if reference_patient is not None and (not np.array_equal(patient, reference_patient) or not np.array_equal(truth, reference_truth)):
                raise RuntimeError('mask arm OOF samples differ')
            reference_patient, reference_truth = (patient, truth)
            ensemble[arm], fold_metrics[arm] = (values, metrics)
        downstream[encoder] = {'metric': 'QWK', 'ensemble_metrics': {arm: _metric(cfg['dataset'], reference_truth, values) for arm, values in ensemble.items()}, 'seed_fold_metrics': fold_metrics, 'residual_vs_mask2d': _paired_bootstrap(cfg['dataset'], reference_patient, reference_truth, ensemble[residual], ensemble[baseline], replicates), 'residual_vs_shuffled': _paired_bootstrap(cfg['dataset'], reference_patient, reference_truth, ensemble[residual], ensemble[shuffled], replicates)}
    payload = {'status': 'PASS', 'protocol_id': protocol_id, 'dataset': cfg['dataset'], 'conditioning': 'fold-specific frozen RGB512 + nucleus-mask 36 rays; no XY/graph edges/cell2D/NCR2D' if rgb_mode else 'nucleus-mask 36 rays only; no RGB/XY/graph edges/cell2D/NCR2D', 'official_test_touched': False, 'probe': probe_summary, 'downstream': downstream}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
