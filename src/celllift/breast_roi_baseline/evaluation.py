from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import dataclasses
from celllift.runtime import json
import os
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from celllift.breast_roi_baseline.cache import atomic_json, sha256_file
from celllift.breast_roi_baseline.fusion import apply_registered_arm, fit_registered_arm
from celllift.breast_roi_baseline.metrics import holm_correction, macro_f1, paired_wsi_cluster_bootstrap
FULL_EXPERT_TO_SHORT = {'E0_RGB_MASK2D': 'E0', 'E1_MASK2D': 'E1', 'E2_MASK_DIRECT3D': 'E2', 'E3_MASK_SHUF_DIRECT3D': 'E3', 'E4_MASK_RESIDUAL3D': 'E4', 'E5_MASK_SHUF_RESIDUAL3D': 'E5'}
ARM_EXPERT = {'A0': 'E0', 'A1': 'E1', 'A2': 'E2', 'A3': 'E3', 'A4': 'E4', 'A5': 'E5'}

def _fixed_list_numpy(column: Any) -> np.ndarray:
    values = column.combine_chunks()
    width = values.type.list_size
    return np.asarray(values.values.to_numpy(zero_copy_only=False), np.float64).reshape(len(values), width)

def _load_prediction(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    table = pq.read_table(path, partitioning=None)
    values = table.to_pydict()
    probability = _fixed_list_numpy(table['probability'])
    output = []
    for index in range(len(table)):
        output.append({'roi_id': str(values['roi_id'][index]), 'wsi_id': str(values['wsi_id'][index]), 'label': int(values['label'][index]), 'role': str(values['role'][index]), 'expert': FULL_EXPERT_TO_SHORT.get(str(values['expert'][index]), str(values['expert'][index])), 'fold': int(values['fold'][index]), 'seed': int(values['seed'][index]), 'probability': probability[index]})
    return output

def _ensemble(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['role'] == role:
            grouped[str(row['expert']), str(row['roi_id'])].append(row)
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (expert, roi), values in grouped.items():
        labels = {int(row['label']) for row in values}
        wsis = {str(row['wsi_id']) for row in values}
        if len(labels) != 1 or len(wsis) != 1:
            raise RuntimeError('prediction ensemble has inconsistent ROI metadata')
        probability = np.mean(np.stack([row['probability'] for row in values]), axis=0)
        probability /= probability.sum()
        output[expert][roi] = {'roi_id': roi, 'wsi_id': next(iter(wsis)), 'label': next(iter(labels)), 'probability': probability, 'members': len(values)}
    return dict(output)

def _aligned(experts: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if set(experts) != {f'E{index}' for index in range(6)}:
        raise RuntimeError(f'incomplete E0--E5 prediction set: {sorted(experts)}')
    rois = sorted(experts['E0'])
    if any((set(values) != set(rois) for values in experts.values())):
        raise RuntimeError('experts do not cover identical ROI sets')
    truth = np.asarray([experts['E0'][roi]['label'] for roi in rois], np.int64)
    wsi = np.asarray([experts['E0'][roi]['wsi_id'] for roi in rois], object)
    probability = {expert: np.stack([values[roi]['probability'] for roi in rois]) for expert, values in experts.items()}
    return (rois, truth, wsi, probability)

def _classification_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
    prediction = probability.argmax(1)
    classes = list(range(probability.shape[1]))
    try:
        auc = float(roc_auc_score(truth, probability, multi_class='ovr', average='macro', labels=classes))
    except ValueError:
        auc = float('nan')
    return {'macro_f1': macro_f1(truth, probability, classes=classes), 'balanced_accuracy': float(balanced_accuracy_score(truth, prediction)), 'accuracy': float(accuracy_score(truth, prediction)), 'macro_ovr_auroc': auc, 'confusion_matrix': confusion_matrix(truth, prediction, labels=classes).tolist()}

def _fit_dict(value: Any) -> dict[str, Any]:
    result = dataclasses.asdict(value)
    for key, item in list(result.items()):
        if isinstance(item, np.ndarray):
            result[key] = item.tolist()
    if hasattr(value, 'boundary_hit'):
        result['boundary_hit'] = bool(value.boundary_hit)
    if hasattr(value, 'alpha_boundary_hits'):
        result['alpha_boundary_hits'] = value.alpha_boundary_hits.tolist()
    return result

def evaluate_fusion(*, prediction_paths: Sequence[str | Path], output_dir: str | Path, task: str, encoder: str, phase: str, bootstrap_replicates: int=10000) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(output_dir)
    manifest_path = destination / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            return previous
    rows = [row for path in prediction_paths for row in _load_prediction(path)]
    oof = _ensemble(rows, 'heldout')
    external = _ensemble(rows, 'external')
    oof_rois, oof_truth, _, oof_probability = _aligned(oof)
    external_rois, truth, wsi, external_probability = _aligned(external)
    all_results: dict[str, Any] = {}
    prediction_records = []
    protocol_probabilities: dict[str, dict[str, np.ndarray]] = {}
    for protocol in ('breast_roi', 'scalar_fusion'):
        arm_values = {}
        fits = {}
        for arm, expert in ARM_EXPERT.items():
            fit = fit_registered_arm(arm, protocol, oof_truth, oof_probability['E0'], None if arm == 'A0' else oof_probability[expert])
            fused = apply_registered_arm(arm, protocol, external_probability['E0'], fit, None if arm == 'A0' else external_probability[expert])
            fits[arm] = _fit_dict(fit)
            arm_values[arm] = fused
            for index, roi in enumerate(external_rois):
                prediction_records.append({'roi_id': roi, 'wsi_id': str(wsi[index]), 'label': int(truth[index]), 'protocol': protocol, 'arm': arm, 'probability': fused[index]})
        protocol_probabilities[protocol] = arm_values
        all_results[protocol] = {'fits': fits, 'arms': {arm: _classification_metrics(truth, value) for arm, value in arm_values.items()}}
    expert_metrics = {expert: _classification_metrics(truth, value) for expert, value in external_probability.items()}
    comparison_families = {'direct3d': {'A2_minus_A1': ('A2', 'A1'), 'A2_minus_A3': ('A2', 'A3'), 'A2_minus_A0': ('A2', 'A0')}, 'residual3d': {'A4_minus_A1': ('A4', 'A1'), 'A4_minus_A5': ('A4', 'A5'), 'A4_minus_A0': ('A4', 'A0')}}
    statistics = {}
    primary = protocol_probabilities['breast_roi']
    for family, comparisons in comparison_families.items():
        values = {name: paired_wsi_cluster_bootstrap(truth, primary[candidate], primary[baseline], wsi, n_resamples=bootstrap_replicates, seed=20260830 + index).to_dict() for index, (name, (candidate, baseline)) in enumerate(comparisons.items())}
        adjusted = holm_correction({name: value['p_value_one_sided'] for name, value in values.items()})
        for name in values:
            values[name].update(adjusted[name])
            values[name]['positive'] = bool(values[name]['ci_low'] > 0 and values[name]['reject'])
        statistics[family] = values
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / 'fusion_predictions.parquet'
    temporary = prediction_path.with_name(f'.{prediction_path.name}.tmp.{os.getpid()}')
    matrix = np.stack([row['probability'] for row in prediction_records]).astype(np.float32)
    fixed = pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1)), matrix.shape[1])
    pq.write_table(pa.table({'roi_id': pa.array([row['roi_id'] for row in prediction_records]), 'wsi_id': pa.array([row['wsi_id'] for row in prediction_records]), 'label': pa.array(np.asarray([row['label'] for row in prediction_records], np.int8)), 'protocol': pa.array([row['protocol'] for row in prediction_records]), 'arm': pa.array([row['arm'] for row in prediction_records]), 'probability': fixed}), temporary, compression='zstd')
    os.replace(temporary, prediction_path)
    payload = {'status': 'PASS', 'task': task, 'encoder': encoder, 'phase': phase, 'oof_rois': len(oof_rois), 'external_rois': len(external_rois), 'expert_metrics': expert_metrics, 'fusion': all_results, 'statistics': statistics, 'prediction_paths': [str(path) for path in prediction_paths], 'fusion_predictions': str(prediction_path), 'fusion_predictions_sha256': sha256_file(prediction_path), 'interpretation': {'direct3d_positive': all((value['positive'] for key, value in statistics['direct3d'].items() if key != 'A2_minus_A0')), 'direct3d_improves_rgb': statistics['direct3d']['A2_minus_A0']['positive'], 'residual3d_positive': all((value['positive'] for key, value in statistics['residual3d'].items() if key != 'A4_minus_A0'))}}
    atomic_json(manifest_path, payload)
    return payload
