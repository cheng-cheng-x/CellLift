from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
import math
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .fusion import apply_crc, cross_fit_crc, cross_fit_sicap, fit_crc_calibration, probability_logit
from .io_utils import atomic_json, atomic_parquet
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS

def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path, partitioning=None).to_pylist()

def _paper_seed_average(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    result_root = Path(cfg['paths']['result_root'])
    folds = int(cfg['split']['validation_folds'])
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for fold in range(folds):
        for seed in SEEDS:
            path = result_root / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}' / 'validation_predictions.parquet'
            for row in _rows(path):
                grouped[fold, str(row['graph_id'])].append(row)
    output = []
    for (fold, graph_id), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f'paper prediction seed coverage mismatch: {fold}/{graph_id}')
        probabilities = np.mean([np.asarray(row['probabilities'], float) for row in values], axis=0)
        output.append({'fold': fold, 'graph_id': graph_id, 'patient_id': str(values[0]['patient_id']), 'label_id': int(values[0]['label_id']), 'probabilities': probabilities})
    return output

def _geometry_seed_average(cfg: Mapping[str, Any], encoder: str, arm_id: str) -> list[dict[str, Any]]:
    result_root = Path(cfg['paths']['result_root']) / 'geometry_only' / 'screen' / encoder / arm_id
    folds = int(cfg['split']['validation_folds'])
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for fold in range(folds):
        for seed in SEEDS:
            path = result_root / f'seed_{seed}' / f'fold_{fold}' / 'validation_predictions.parquet'
            for row in _rows(path):
                key = str(row['graph_id'] if 'graph_id' in row else row['sample_id']) if cfg['dataset'] == 'sicapv2' else str(row['patient_id'])
                grouped[fold, key].append(row)
    output = []
    for (fold, key), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f'geometry prediction seed coverage mismatch: {encoder}/{arm_id}/{fold}/{key}')
        if cfg['dataset'] == 'sicapv2':
            probability = np.mean([[float(row[f'prob_{index}']) for index in range(4)] for row in values], axis=0)
        else:
            probability = float(np.mean([float(row['score']) for row in values]))
        output.append({'fold': fold, 'key': key, 'patient_id': str(values[0]['patient_id']), 'label_id': int(values[0]['y_true']), 'probability': probability})
    return output

def _paper_single_seed(cfg: Mapping[str, Any], seed: int) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'paper_baseline'
    output = []
    for fold in range(int(cfg['split']['validation_folds'])):
        for row in _rows(root / f'fold_{fold:02d}' / f'seed_{seed}' / 'validation_predictions.parquet'):
            output.append({**row, 'fold': fold})
    return output

def _geometry_single_seed(cfg: Mapping[str, Any], encoder: str, arm_id: str, seed: int) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'geometry_only' / 'screen' / encoder / arm_id
    output = []
    for fold in range(int(cfg['split']['validation_folds'])):
        output.extend(({**row, 'fold': fold} for row in _rows(root / f'seed_{seed}' / f'fold_{fold}' / 'validation_predictions.parquet')))
    return output

def _validation_component_metrics(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.metrics import roc_auc_score
    dataset = str(cfg['dataset'])
    output: list[dict[str, Any]] = []
    eligible = _geometry_eligible_ids(cfg) if dataset == 'tcga_crc_msi' else set()
    for seed in SEEDS:
        paper = _paper_single_seed(cfg, seed)
        if dataset == 'sicapv2':
            pmap = {(int(row['fold']), str(row['graph_id'])): row for row in paper}
            keys = sorted(pmap)
            labels = np.asarray([int(pmap[key]['label_id']) for key in keys])
            folds = np.asarray([key[0] for key in keys])
            rgb = np.stack([np.asarray(pmap[key]['probabilities'], float) for key in keys])
            output.append({'component': 'seed', 'seed': seed, 'fold': None, 'encoder': 'paper', 'arm_id': 'R0', **_sicap_metrics(labels, rgb)})
            for fold in sorted(set(map(int, folds))):
                held = folds == fold
                output.append({'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': 'paper', 'arm_id': 'R0', **_sicap_metrics(labels[held], rgb[held])})
            for encoder in ENCODERS:
                for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                    geometry = _geometry_single_seed(cfg, encoder, o_arm, seed)
                    gmap = {(int(row['fold']), str(row['graph_id'] if 'graph_id' in row else row['sample_id'])): row for row in geometry}
                    if set(gmap) != set(keys):
                        raise RuntimeError(f'single-seed SICAP join mismatch: {seed}/{encoder}/{o_arm}')
                    gp = np.stack([[float(gmap[key][f'prob_{index}']) for index in range(4)] for key in keys])
                    gl = np.log(np.clip(gp, 1e-07, 1.0))
                    rl = np.log(np.clip(rgb, 1e-07, 1.0))
                    fused, _ = cross_fit_sicap(folds, rl, gl, labels)
                    fp = np.exp(fused - fused.max(1, keepdims=True))
                    fp /= fp.sum(1, keepdims=True)
                    r_arm = 'R' + o_arm[1:]
                    output.extend([{'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels, gp)}, {'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': r_arm, **_sicap_metrics(labels, fp)}])
                    for fold in sorted(set(map(int, folds))):
                        held = folds == fold
                        output.extend([{'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels[held], gp[held])}, {'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': r_arm, **_sicap_metrics(labels[held], fp[held])}])
        else:
            paper_patients = _crc_paper_patients(paper)
            pl = np.asarray([row['label_id'] for row in paper_patients])
            ps = np.asarray([row['score'] for row in paper_patients])
            pf = np.asarray([row['fold'] for row in paper_patients])
            output.append({'component': 'seed', 'seed': seed, 'fold': None, 'encoder': 'paper', 'arm_id': 'R0', **_crc_metrics(pl, ps, pf)})
            for fold in sorted(set(map(int, pf))):
                held = pf == fold
                output.append({'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': 'paper', 'arm_id': 'R0', 'patients': int(held.sum()), 'patient_AUROC': float(roc_auc_score(pl[held], ps[held]))})
            common_paper = [row for row in paper if str(row['graph_id']) in eligible]
            common = {(int(row['fold']), str(row['patient_id'])): row for row in _crc_paper_patients(common_paper)}
            keys = sorted(common)
            labels = np.asarray([common[key]['label_id'] for key in keys])
            folds = np.asarray([key[0] for key in keys])
            rgb = np.asarray([common[key]['logit'] for key in keys])
            for encoder in ENCODERS:
                for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                    geometry = _geometry_single_seed(cfg, encoder, o_arm, seed)
                    gmap = {(int(row['fold']), str(row['patient_id'])): row for row in geometry}
                    if set(keys) - set(gmap):
                        raise RuntimeError(f'single-seed CRC join mismatch: {seed}/{encoder}/{o_arm}')
                    gp = np.asarray([float(gmap[key]['score']) for key in keys])
                    gl = probability_logit(gp)
                    fused, _ = cross_fit_crc(folds, rgb, gl, labels)
                    fp = 1.0 / (1.0 + np.exp(-fused))
                    r_arm = 'R' + o_arm[1:]
                    output.extend([{'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': o_arm, **_crc_metrics(labels, gp, folds)}, {'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': r_arm, **_crc_metrics(labels, fp, folds)}])
                    for fold in sorted(set(map(int, folds))):
                        held = folds == fold
                        output.extend([{'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': o_arm, 'patients': int(held.sum()), 'patient_AUROC': float(roc_auc_score(labels[held], gp[held]))}, {'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': r_arm, 'patients': int(held.sum()), 'patient_AUROC': float(roc_auc_score(labels[held], fp[held]))}])
    primary = 'QWK' if dataset == 'sicapv2' else 'mean_fold_patient_AUROC'
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in output:
        if row['component'] == 'seed' and primary in row:
            grouped[str(row['encoder']), str(row['arm_id'])].append(float(row[primary]))
    summary = []
    for (encoder, arm), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f'five-seed validation component coverage mismatch: {encoder}/{arm}')
        summary.append({'encoder': encoder, 'arm_id': arm, 'metric': primary, 'seed_values': values, 'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)), 'seeds': list(SEEDS)})
    return (output, summary)

def _qwk(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(labels, scores.argmax(1), weights='quadratic'))

def _sicap_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    predicted = probability.argmax(1)
    return {'patches': int(len(labels)), 'QWK': _qwk(labels, probability), 'accuracy': float(accuracy_score(labels, predicted)), 'macro_F1': float(f1_score(labels, predicted, average='macro', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)), 'per_class_F1': f1_score(labels, predicted, labels=[0, 1, 2, 3], average=None, zero_division=0).tolist(), 'per_class_AUROC': [float(roc_auc_score(labels == index, probability[:, index])) for index in range(4)], 'confusion_matrix': confusion_matrix(labels, predicted, labels=[0, 1, 2, 3]).tolist()}

def _mean_fold_auc(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> tuple[float, dict[int, float]]:
    from sklearn.metrics import roc_auc_score
    values = {fold: float(roc_auc_score(labels[folds == fold], scores[folds == fold])) for fold in sorted(set(map(int, folds)))}
    return (float(np.mean(list(values.values()))), values)

def _crc_metrics(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    fold_auc, fold_auprc = ({}, {})
    prediction = np.zeros(len(labels), dtype=np.int64)
    thresholds = {}
    for fold in sorted(set(map(int, folds))):
        held, fit = (folds == fold, folds != fold)
        fold_auc[fold] = float(roc_auc_score(labels[held], scores[held]))
        fold_auprc[fold] = float(average_precision_score(labels[held], scores[held]))
        candidates = np.unique(scores[fit])
        best_threshold, best_value = (0.5, -np.inf)
        for threshold in candidates:
            value = balanced_accuracy_score(labels[fit], scores[fit] >= threshold)
            if value > best_value + 1e-12:
                best_value, best_threshold = (float(value), float(threshold))
        thresholds[fold] = best_threshold
        prediction[held] = scores[held] >= best_threshold
    positive, negative = (labels == 1, labels == 0)
    return {'patients': int(len(labels)), 'mean_fold_patient_AUROC': float(np.mean(list(fold_auc.values()))), 'mean_fold_patient_AUPRC': float(np.mean(list(fold_auprc.values()))), 'balanced_accuracy': float(balanced_accuracy_score(labels, prediction)), 'sensitivity': float(prediction[positive].mean()), 'specificity': float((prediction[negative] == 0).mean()), 'fold_AUROC': {str(key): value for key, value in fold_auc.items()}, 'fold_AUPRC': {str(key): value for key, value in fold_auprc.items()}, 'cross_fitted_threshold': {str(key): value for key, value in thresholds.items()}}

def _geometry_eligible_ids(cfg: Mapping[str, Any]) -> set[str]:
    rows = _rows(Path(cfg['paths']['model_input_root']) / '03_graph_cache' / 'graph_index.parquet')
    return {str(row['graph_id']) for row in rows if str(row['split']).lower() == 'train'}

def _crc_paper_patients(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row['fold']), str(row['patient_id'])].append(row)
    output = []
    for (fold, patient), values in sorted(groups.items()):
        if len(values) < 10:
            continue
        positive = sum((float(row['probabilities'][1]) >= 0.5 for row in values))
        total = len(values)
        output.append({'fold': fold, 'patient_id': patient, 'label_id': int(values[0]['label_id']), 'score': positive / total, 'logit': math.log((positive + 0.5) / (total - positive + 0.5)), 'tiles': total})
    return output

def _crc_endpoint_geometry_extras(endpoint_keys: Sequence[tuple[int, str]], geometry_keys: Sequence[tuple[int, str]], tile_counts: Mapping[tuple[int, str], int]) -> set[tuple[int, str]]:
    endpoint, geometry = (set(endpoint_keys), set(geometry_keys))
    missing, extra = (endpoint - geometry, geometry - endpoint)
    if missing:
        raise RuntimeError(f'paper endpoint patients missing geometry: {len(missing)}')
    if any((tile_counts.get(key, 0) >= 10 for key in extra)):
        raise RuntimeError('geometry extra patient is not explained by the <10-tile endpoint rule')
    return extra

def evaluate_validation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    from .qc import compute_ncr_qc
    ncr_qc = compute_ncr_qc(cfg, include_test=False)
    paper = _paper_seed_average(cfg)
    result_root = Path(cfg['paths']['result_root'])
    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    if dataset == 'sicapv2':
        paper_map = {(int(row['fold']), str(row['graph_id'])): row for row in paper}
        keys = sorted(paper_map)
        labels = np.asarray([paper_map[key]['label_id'] for key in keys])
        folds = np.asarray([key[0] for key in keys])
        rgb_probability = np.stack([paper_map[key]['probabilities'] for key in keys])
        rgb_logits = np.log(np.clip(rgb_probability, 1e-07, 1.0))
        metrics.append({'encoder': 'paper', 'arm_id': 'R0', **_sicap_metrics(labels, rgb_probability)})
        for key, label, probability in zip(keys, labels, rgb_probability):
            prediction_rows.append({'encoder': 'paper', 'arm_id': 'R0', 'fold': key[0], 'graph_id': key[1], 'patient_id': paper_map[key]['patient_id'], 'label_id': int(label), 'probabilities': probability.tolist()})
        for encoder in ENCODERS:
            for o_arm in [key for key in ARMS if key.startswith('O')]:
                geometry = _geometry_seed_average(cfg, encoder, o_arm)
                geometry_map = {(int(row['fold']), str(row['key'])): row for row in geometry}
                if set(geometry_map) != set(keys):
                    raise RuntimeError(f'paper/geometry join mismatch: {encoder}/{o_arm}')
                geometry_probability = np.stack([geometry_map[key]['probability'] for key in keys])
                geometry_logits = np.log(np.clip(geometry_probability, 1e-07, 1.0))
                metrics.append({'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels, geometry_probability)})
                for key, label, probability in zip(keys, labels, geometry_probability):
                    prediction_rows.append({'encoder': encoder, 'arm_id': o_arm, 'fold': key[0], 'graph_id': key[1], 'patient_id': paper_map[key]['patient_id'], 'label_id': int(label), 'probabilities': probability.tolist()})
                r_arm = 'R' + o_arm[1:]
                fused_logits, calibrations = cross_fit_sicap(folds, rgb_logits, geometry_logits, labels)
                fused_probability = np.exp(fused_logits - fused_logits.max(1, keepdims=True))
                fused_probability /= fused_probability.sum(1, keepdims=True)
                metrics.append({'encoder': encoder, 'arm_id': r_arm, **_sicap_metrics(labels, fused_probability), 'calibrations_json': str({str(fold): calibration.__dict__ for fold, calibration in calibrations.items()})})
                for key, label, probability in zip(keys, labels, fused_probability):
                    prediction_rows.append({'encoder': encoder, 'arm_id': r_arm, 'fold': key[0], 'graph_id': key[1], 'patient_id': paper_map[key]['patient_id'], 'label_id': int(label), 'probabilities': probability.tolist()})
    else:
        rgb_patients = _crc_paper_patients(paper)
        rgb_map = {(int(row['fold']), str(row['patient_id'])): row for row in rgb_patients}
        r0_keys = sorted(rgb_map)
        r0_labels = np.asarray([rgb_map[key]['label_id'] for key in r0_keys])
        r0_folds = np.asarray([key[0] for key in r0_keys])
        r0_score = np.asarray([rgb_map[key]['score'] for key in r0_keys])
        metrics.append({'encoder': 'paper', 'arm_id': 'R0', 'tile_universe': 'all_source', **_crc_metrics(r0_labels, r0_score, r0_folds)})
        for key, label, score in zip(r0_keys, r0_labels, r0_score):
            prediction_rows.append({'encoder': 'paper', 'arm_id': 'R0', 'fold': key[0], 'patient_id': key[1], 'label_id': int(label), 'score': float(score), 'tiles': int(rgb_map[key]['tiles']), 'tile_universe': 'all_source'})
        eligible = _geometry_eligible_ids(cfg)
        common_paper = [row for row in paper if str(row['graph_id']) in eligible]
        common_tile_counts = Counter(((int(row['fold']), str(row['patient_id'])) for row in common_paper))
        common_rgb_map = {(int(row['fold']), str(row['patient_id'])): row for row in _crc_paper_patients(common_paper)}
        keys = sorted(common_rgb_map)
        labels = np.asarray([common_rgb_map[key]['label_id'] for key in keys])
        folds = np.asarray([key[0] for key in keys])
        rgb_logits = np.asarray([common_rgb_map[key]['logit'] for key in keys])
        for encoder in ENCODERS:
            for o_arm in [key for key in ARMS if key.startswith('O')]:
                geometry = _geometry_seed_average(cfg, encoder, o_arm)
                geometry_map = {(int(row['fold']), str(row['patient_id'])): row for row in geometry}
                extra = _crc_endpoint_geometry_extras(keys, list(geometry_map), common_tile_counts)
                geometry_score = np.asarray([geometry_map[key]['probability'] for key in keys])
                geometry_logits = probability_logit(geometry_score)
                metrics.append({'encoder': encoder, 'arm_id': o_arm, 'tile_universe': 'geometry_eligible', 'geometry_patients_excluded_under_10_tiles': len(extra), **_crc_metrics(labels, geometry_score, folds)})
                for key, label, score in zip(keys, labels, geometry_score):
                    prediction_rows.append({'encoder': encoder, 'arm_id': o_arm, 'fold': key[0], 'patient_id': key[1], 'label_id': int(label), 'score': float(score), 'tile_universe': 'geometry_eligible'})
                r_arm = 'R' + o_arm[1:]
                fused_logits, calibrations = cross_fit_crc(folds, rgb_logits, geometry_logits, labels)
                fused_score = 1.0 / (1.0 + np.exp(-fused_logits))
                metrics.append({'encoder': encoder, 'arm_id': r_arm, 'tile_universe': 'geometry_eligible', 'geometry_patients_excluded_under_10_tiles': len(extra), **_crc_metrics(labels, fused_score, folds), 'calibrations_json': str({str(fold): calibration.__dict__ for fold, calibration in calibrations.items()})})
                for key, label, score in zip(keys, labels, fused_score):
                    prediction_rows.append({'encoder': encoder, 'arm_id': r_arm, 'fold': key[0], 'patient_id': key[1], 'label_id': int(label), 'score': float(score), 'tiles': int(common_rgb_map[key]['tiles']), 'tile_universe': 'geometry_eligible'})
    output = result_root / 'metrics' / 'validation'
    atomic_parquet(output / 'metrics.parquet', metrics)
    atomic_parquet(output / 'fused_predictions.parquet', prediction_rows)
    from .validation_statistics import compute_validation_statistics
    statistics = compute_validation_statistics(cfg, prediction_rows)
    components, seed_summary = _validation_component_metrics(cfg)
    atomic_parquet(output / 'component_metrics.parquet', components)
    atomic_parquet(output / 'seed_mean_std.parquet', seed_summary)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'route': 'patch_scalar' if dataset == 'sicapv2' else 'patient_temperature_scalar', 'metrics': metrics, 'statistics': statistics, 'official_test_touched': False, 'component_metrics_rows': len(components), 'seed_summary_rows': len(seed_summary), 'ncr_qc': ncr_qc}
    atomic_json(output / 'summary.json', payload)
    if dataset == 'tcga_crc_msi':
        payload['tile_route'] = evaluate_tile_validation(cfg)
    return payload

def _tile_geometry_seed_average(cfg: Mapping[str, Any], encoder: str, arm_id: str) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'tile_fusion' / 'experts' / encoder / arm_id
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for fold in range(int(cfg['split']['validation_folds'])):
        for seed in SEEDS:
            path = root / f'seed_{seed}' / f'fold_{fold}' / 'validation_tile_predictions.parquet'
            for row in _rows(path):
                grouped[fold, str(row['graph_id'])].append(row)
    output = []
    for (fold, graph_id), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f'tile geometry seed coverage mismatch: {encoder}/{arm_id}/{fold}/{graph_id}')
        output.append({'fold': fold, 'graph_id': graph_id, 'patient_id': str(values[0]['patient_id']), 'label_id': int(values[0]['label_id']), 'score': float(np.mean([float(row['score']) for row in values]))})
    return output

def _tile_calibration_weight(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    patient_count = defaultdict(int)
    class_patients: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        patient_count[str(row['patient_id'])] += 1
        class_patients[int(row['label_id'])].add(str(row['patient_id']))
    class_weight = {label: 1.0 / max(1, len(patients)) for label, patients in class_patients.items()}
    value = np.asarray([class_weight[int(row['label_id'])] / patient_count[str(row['patient_id'])] for row in rows], np.float64)
    return value / value.mean()

def evaluate_tile_validation(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if cfg['dataset'] != 'tcga_crc_msi':
        raise RuntimeError('tile fusion is CRC-only')
    eligible = _geometry_eligible_ids(cfg)
    paper = [row for row in _paper_seed_average(cfg) if str(row['graph_id']) in eligible]
    paper_map = {(int(row['fold']), str(row['graph_id'])): row for row in paper}
    keys = sorted(paper_map)
    labels = np.asarray([paper_map[key]['label_id'] for key in keys])
    folds = np.asarray([key[0] for key in keys])
    patients = np.asarray([paper_map[key]['patient_id'] for key in keys], object)
    rgb_probability = np.asarray([paper_map[key]['probabilities'][1] for key in keys])
    rgb_logits = probability_logit(rgb_probability)
    result_root = Path(cfg['paths']['result_root'])
    metrics, predictions = ([], [])
    paper_patient_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, (fold, patient) in enumerate(zip(folds, patients)):
        paper_patient_groups[int(fold), str(patient)].append(index)
    paper_patient_rows = []
    for (fold, patient), indices in sorted(paper_patient_groups.items()):
        if len(indices) < 10:
            continue
        paper_patient_rows.append({'fold': fold, 'patient_id': patient, 'label_id': int(labels[indices[0]]), 'score': float(np.mean(rgb_probability[indices] >= 0.5)), 'tiles': len(indices)})
    paper_patient_labels = np.asarray([row['label_id'] for row in paper_patient_rows])
    paper_patient_scores = np.asarray([row['score'] for row in paper_patient_rows])
    paper_patient_folds = np.asarray([row['fold'] for row in paper_patient_rows])
    metrics.append({'encoder': 'paper', 'arm_id': 'R0', 'route': 'tile_geometry_eligible_reference', **_crc_metrics(paper_patient_labels, paper_patient_scores, paper_patient_folds)})
    predictions.extend(({'encoder': 'paper', 'arm_id': 'R0', **row} for row in paper_patient_rows))
    for encoder in ENCODERS:
        for o_arm in [key for key in ARMS if key.startswith('O')]:
            geometry = _tile_geometry_seed_average(cfg, encoder, o_arm)
            geometry_map = {(int(row['fold']), str(row['graph_id'])): row for row in geometry}
            if set(geometry_map) != set(keys):
                raise RuntimeError(f'paper/tile-geometry join mismatch: {encoder}/{o_arm}')
            geometry_logits = probability_logit(np.asarray([geometry_map[key]['score'] for key in keys]))
            fused_logits = np.empty_like(rgb_logits)
            calibrations = {}
            for fold in sorted(set(map(int, folds))):
                held = folds == fold
                fit_rows = [{'patient_id': patients[index], 'label_id': int(labels[index])} for index in np.flatnonzero(~held)]
                calibration = fit_crc_calibration(rgb_logits[~held], geometry_logits[~held], labels[~held], sample_weight=_tile_calibration_weight(fit_rows))
                fused_logits[held] = apply_crc(rgb_logits[held], geometry_logits[held], calibration)
                calibrations[str(fold)] = calibration.__dict__
            positive = fused_logits >= 0
            patient_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
            for index, (fold, patient) in enumerate(zip(folds, patients)):
                patient_groups[int(fold), str(patient)].append(index)
            patient_rows = []
            for (fold, patient), indices in sorted(patient_groups.items()):
                if len(indices) < 10:
                    continue
                patient_rows.append({'fold': fold, 'patient_id': patient, 'label_id': int(labels[indices[0]]), 'score': float(np.mean(positive[indices])), 'tiles': len(indices)})
            patient_labels = np.asarray([row['label_id'] for row in patient_rows])
            patient_scores = np.asarray([row['score'] for row in patient_rows])
            patient_folds = np.asarray([row['fold'] for row in patient_rows])
            r_arm = 'R' + o_arm[1:]
            metrics.append({'encoder': encoder, 'arm_id': r_arm, **_crc_metrics(patient_labels, patient_scores, patient_folds), 'calibrations_json': str(calibrations)})
            predictions.extend(({'encoder': encoder, 'arm_id': r_arm, **row} for row in patient_rows))
    output = result_root / 'metrics' / 'validation_tile'
    atomic_parquet(output / 'metrics.parquet', metrics)
    atomic_parquet(output / 'predictions.parquet', predictions)
    from .validation_statistics import compute_crc_tile_validation_statistics
    statistics = compute_crc_tile_validation_statistics(cfg, predictions)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': 'tcga_crc_msi', 'route': 'tile_temperature_scalar_then_patient_hard_vote', 'patient_normalized_class_balanced_calibration': True, 'metrics': metrics, 'statistics': statistics, 'official_test_touched': False}
    atomic_json(output / 'summary.json', payload)
    return payload
