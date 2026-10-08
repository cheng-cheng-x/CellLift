from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.conditional_geometry.late_fusion_protocol import BASE_ARM, GEOMETRY_ARMS, GEOMETRY_PROTOCOL_ID, OFFICIAL_TEST_ALLOWED, PRIMARY_WEIGHT, PROTOCOL_ID, RANK_PROTOCOL_ID, RGB_MASK_PROTOCOL_ID, SENSITIVITY_WEIGHTS
from celllift.conditional_geometry.scripts.evaluate import _auroc, _metric, _paired_bootstrap, _paired_fold_stratified_crc_bootstrap

@dataclass(frozen=True)
class PredictionBatch:
    identifiers: np.ndarray
    patients: np.ndarray
    truth: np.ndarray
    values: np.ndarray

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)
    return _sha256(path)

def _job_dir(root: Path, encoder: str, arm: str, seed: int, fold: int) -> Path:
    return root / 'screen' / encoder / arm / f'seed_{seed}' / f'fold_{fold}'

def _read_source(root: Path, dataset: str, encoder: str, arm: str, seed: int, fold: int, *, protocol_id: str, use_rgb: bool) -> tuple[PredictionBatch, dict[str, Any]]:
    import pyarrow.parquet as pq
    directory = _job_dir(root, encoder, arm, seed, fold)
    job_path = directory / 'job.json'
    prediction_path = directory / 'validation_predictions.parquet'
    if not job_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f'incomplete source job: {directory}')
    job = json.loads(job_path.read_text(encoding='utf-8'))
    expected = {'status': 'PASS', 'protocol_id': protocol_id, 'dataset': dataset, 'encoder': encoder, 'seed': seed, 'fold': fold}
    for key, value in expected.items():
        if job.get(key) != value:
            raise RuntimeError(f'source job {key} mismatch at {job_path}: {job.get(key)!r} != {value!r}')
    arm_payload = job.get('arm', {})
    if arm_payload.get('arm_id') != arm or bool(arm_payload.get('use_rgb')) != use_rgb:
        raise RuntimeError(f'source arm binding mismatch at {job_path}')
    recorded = job.get('outputs', {}).get('predictions', {}).get('sha256')
    actual = _sha256(prediction_path)
    if recorded != actual:
        raise RuntimeError(f'source prediction checksum mismatch at {prediction_path}')
    rows = pq.read_table(prediction_path, partitioning=None).to_pylist()
    if not rows:
        raise RuntimeError(f'empty source prediction: {prediction_path}')
    id_key = 'graph_id' if dataset == 'sicapv2' else 'patient_id'
    order = np.argsort(np.asarray([str(row[id_key]) for row in rows], object), kind='stable')
    identifiers = np.asarray([str(rows[index][id_key]) for index in order], object)
    if len(set(map(str, identifiers))) != len(identifiers):
        raise RuntimeError(f'duplicate source identifiers at {prediction_path}')
    patients = np.asarray([str(rows[index]['patient_id']) for index in order], object)
    truth = np.asarray([int(rows[index]['y_true']) for index in order], np.int64)
    if dataset == 'sicapv2':
        values = np.asarray([[float(rows[index][f'prob_{klass}']) for klass in range(4)] for index in order], np.float64)
        if values.shape != (len(rows), 4) or not np.allclose(values.sum(1), 1.0, atol=1e-05):
            raise RuntimeError(f'invalid SICAP probabilities at {prediction_path}')
    else:
        values = np.asarray([float(rows[index]['score']) for index in order], np.float64)
        if np.any((values < 0) | (values > 1)):
            raise RuntimeError(f'invalid CRC probabilities at {prediction_path}')
    provenance = {'job': str(job_path), 'job_sha256': _sha256(job_path), 'prediction': str(prediction_path), 'prediction_sha256': actual}
    return (PredictionBatch(identifiers, patients, truth, values), provenance)

def _aligned(reference: PredictionBatch, candidate: PredictionBatch) -> None:
    if not np.array_equal(reference.identifiers, candidate.identifiers):
        raise RuntimeError('late-fusion source identifiers are not aligned')
    if not np.array_equal(reference.patients, candidate.patients):
        raise RuntimeError('late-fusion source patients are not aligned')
    if not np.array_equal(reference.truth, candidate.truth):
        raise RuntimeError('late-fusion source labels are not aligned')

def convex_fusion(base: np.ndarray, geometry: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= float(weight) <= 0.5:
        raise ValueError('late-fusion geometry weight must be in [0, 0.5]')
    if base.shape != geometry.shape:
        raise ValueError('late-fusion probability shapes differ')
    fused = (1.0 - float(weight)) * np.asarray(base, np.float64) + float(weight) * np.asarray(geometry, np.float64)
    if not np.all(np.isfinite(fused)) or np.any((fused < 0) | (fused > 1)):
        raise RuntimeError('late-fusion probabilities are invalid')
    return fused

def _midrank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not len(values) or (not np.all(np.isfinite(values))):
        raise ValueError('rank fusion requires one finite score per patient')
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (0.5 * (start + stop - 1) + 0.5) / len(values)
        start = stop
    return ranks

def rank_convex_fusion(base: np.ndarray, geometry: np.ndarray, weight: float) -> np.ndarray:
    if base.ndim != 1 or geometry.ndim != 1:
        raise ValueError('rank fusion is defined only for binary patient scores')
    return convex_fusion(_midrank_percentile(base), _midrank_percentile(geometry), weight)

def _prediction_rows(dataset: str, batch: PredictionBatch, values: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, identifier in enumerate(batch.identifiers):
        row = {'sample_id': str(identifier), 'patient_id': str(batch.patients[index]), 'y_true': int(batch.truth[index])}
        if dataset == 'sicapv2':
            row['graph_id'] = str(identifier)
            row.update({f'prob_{klass}': float(values[index, klass]) for klass in range(4)})
        else:
            row['score'] = float(values[index])
        rows.append(row)
    return rows

def _ensemble_sicap(batches: Mapping[tuple[int, int], tuple[PredictionBatch, np.ndarray]], seeds: Sequence[int], folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_seed = {}
    for seed in seeds:
        combined: dict[str, tuple[str, int, np.ndarray]] = {}
        for fold in range(folds):
            batch, values = batches[int(seed), fold]
            for index, identifier in enumerate(batch.identifiers):
                key = str(identifier)
                if key in combined:
                    raise RuntimeError(f'duplicate SICAP OOF graph {key}')
                combined[key] = (str(batch.patients[index]), int(batch.truth[index]), values[index])
        by_seed[int(seed)] = combined
    identifiers = sorted(next(iter(by_seed.values())))
    if any((sorted(values) != identifiers for values in by_seed.values())):
        raise RuntimeError('SICAP late-fusion seed samples differ')
    patients = np.asarray([by_seed[int(seeds[0])][key][0] for key in identifiers], object)
    truth = np.asarray([by_seed[int(seeds[0])][key][1] for key in identifiers], np.int64)
    stack = np.stack([[by_seed[int(seed)][key][2] for key in identifiers] for seed in seeds])
    return (patients, truth, stack.mean(0))

def _ensemble_crc(batches: Mapping[tuple[int, int], tuple[PredictionBatch, np.ndarray]], seeds: Sequence[int], folds: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output = []
    for fold in range(folds):
        reference = batches[int(seeds[0]), fold][0]
        stack = []
        for seed in seeds:
            batch, values = batches[int(seed), fold]
            _aligned(reference, batch)
            stack.append(values)
        output.append((reference.patients, reference.truth, np.stack(stack).mean(0)))
    return output

def _holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (float(pvalues[key]), key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted

def _weight_key(weight: float) -> str:
    return f'w{int(round(100 * weight)):02d}'

def evaluate_complete_geometry_late_fusion(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if OFFICIAL_TEST_ALLOWED or cfg.get('split', {}).get('official_test_frozen') is not True:
        raise RuntimeError('late_geometry_fusion is validation-only')
    configured_protocol = str(cfg.get('protocol_id'))
    if configured_protocol not in {PROTOCOL_ID, RANK_PROTOCOL_ID}:
        raise RuntimeError('late-fusion protocol mismatch')
    dataset = str(cfg['dataset'])
    if dataset not in {'sicapv2', 'tcga_crc_msi'}:
        raise ValueError('unsupported late-fusion dataset')
    fusion_mode = str(cfg['fusion'].get('mode', 'probability'))
    expected_mode = 'rank' if configured_protocol == RANK_PROTOCOL_ID else 'probability'
    if fusion_mode != expected_mode:
        raise RuntimeError('late-fusion mode/protocol mismatch')
    if fusion_mode == 'rank' and dataset != 'tcga_crc_msi':
        raise RuntimeError('rank late fusion is registered only for CRC patient AUROC')
    fusion_function = rank_convex_fusion if fusion_mode == 'rank' else convex_fusion
    paths = cfg['paths']
    base_root = Path(paths['rgb_mask_result_root'])
    geometry_root = Path(paths['geometry_result_root'])
    result_root = Path(paths['result_root'])
    seeds = tuple(map(int, cfg['fusion']['seeds']))
    encoders = tuple(map(str, cfg['fusion']['encoders']))
    folds = int(cfg['split']['validation_folds'])
    primary_weight = float(cfg['fusion'].get('primary_weight', PRIMARY_WEIGHT))
    weights = tuple(map(float, cfg['fusion'].get('sensitivity_weights', SENSITIVITY_WEIGHTS)))
    if primary_weight != PRIMARY_WEIGHT or weights != SENSITIVITY_WEIGHTS:
        raise RuntimeError('late_geometry_fusion fusion weights differ from the locked protocol')
    replicates = int(cfg['runtime']['bootstrap_replicates'])
    source_provenance: dict[str, Any] = {}
    result: dict[str, Any] = {}
    for encoder in encoders:
        loaded: dict[tuple[int, int], tuple[PredictionBatch, dict[str, PredictionBatch]]] = {}
        for seed in seeds:
            for fold in range(folds):
                base, base_provenance = _read_source(base_root, dataset, encoder, BASE_ARM, seed, fold, protocol_id=RGB_MASK_PROTOCOL_ID, use_rgb=True)
                geometries = {}
                provenance = {'rgb_mask_base': base_provenance, 'geometry': {}}
                for arm in GEOMETRY_ARMS:
                    geometry, geometry_provenance = _read_source(geometry_root, dataset, encoder, arm, seed, fold, protocol_id=GEOMETRY_PROTOCOL_ID, use_rgb=False)
                    _aligned(base, geometry)
                    geometries[arm] = geometry
                    provenance['geometry'][arm] = geometry_provenance
                loaded[seed, fold] = (base, geometries)
                source_provenance[f'{encoder}/seed_{seed}/fold_{fold}'] = provenance
        sensitivity = {}
        primary_batches = {}
        primary_metrics = {}
        for weight in weights:
            weight_metrics = {}
            for arm in GEOMETRY_ARMS:
                batches = {}
                seed_fold_metrics = {}
                for seed in seeds:
                    for fold in range(folds):
                        base, geometries = loaded[seed, fold]
                        fused = fusion_function(base.values, geometries[arm].values, weight)
                        batches[seed, fold] = (base, fused)
                        seed_fold_metrics[f'seed_{seed}/fold_{fold}'] = _metric(dataset, base.truth, fused)
                        if weight == primary_weight:
                            path = result_root / 'predictions' / 'primary' / encoder / arm / f'seed_{seed}' / f'fold_{fold}.parquet'
                            checksum = _atomic_parquet(path, _prediction_rows(dataset, base, fused))
                            source_provenance[f'{encoder}/seed_{seed}/fold_{fold}'][f'fused_{arm}'] = {'path': str(path), 'sha256': checksum, 'weight': weight}
                if dataset == 'sicapv2':
                    patient, truth, ensemble = _ensemble_sicap(batches, seeds, folds)
                    metric_value = _metric(dataset, truth, ensemble)
                    aggregate = (patient, truth, ensemble)
                else:
                    ensemble_folds = _ensemble_crc(batches, seeds, folds)
                    metric_value = float(np.mean([_auroc(truth, score) for _, truth, score in ensemble_folds]))
                    aggregate = ensemble_folds
                weight_metrics[arm] = {'metric': float(metric_value), 'seed_fold_metrics': seed_fold_metrics}
                if weight == primary_weight:
                    primary_batches[arm] = aggregate
                    primary_metrics[arm] = float(metric_value)
            sensitivity[_weight_key(weight)] = {'geometry_weight': weight, 'metrics': {arm: values['metric'] for arm, values in weight_metrics.items()}}
        zero_weight_metric = sensitivity['w00']['metrics'][GEOMETRY_ARMS[0]]
        if dataset == 'sicapv2':
            patient, truth, real = primary_batches['M3_MASK_RES3D']
            _, _, geometry_2d = primary_batches['M3_MASK2D']
            _, _, shuffled = primary_batches['M3_MASK_SHUF_RES3D']
            base_values = _ensemble_sicap({key: (value[0], value[0].values) for key, value in loaded.items()}, seeds, folds)[2]
            raw_base_metric = _metric(dataset, truth, base_values)
            comparisons = {'real3d_vs_rgb_mask_base': _paired_bootstrap(dataset, patient, truth, real, base_values, replicates), 'real3d_vs_2d_fusion': _paired_bootstrap(dataset, patient, truth, real, geometry_2d, replicates), 'real3d_vs_shuffled3d_fusion': _paired_bootstrap(dataset, patient, truth, real, shuffled, replicates), '2d_fusion_vs_rgb_mask_base': _paired_bootstrap(dataset, patient, truth, geometry_2d, base_values, replicates)}
        else:
            real = primary_batches['M3_MASK_RES3D']
            geometry_2d = primary_batches['M3_MASK2D']
            shuffled = primary_batches['M3_MASK_SHUF_RES3D']
            base_folds = _ensemble_crc({key: (value[0], value[0].values) for key, value in loaded.items()}, seeds, folds)
            raw_base_metric = float(np.mean([_auroc(y, score) for _, y, score in base_folds]))
            comparisons = {'real3d_vs_rgb_mask_base': _paired_fold_stratified_crc_bootstrap(real, base_folds, replicates), 'real3d_vs_2d_fusion': _paired_fold_stratified_crc_bootstrap(real, geometry_2d, replicates), 'real3d_vs_shuffled3d_fusion': _paired_fold_stratified_crc_bootstrap(real, shuffled, replicates), '2d_fusion_vs_rgb_mask_base': _paired_fold_stratified_crc_bootstrap(geometry_2d, base_folds, replicates)}
        adjusted = _holm_adjust({key: value['one_sided_p'] for key, value in comparisons.items()})
        for key, value in comparisons.items():
            value['holm_adjusted_one_sided_p'] = adjusted[key]
            value['holm_positive'] = bool(value['positive_ci_lower_gt_zero'] and adjusted[key] < 0.05)
        result[encoder] = {'metric': 'QWK' if dataset == 'sicapv2' else 'mean_fold_patient_AUROC', 'rgb_mask_base_metric': float(raw_base_metric), 'zero_weight_metric': float(zero_weight_metric), 'primary_weight': primary_weight, 'primary_metrics': primary_metrics, 'primary_comparisons': comparisons, 'sensitivity': sensitivity}
    payload = {'status': 'PASS', 'protocol_id': configured_protocol, 'dataset': dataset, 'official_test_touched': False, 'fusion': {'mode': fusion_mode, 'formula': '(1-weight)*rank(RGB_MASK2D_score) + weight*rank(complete_geometry_expert_score)' if fusion_mode == 'rank' else '(1-weight)*RGB_MASK2D_probability + weight*complete_geometry_expert_probability', 'primary_weight': primary_weight, 'sensitivity_weights': list(weights), 'selection': 'primary weight frozen before any rank_fusion metric; sweep is descriptive only' if configured_protocol == RANK_PROTOCOL_ID else 'primary weight frozen before any late_geometry_fusion metric; sweep is descriptive only'}, 'results': result, 'source_provenance': source_provenance}
    _atomic_json(result_root / 'validation_summary.json', payload)
    return payload
