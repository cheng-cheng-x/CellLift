from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.experiment import PRIMARY_CONTRAST, SECONDARY_CONTRASTS
from celllift.set_encoding.tasks.metrics import binary_auprc, binary_auroc, binary_metrics, multiclass_metrics, quadratic_weighted_kappa
from celllift.set_encoding.tasks.statistics import holm_adjust, paired_cluster_bootstrap, paired_delong
from celllift.set_encoding.tasks.crc import historical_tile_hard_vote
from celllift.set_encoding.scripts.aggregate_results import aggregate_official_test, aggregate_phase, choose_screening_encoders, freeze_selection_after_confirmation

def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, path)

def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')

def _load_prediction(path: Path, dataset: str) -> dict[str, Any]:
    import pyarrow.parquet as pq
    rows = pq.read_table(path, partitioning=None).to_pylist()
    if not rows:
        raise RuntimeError(f'empty prediction table {path}')
    required = {'sample_id', 'patient_id', 'y_true'}
    if any((not required.issubset(row) for row in rows)):
        raise RuntimeError(f'prediction schema is incomplete: {path}')
    result: dict[str, Any] = {'sample_id': np.asarray([str(row['sample_id']) for row in rows]), 'patient_id': np.asarray([str(row['patient_id']) for row in rows]), 'y_true': np.asarray([int(row['y_true']) for row in rows])}
    if dataset == 'sicapv2':
        result['prediction'] = np.asarray([[float(row[f'prob_{index}']) for index in range(4)] for row in rows])
        if all(('cribriform_score' in row and 'g4c_valid' in row for row in rows)):
            result['cribriform_score'] = np.asarray([float(row['cribriform_score']) for row in rows])
            result['g4c_valid'] = np.asarray([bool(row['g4c_valid']) for row in rows])
            result['g4c_label'] = np.asarray([int(row['g4c_label']) if row.get('g4c_valid') else -1 for row in rows])
    else:
        result['prediction'] = np.asarray([float(row['score']) for row in rows])
        result['threshold'] = float(rows[0].get('validation_threshold', 0.5))
    return result

def _aligned(first: dict[str, Any], second: dict[str, Any]) -> None:
    for key in ('sample_id', 'patient_id', 'y_true'):
        if not np.array_equal(first[key], second[key]):
            raise RuntimeError(f'paired predictions are not aligned on {key}')

def _apply_hierarchical_inference(contrasts: dict[str, Any], primary_key: str, secondary_keys: list[str]) -> None:
    if secondary_keys:
        adjusted = holm_adjust({name: contrasts[name]['p_value'] for name in secondary_keys})
        for name in secondary_keys:
            contrasts[name]['holm'] = adjusted[name]
    primary = contrasts.get(primary_key)
    if primary is None:
        return
    primary['positive'] = bool(primary['ci_low'] > 0 and primary['p_value'] < 0.05)
    reached = bool(primary['positive'])
    for name in secondary_keys:
        record = contrasts[name]
        record['hierarchy_reached'] = reached
        candidate_positive = bool(record['ci_low'] > 0 and record['holm']['reject'])
        record['positive'] = bool(reached and candidate_positive)
        reached = bool(record['positive'])

def summarize_ncr_qc(cfg: dict[str, Any], *, official_test: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq
    data_root = Path(cfg['paths']['data_root'])
    model_input = Path(cfg['paths']['model_input_root'])
    graph_rows = pq.read_table(model_input / '03_graph_cache' / 'graph_index.parquet', partitioning=None).to_pylist()
    wanted = 'test' if official_test else 'train'
    metadata = {str(row['graph_id']): row for row in graph_rows if str(row.get('split', row.get('official_split', ''))).lower() == wanted}
    if not metadata:
        raise RuntimeError(f'NCR QC found no {wanted} graphs')
    summary: dict[str, dict[str, Any]] = {}
    for path in sorted((data_root / '04_ncr_features').glob('*.parquet')):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=['graph_id', 'valid_ncr_2d', 'ncr_reason_2d', 'valid_ncr_3d', 'ncr_reason_3d']):
            for row in batch.to_pylist():
                graph_id = str(row['graph_id'])
                meta = metadata.get(graph_id)
                if meta is None:
                    continue
                label = str(meta.get('label_id', meta.get('label', meta.get('target', 'unknown'))))
                key = f'{wanted}/label_{label}'
                record = summary.setdefault(key, {'split': wanted, 'label': label, 'anchors': 0, 'invalid_ncr_2d': 0, 'invalid_ncr_3d': 0, 'invalid_reasons_2d': {}, 'invalid_reasons_3d': {}})
                record['anchors'] += 1
                for mode in ('2d', '3d'):
                    if not bool(row[f'valid_ncr_{mode}']):
                        record[f'invalid_ncr_{mode}'] += 1
                        reason = str(row[f'ncr_reason_{mode}'])
                        reasons = record[f'invalid_reasons_{mode}']
                        reasons[reason] = reasons.get(reason, 0) + 1
    if not summary:
        raise RuntimeError('NCR QC did not join any cached anchor')
    total = {'anchors': sum((row['anchors'] for row in summary.values())), 'invalid_ncr_2d': sum((row['invalid_ncr_2d'] for row in summary.values())), 'invalid_ncr_3d': sum((row['invalid_ncr_3d'] for row in summary.values()))}
    for row in summary.values():
        row['invalid_fraction_2d'] = row['invalid_ncr_2d'] / row['anchors']
        row['invalid_fraction_3d'] = row['invalid_ncr_3d'] / row['anchors']
    total['invalid_fraction_2d'] = total['invalid_ncr_2d'] / total['anchors']
    total['invalid_fraction_3d'] = total['invalid_ncr_3d'] / total['anchors']
    return {'split': wanted, 'total': total, 'groups': summary}

def summarize_historical_rgb(cfg: dict[str, Any], *, official_test: bool) -> dict[str, Any] | None:
    if cfg['dataset'] != 'tcga_crc_msi':
        return None
    import pyarrow.parquet as pq
    data_root = Path(cfg['paths']['data_root'])
    expected_folds = set(range(int(cfg['split']['validation_folds'])))
    manifests: dict[int, dict[str, Any]] = {}
    for path in sorted((data_root / '05_rgb_features').glob('fold_*/seed_42/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        fold = int(manifest['fold'])
        if manifest.get('status') == 'PASS' and int(manifest['seed']) == 42:
            manifests[fold] = manifest
    if set(manifests) != expected_folds:
        raise RuntimeError(f'historical RGB reference has incomplete folds: {sorted(manifests)}')
    validation = {str(fold): float(manifests[fold]['checkpoint_selection_value']) for fold in sorted(manifests)}
    payload: dict[str, Any] = {'method': 'fraction of tile probabilities >= 0.5 within patient', 'role': 'historical_reference_only', 'validation_fold_auroc': validation, 'validation_mean_auroc': float(np.mean(list(validation.values()))), 'validation_std_auroc': float(np.std(list(validation.values()), ddof=1))}
    if not official_test:
        return payload
    fold_rows: list[list[dict[str, Any]]] = []
    fold_aurocs: dict[str, float] = {}
    for fold in sorted(expected_folds):
        cache_path = Path(manifests[fold]['cache']['path'])
        table = pq.read_table(cache_path, partitioning=None, columns=['graph_id', 'patient_id', 'official_split', 'label_id', 'rgb_score'])
        rows = sorted((row for row in table.to_pylist() if str(row['official_split']).lower() == 'test'), key=lambda row: str(row['graph_id']))
        if not rows:
            raise RuntimeError(f'fold {fold} RGB cache has no official TEST tile scores')
        fold_rows.append(rows)
        patients, scores = historical_tile_hard_vote([row['rgb_score'] for row in rows], [row['patient_id'] for row in rows])
        truths = []
        for patient in patients:
            values = {int(row['label_id']) for row in rows if str(row['patient_id']) == str(patient)}
            if len(values) != 1:
                raise RuntimeError(f'CRC TEST patient labels are inconsistent: {patient}')
            truths.append(next(iter(values)))
        fold_aurocs[str(fold)] = binary_auroc(truths, scores)
    reference = fold_rows[0]
    reference_keys = [(str(row['graph_id']), str(row['patient_id']), int(row['label_id'])) for row in reference]
    for rows in fold_rows[1:]:
        keys = [(str(row['graph_id']), str(row['patient_id']), int(row['label_id'])) for row in rows]
        if keys != reference_keys:
            raise RuntimeError('historical RGB TEST fold tile scores are not aligned')
    score_matrix = np.asarray([[float(row['rgb_score']) for row in rows] for rows in fold_rows], dtype=np.float64)
    repeated_patients = np.tile(np.asarray([str(row['patient_id']) for row in reference], dtype=object), score_matrix.shape[0])
    patients, ensemble_scores = historical_tile_hard_vote(score_matrix.reshape(-1), repeated_patients)
    ensemble_truth = []
    for patient in patients:
        values = {int(row['label_id']) for row in reference if str(row['patient_id']) == str(patient)}
        if len(values) != 1:
            raise RuntimeError(f'CRC TEST patient labels are inconsistent: {patient}')
        ensemble_truth.append(next(iter(values)))
    payload['official_test_fold_auroc'] = fold_aurocs
    payload['official_test_ensemble'] = {'patients': len(patients), 'auroc': binary_auroc(ensemble_truth, ensemble_scores), 'auprc': binary_auprc(ensemble_truth, ensemble_scores)}
    return payload

def evaluate_results(cfg: dict[str, Any], *, official_test: bool=False) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    confirmation_aggregate: dict[str, Any] | None = None
    if official_test:
        gate = result_root / 'selection_frozen.json'
        if not gate.is_file() or json.loads(gate.read_text(encoding='utf-8')).get('status') != 'PASS':
            raise RuntimeError('official TEST is locked until selection_frozen.json is PASS')
        aggregate_official_test(cfg)
    split = 'official_test' if official_test else 'validation'
    if not official_test:
        phase = 'confirm' if any((result_root / 'confirm').glob('*/*/seed_*/fold_*/job.json')) else 'screen'
        aggregate = aggregate_phase(cfg, phase=phase)
        if phase == 'screen':
            choose_screening_encoders(cfg, aggregate)
        else:
            confirmation_aggregate = aggregate
    prediction_root = result_root / 'predictions' / split
    paths = {path.stem: path for path in prediction_root.glob('*.parquet')}
    if not paths:
        raise FileNotFoundError(f'no prediction tables in {prediction_root}')
    predictions = {arm: _load_prediction(path, cfg['dataset']) for arm, path in paths.items()}
    metrics: dict[str, Any] = {}
    for arm, values in predictions.items():
        if cfg['dataset'] == 'sicapv2':
            metrics[arm] = multiclass_metrics(values['y_true'], values['prediction'], ('NC', 'G3', 'G4', 'G5'))
            if 'g4c_valid' in values and values['g4c_valid'].any():
                mask = values['g4c_valid']
                metrics[arm]['cribriform_auroc'] = binary_auroc(values['g4c_label'][mask], values['cribriform_score'][mask])
                metrics[arm]['cribriform_auprc'] = binary_auprc(values['g4c_label'][mask], values['cribriform_score'][mask])
                metrics[arm]['cribriform_samples'] = int(mask.sum())
        else:
            metrics[arm] = binary_metrics(values['y_true'], values['prediction'], values['threshold'])
    contrasts: dict[str, Any] = {}
    encoders = sorted({key.split('__', 1)[0] for key in predictions if '__' in key}) or ['']
    for encoder in encoders:
        prefix = f'{encoder}__' if encoder else ''
        requested = [(PRIMARY_CONTRAST[0], PRIMARY_CONTRAST[1], 'primary')] + list(SECONDARY_CONTRASTS)
        local_names: list[str] = []
        for arm_a, arm_b, name in requested:
            key_a, key_b = (prefix + arm_a, prefix + arm_b)
            if key_a not in predictions or key_b not in predictions:
                continue
            first, second = (predictions[key_a], predictions[key_b])
            _aligned(first, second)
            if cfg['dataset'] == 'sicapv2':
                metric = lambda truth, probs: quadratic_weighted_kappa(truth, probs.argmax(axis=1), 4)
            else:
                from celllift.set_encoding.tasks.metrics import binary_auroc
                metric = binary_auroc
            bootstrap = paired_cluster_bootstrap(first['y_true'], first['prediction'], second['prediction'], first['patient_id'], metric, n_resamples=int(cfg['runtime']['bootstrap_replicates']), seed=20260811)
            record: dict[str, Any] = bootstrap.to_dict()
            if cfg['dataset'] == 'tcga_crc_msi':
                record['delong'] = paired_delong(first['y_true'], first['prediction'], second['prediction'])
            contrast_key = f'{encoder}__{name}' if encoder else name
            contrasts[contrast_key] = record
            if name != 'primary':
                local_names.append(contrast_key)
        primary_key = f'{encoder}__primary' if encoder else 'primary'
        _apply_hierarchical_inference(contrasts, primary_key, local_names)
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'split': split, 'metrics': metrics, 'contrasts': contrasts, 'ncr_qc': summarize_ncr_qc(cfg, official_test=official_test), 'historical_rgb_reference': summarize_historical_rgb(cfg, official_test=official_test)}
    destination = result_root / 'metrics' / f'evaluation_{split}.json'
    _atomic_json(destination, payload)
    if confirmation_aggregate is not None:
        freeze_selection_after_confirmation(cfg, confirmation_aggregate)
    return payload

def write_report(cfg: dict[str, Any]) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    candidates = [result_root / 'metrics' / 'evaluation_validation.json', result_root / 'metrics' / 'evaluation_official_test.json']
    reports = [json.loads(path.read_text(encoding='utf-8')) for path in candidates if path.is_file()]
    if not reports:
        raise FileNotFoundError('no evaluation JSON is available')
    lines = [f"# {cfg['dataset']} SetEncoder set_encoding results", '']
    for report in reports:
        aggregate_path = result_root / 'metrics' / ('aggregate_official_test.json' if report['split'] == 'official_test' else 'aggregate_confirm.json' if (result_root / 'metrics' / 'aggregate_confirm.json').is_file() else 'aggregate_screen.json')
        aggregate = json.loads(aggregate_path.read_text(encoding='utf-8')) if aggregate_path.is_file() else {'models': {}}
        lines.extend((f"## {report['split']}", ''))
        metric_name = 'qwk' if cfg['dataset'] == 'sicapv2' else 'auroc'
        if cfg['dataset'] == 'sicapv2':
            lines.extend(('| Encoder/arm | QWK | Macro-F1 | Balanced acc. | AUROC NC | AUROC G3 | AUROC G4 | AUROC G5 | Seed QWK mean ± SD | Seeds |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|'))
        else:
            lines.extend(('| Encoder/arm | AUROC | AUPRC | Balanced acc. | Sensitivity | Specificity | Frozen threshold | Seed AUROC mean ± SD | Seeds |', '|---|---:|---:|---:|---:|---:|---:|---:|---|'))
        for arm, values in sorted(report['metrics'].items()):
            record = aggregate.get('models', {}).get(arm, {})
            seed_metrics = record.get('seed_metrics', {})
            seed_values = [float(item[metric_name]) for item in seed_metrics.values() if metric_name in item]
            mean_sd = '—'
            if seed_values:
                mean = float(np.mean(seed_values))
                sd = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
                mean_sd = f'{mean:.6f} ± {sd:.6f}'
            seeds = ','.join(map(str, record.get('seeds', []))) or '—'
            if cfg['dataset'] == 'sicapv2':
                lines.append(f"| {arm} | {values['qwk']:.6f} | {values['macro_f1']:.6f} | {values['balanced_accuracy']:.6f} | {values['auroc_NC']:.6f} | {values['auroc_G3']:.6f} | {values['auroc_G4']:.6f} | {values['auroc_G5']:.6f} | {mean_sd} | {seeds} |")
            else:
                lines.append(f"| {arm} | {values['auroc']:.6f} | {values['auprc']:.6f} | {values['balanced_accuracy']:.6f} | {values['sensitivity']:.6f} | {values['specificity']:.6f} | {values['threshold']:.6f} | {mean_sd} | {seeds} |")
        fold_rows = []
        for arm, record in sorted(aggregate.get('models', {}).items()):
            for seed, folds in sorted(record.get('fold_metrics', {}).items(), key=lambda item: int(item[0])):
                for fold, fold_metrics in sorted(folds.items(), key=lambda item: int(item[0])):
                    if metric_name in fold_metrics:
                        fold_rows.append((arm, seed, fold, float(fold_metrics[metric_name])))
        if fold_rows:
            lines.extend(('', f'### Per-seed/per-fold {metric_name.upper()}', '', '| Encoder/arm | Seed | Fold | Metric |', '|---|---:|---:|---:|'))
            for arm, seed, fold, value in fold_rows:
                lines.append(f'| {arm} | {seed} | {fold} | {value:.6f} |')
        if cfg['dataset'] == 'sicapv2':
            crib = [(arm, values) for arm, values in sorted(report['metrics'].items()) if 'cribriform_auroc' in values]
            if crib:
                lines.extend(('', '### Masked G4 cribriform auxiliary task', '', '| Encoder/arm | N | AUROC | AUPRC |', '|---|---:|---:|---:|'))
                for arm, values in crib:
                    lines.append(f"| {arm} | {values['cribriform_samples']} | {values['cribriform_auroc']:.6f} | {values['cribriform_auprc']:.6f} |")
        lines.extend(('', '### Contrasts', ''))
        for name, values in report['contrasts'].items():
            holm = f", Holm p={values['holm']['p_adjusted']:.6g}, reject={values['holm']['reject']}" if 'holm' in values else ''
            delong = f", DeLong p={values['delong']['p_value']:.6g}" if 'delong' in values else ''
            hierarchy = f", hierarchy_reached={values['hierarchy_reached']}" if 'hierarchy_reached' in values else ''
            positive = f", positive={values['positive']}" if 'positive' in values else ''
            lines.append(f"- {name}: delta={values['observed_delta']:.6f}, 95% CI [{values['ci_low']:.6f}, {values['ci_high']:.6f}], p={values['p_value']:.6g}{holm}{delong}{hierarchy}{positive}.")
        qc = report.get('ncr_qc')
        if qc:
            total = qc['total']
            lines.extend(('', '### Raw NCR validity', '', f"Total anchors: {total['anchors']}; invalid NCR2D: {total['invalid_ncr_2d']} ({total['invalid_fraction_2d']:.4%}); invalid NCR3D: {total['invalid_ncr_3d']} ({total['invalid_fraction_3d']:.4%}).", '', '| Split/class | Anchors | Invalid NCR2D | Invalid NCR3D |', '|---|---:|---:|---:|'))
            for key, values in sorted(qc['groups'].items()):
                lines.append(f"| {key} | {values['anchors']} | {values['invalid_ncr_2d']} ({values['invalid_fraction_2d']:.4%}) | {values['invalid_ncr_3d']} ({values['invalid_fraction_3d']:.4%}) |")
        historical = report.get('historical_rgb_reference')
        if historical:
            lines.extend(('', '### Historical CRC RGB tile hard-vote reference', '', 'This reference is reported separately and is not the controlled gated-MIL aggregator.', '', '| Fold | Validation patient AUROC |', '|---:|---:|'))
            for fold, value in sorted(historical['validation_fold_auroc'].items()):
                lines.append(f'| {fold} | {value:.6f} |')
            lines.append(f"\nValidation fold mean ± SD: {historical['validation_mean_auroc']:.6f} ± {historical['validation_std_auroc']:.6f}.")
            if 'official_test_ensemble' in historical:
                value = historical['official_test_ensemble']
                lines.append(f"\nFrozen-fold TEST hard-vote ensemble: N={value['patients']}, AUROC={value['auroc']:.6f}, AUPRC={value['auprc']:.6f}.")
        lines.append('')
    lines.extend(('## Interpretation boundary', '', 'Cell geometry is a frozen model-predicted envelope, not an observed cell boundary. A negative result concerns this reconstruction candidate; a positive NCR-only result does not establish a generic 3D morphology effect.', ''))
    destination = result_root / 'metrics' / 'report.md'
    _atomic_text(destination, '\n'.join(lines))
    return {'status': 'PASS', 'report': str(destination), 'sections': len(reports)}
