from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import numpy as np
from celllift.set_encoding.experiment import ENCODERS, SCREENING_PROTOCOL_ID, confirmation_arms, screening_arms
from celllift.set_encoding.tasks.metrics import binary_metrics, multiclass_metrics, select_youden_threshold

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)

def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _metric(dataset: str, rows: list[dict[str, Any]], *, frozen_threshold: float | None=None) -> dict[str, float]:
    truth = np.asarray([int(row['y_true']) for row in rows])
    if dataset == 'sicapv2':
        probs = np.asarray([[float(row[f'prob_{i}']) for i in range(4)] for row in rows])
        return multiclass_metrics(truth, probs, ('NC', 'G3', 'G4', 'G5'))
    scores = np.asarray([float(row['score']) for row in rows])
    threshold = select_youden_threshold(truth, scores) if frozen_threshold is None else float(frozen_threshold)
    return binary_metrics(truth, scores, threshold)

def _sort_and_validate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = sorted((dict(row) for row in rows), key=lambda row: str(row['sample_id']))
    ids = [str(row['sample_id']) for row in result]
    if not result or len(ids) != len(set(ids)):
        raise RuntimeError('fold predictions are empty or contain duplicate sample IDs')
    return result

def _ensemble(dataset: str, per_seed: list[list[dict[str, Any]]], *, fit_validation_threshold: bool=True) -> list[dict[str, Any]]:
    reference = per_seed[0]
    keys = ('sample_id', 'patient_id', 'y_true')
    for current in per_seed[1:]:
        if len(current) != len(reference):
            raise RuntimeError('seed predictions have different sample counts')
        for first, second in zip(reference, current):
            if any((first[key] != second[key] for key in keys)):
                raise RuntimeError('seed predictions are not exactly aligned')
    result: list[dict[str, Any]] = []
    for index, row in enumerate(reference):
        output = {key: row[key] for key in keys}
        if 'graph_id' in row:
            output['graph_id'] = row['graph_id']
        if dataset == 'sicapv2':
            for column in range(4):
                output[f'prob_{column}'] = float(np.mean([seed[index][f'prob_{column}'] for seed in per_seed]))
            if all(('cribriform_score' in seed[index] for seed in per_seed)):
                output['cribriform_score'] = float(np.mean([seed[index]['cribriform_score'] for seed in per_seed]))
            if 'g4c_valid' in row:
                output['g4c_valid'] = bool(row['g4c_valid'])
                output['g4c_label'] = row.get('g4c_label')
        else:
            output['score'] = float(np.mean([seed[index]['score'] for seed in per_seed]))
        result.append(output)
    if dataset == 'tcga_crc_msi' and fit_validation_threshold:
        threshold = select_youden_threshold([row['y_true'] for row in result], [row['score'] for row in result])
        for row in result:
            row['validation_threshold'] = float(threshold)
    return result

def aggregate_phase(cfg: dict[str, Any], *, phase: str) -> dict[str, Any]:
    if phase not in {'screen', 'confirm'}:
        raise ValueError('phase must be screen or confirm')
    result_root = Path(cfg['paths']['result_root'])
    phase_root = result_root / phase
    groups: dict[tuple[str, str, int], list[tuple[int, Path, dict[str, Any]]]] = defaultdict(list)
    for manifest_path in phase_root.glob('*/*/seed_*/fold_*/job.json'):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS':
            continue
        if manifest.get('protocol_id') != SCREENING_PROTOCOL_ID:
            raise RuntimeError(f"incompatible downstream protocol in {manifest_path}: {manifest.get('protocol_id')!r}")
        encoder = str(manifest['encoder'])
        arm = str(manifest['arm']['arm_id'])
        seed, fold = (int(manifest['seed']), int(manifest['fold']))
        prediction = manifest_path.with_name('validation_predictions.parquet')
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        groups[encoder, arm, seed].append((fold, prediction, dict(manifest.get('metrics', {}))))
    if not groups:
        raise FileNotFoundError(f'no completed jobs under {phase_root}')
    expected_folds = set(range(int(cfg['split']['validation_folds'])))
    by_model: dict[tuple[str, str], list[tuple[int, list[dict[str, Any]], dict[str, float]]]] = defaultdict(list)
    for (encoder, arm, seed), pieces in sorted(groups.items()):
        folds = [fold for fold, _, _ in pieces]
        if len(folds) != len(set(folds)) or set(folds) != expected_folds:
            raise RuntimeError(f'incomplete fold coverage for {encoder}/{arm}/seed_{seed}: {sorted(folds)}')
        joined = _sort_and_validate((row for _, path, _ in sorted(pieces) for row in _rows(path)))
        metric = _metric(str(cfg['dataset']), joined)
        by_seed_path = result_root / 'predictions' / 'validation' / 'by_seed' / encoder / arm / f'seed_{seed}.parquet'
        _atomic_parquet(by_seed_path, joined)
        by_model[encoder, arm].append((seed, joined, metric))
    payload_models: dict[str, Any] = {}
    for (encoder, arm), values in sorted(by_model.items()):
        values.sort(key=lambda item: item[0])
        ensemble = _ensemble(str(cfg['dataset']), [item[1] for item in values])
        destination = result_root / 'predictions' / 'validation' / f'{encoder}__{arm}.parquet'
        _atomic_parquet(destination, ensemble)
        primary_name = 'qwk' if cfg['dataset'] == 'sicapv2' else 'auroc'
        seed_values = [float(item[2][primary_name]) for item in values]
        fold_primary_mean_by_seed = {str(seed): float(np.mean([float(metric[primary_name]) for _, _, metric in sorted(groups[encoder, arm, seed])])) for seed, _, _ in values}
        payload_models[f'{encoder}__{arm}'] = {'encoder': encoder, 'arm': arm, 'seeds': [item[0] for item in values], 'folds_per_seed': len(expected_folds), 'samples': len(ensemble), 'seed_metrics': {str(item[0]): item[2] for item in values}, 'fold_metrics': {str(seed): {str(fold): metric for fold, _, metric in sorted(groups[encoder, arm, seed])} for seed, _, _ in values}, 'fold_primary_mean_by_seed': fold_primary_mean_by_seed, 'seed_primary_mean': float(np.mean(seed_values)), 'seed_primary_std': float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0, 'ensemble_metrics': _metric(str(cfg['dataset']), ensemble), 'prediction': str(destination)}
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'phase': phase, 'protocol_id': SCREENING_PROTOCOL_ID, 'models': payload_models}
    _atomic_json(result_root / 'metrics' / f'aggregate_{phase}.json', payload)
    return payload

def aggregate_official_test(cfg: dict[str, Any]) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    gate_path = result_root / 'selection_frozen.json'
    if not gate_path.is_file() or json.loads(gate_path.read_text(encoding='utf-8')).get('status') != 'PASS':
        raise RuntimeError('official TEST aggregation is locked')
    groups: dict[tuple[str, str, int], list[tuple[int, Path]]] = defaultdict(list)
    for manifest_path in (result_root / 'official_test').glob('*/*/seed_*/fold_*/job.json'):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS':
            continue
        if manifest.get('protocol_id') != SCREENING_PROTOCOL_ID:
            raise RuntimeError(f'incompatible official TEST protocol in {manifest_path}')
        key = (str(manifest['encoder']), str(manifest['arm']['arm_id']), int(manifest['seed']))
        groups[key].append((int(manifest['fold']), manifest_path.with_name('official_test_predictions.parquet')))
    if not groups:
        raise FileNotFoundError('no frozen official TEST jobs')
    expected_folds = set(range(int(cfg['split']['validation_folds'])))
    per_model: dict[tuple[str, str], list[tuple[int, list[dict[str, Any]], dict[str, float]]]] = defaultdict(list)
    for (encoder, arm, seed), pieces in sorted(groups.items()):
        folds = [fold for fold, _ in pieces]
        if set(folds) != expected_folds or len(folds) != len(set(folds)):
            raise RuntimeError(f'incomplete official TEST folds for {encoder}/{arm}/seed_{seed}')
        fold_rows = [_sort_and_validate(_rows(path)) for _, path in sorted(pieces)]
        seed_rows = _ensemble(str(cfg['dataset']), fold_rows, fit_validation_threshold=False)
        frozen_threshold: float | None = None
        if cfg['dataset'] == 'tcga_crc_msi':
            validation = _rows(result_root / 'predictions' / 'validation' / f'{encoder}__{arm}.parquet')
            threshold = {float(row['validation_threshold']) for row in validation}
            if len(threshold) != 1:
                raise RuntimeError('frozen validation ensemble threshold is not unique')
            frozen_threshold = next(iter(threshold))
            for row in seed_rows:
                row['validation_threshold'] = frozen_threshold
        destination = result_root / 'predictions' / 'official_test' / 'by_seed' / encoder / arm / f'seed_{seed}.parquet'
        _atomic_parquet(destination, seed_rows)
        per_model[encoder, arm].append((seed, seed_rows, _metric(str(cfg['dataset']), seed_rows, frozen_threshold=frozen_threshold)))
    models: dict[str, Any] = {}
    for (encoder, arm), values in sorted(per_model.items()):
        values.sort(key=lambda item: item[0])
        ensemble = _ensemble(str(cfg['dataset']), [item[1] for item in values], fit_validation_threshold=False)
        frozen_threshold: float | None = None
        if cfg['dataset'] == 'tcga_crc_msi':
            validation = _rows(result_root / 'predictions' / 'validation' / f'{encoder}__{arm}.parquet')
            thresholds = {float(row['validation_threshold']) for row in validation}
            if len(thresholds) != 1:
                raise RuntimeError('frozen validation ensemble threshold is not unique')
            frozen_threshold = next(iter(thresholds))
            for row in ensemble:
                row['validation_threshold'] = frozen_threshold
        destination = result_root / 'predictions' / 'official_test' / f'{encoder}__{arm}.parquet'
        _atomic_parquet(destination, ensemble)
        models[f'{encoder}__{arm}'] = {'encoder': encoder, 'arm': arm, 'seeds': [item[0] for item in values], 'folds_per_seed': len(expected_folds), 'samples': len(ensemble), 'seed_metrics': {str(item[0]): item[2] for item in values}, 'ensemble_metrics': _metric(str(cfg['dataset']), ensemble, frozen_threshold=frozen_threshold), 'prediction': str(destination)}
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'phase': 'official_test', 'models': models}
    _atomic_json(result_root / 'metrics' / 'aggregate_official_test.json', payload)
    return payload

def choose_screening_encoders(cfg: dict[str, Any], aggregate: dict[str, Any] | None=None) -> dict[str, Any]:
    if aggregate is None:
        aggregate = aggregate_phase(cfg, phase='screen')
    models = aggregate['models']
    result_root = Path(cfg['paths']['result_root'])
    for fold in range(int(cfg['split']['validation_folds'])):
        hashes: dict[str, str] = {}
        for encoder in ENCODERS:
            manifest_path = result_root / 'screen' / encoder / 'S0' / 'seed_42' / f'fold_{fold}' / 'job.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('protocol_id') != SCREENING_PROTOCOL_ID:
                raise RuntimeError(f'S0 protocol mismatch in {manifest_path}')
            hashes[encoder] = str(manifest['outputs']['predictions']['sha256'])
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f'encoder-independent S0 mismatch at fold {fold}: {hashes}')
    required = {arm.arm_id for arm in screening_arms()}
    rankings: list[dict[str, Any]] = []
    primary = 'qwk' if cfg['dataset'] == 'sicapv2' else 'auroc'
    for encoder in ENCODERS:
        missing = sorted(required - {key.split('__', 1)[1] for key in models if key.startswith(f'{encoder}__')})
        if missing:
            raise RuntimeError(f'screening incomplete for {encoder}: {missing}')

        def score(arm: str) -> float:
            record = models[f'{encoder}__{arm}']
            fold_means = record.get('fold_primary_mean_by_seed', {})
            if set(fold_means) != {'42'}:
                raise RuntimeError(f'screening selection requires only seed 42 fold means for {encoder}/{arm}, got {sorted(fold_means)}')
            return float(fold_means['42'])
        auxiliary = (score('S2') - score('S1'), score('S4') - score('S3'), score('S6') - score('S5'))
        rankings.append({'encoder': encoder, 'primary_delta_s8_minus_s7': score('S8') - score('S7'), 'auxiliary_deltas': list(auxiliary), 'auxiliary_mean': float(np.mean(auxiliary)), 'fold_mean_scores': {arm: score(arm) for arm in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8')}})
    cost_rank = {'meanpool': 0, 'deepsets': 1, 'set_transformer': 2}
    rankings.sort(key=lambda row: (-row['primary_delta_s8_minus_s7'], -row['auxiliary_mean'], cost_rank[row['encoder']]))
    best_delta = rankings[0]['primary_delta_s8_minus_s7']
    tied = [row for row in rankings if best_delta - row['primary_delta_s8_minus_s7'] < 0.002]
    tied.sort(key=lambda row: (-row['auxiliary_mean'], cost_rank[row['encoder']]))
    winner = tied[0]['encoder']
    tied_names = {row['encoder'] for row in tied}
    ordered = [row['encoder'] for row in tied] + [row['encoder'] for row in rankings if row['encoder'] not in tied_names]
    rank_lookup = {encoder: index + 1 for index, encoder in enumerate(ordered)}
    for row in rankings:
        row['selection_rank'] = rank_lookup[row['encoder']]
    rankings.sort(key=lambda row: row['selection_rank'])
    if winner == 'meanpool':
        runner_up = next((encoder for encoder in ordered if encoder != 'meanpool'))
        retained = ['meanpool', runner_up]
    else:
        runner_up = next((encoder for encoder in ordered if encoder != winner))
        retained = [winner, 'meanpool']
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'selection_source': 'validation_only', 'winner': winner, 'runner_up': runner_up, 'retained_encoders': retained, 'delta_tolerance': 0.002, 'rankings': rankings, 'selection_metric_aggregation': 'arithmetic_mean_of_fold_metrics_set_encoding', 'protocol_id': SCREENING_PROTOCOL_ID, 'retired_encoders': ['set_transformer']}
    _atomic_json(result_root / 'screening_selection.json', payload)
    return payload

def freeze_selection_after_confirmation(cfg: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    screen_path = result_root / 'screening_selection.json'
    if not screen_path.is_file():
        raise RuntimeError('screening_selection.json is absent')
    screen = json.loads(screen_path.read_text(encoding='utf-8'))
    if screen.get('protocol_id') != SCREENING_PROTOCOL_ID:
        raise RuntimeError('screening selection uses an incompatible downstream protocol')
    if screen.get('selection_metric_aggregation') != 'arithmetic_mean_of_fold_metrics_set_encoding':
        raise RuntimeError('screening selection does not use registered fold-mean aggregation')
    required_arms = {arm.arm_id for arm in confirmation_arms()}
    required_seeds = {17, 42, 73, 101, 137}
    for encoder in screen['retained_encoders']:
        for arm in required_arms:
            record = aggregate['models'].get(f'{encoder}__{arm}')
            if record is None or set(record['seeds']) != required_seeds:
                raise RuntimeError(f'confirmation incomplete for {encoder}/{arm}')
    source = result_root / 'metrics' / 'aggregate_confirm.json'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'official_test_unlocked': True, 'winner': screen['winner'], 'retained_encoders': screen['retained_encoders'], 'protocol_id': SCREENING_PROTOCOL_ID, 'selection_metric_aggregation': screen['selection_metric_aggregation'], 'confirmation_arms': sorted(required_arms), 'confirmation_seeds': sorted(required_seeds), 'validation_aggregate': str(source), 'validation_aggregate_sha256': digest}
    _atomic_json(result_root / 'selection_frozen.json', payload)
    return payload
