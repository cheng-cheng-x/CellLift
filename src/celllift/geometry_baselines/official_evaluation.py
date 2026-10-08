from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import json
import math
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .fusion import apply_crc, apply_sicap, continuity_logit, fit_crc_calibration, fit_sicap_alpha, probability_logit
from .io_utils import atomic_json, atomic_parquet, sha256
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS, require_test_gate
from .statistics import holm_adjust
from .validation import _crc_paper_patients, _geometry_eligible_ids, _geometry_seed_average, _paper_seed_average, _tile_calibration_weight, _tile_geometry_seed_average
from .validation_statistics import INTERACTIONS, PAIR_FAMILIES, _delong_fold, _qwk_from_confusion, _result_from_distribution, _sicap_distributions, _weighted_auc

def _rows(path: Path, **kwargs) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path, partitioning=None, **kwargs).to_pylist()

def _verify_artifact(path: str | Path, expected: str, context: str) -> None:
    target = Path(path)
    if not target.is_file() or sha256(target) != expected:
        raise RuntimeError(f'official staging artifact checksum mismatch: {context}: {target}')

def _verify_job_artifacts(value: Mapping[str, Any], context: str) -> None:
    for name in ('checkpoint', 'predictions'):
        if name in value and f'{name}_sha256' in value:
            _verify_artifact(str(value[name]), str(value[f'{name}_sha256']), f'{context}/{name}')

def _softmax(logits: np.ndarray) -> np.ndarray:
    value = logits - logits.max(axis=-1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=-1, keepdims=True)

def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60, 60)))

def _paper_test(cfg: Mapping[str, Any], seeds: Sequence[int]=SEEDS) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'official_test' / 'paper_baseline'
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in seeds:
        for row in _rows(root / f'seed_{seed}' / 'official_test_predictions.parquet'):
            grouped[str(row['graph_id'])].append(row)
    output = []
    for graph, values in sorted(grouped.items()):
        if len(values) != len(seeds):
            raise RuntimeError(f'official paper seed coverage mismatch: {graph}')
        output.append({'graph_id': graph, 'patient_id': str(values[0]['patient_id']), 'probabilities': np.mean([np.asarray(row['probabilities'], float) for row in values], axis=0)})
    return output

def _geometry_test(cfg: Mapping[str, Any], encoder: str, arm: str, *, tile: bool=False, seeds: Sequence[int]=SEEDS, folds: Sequence[int] | None=None) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'official_test' / ('geometry_tile' if tile else 'geometry_patient') / encoder / arm
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_folds = tuple(range(int(cfg['split']['validation_folds']))) if folds is None else tuple(folds)
    for seed in seeds:
        for fold in selected_folds:
            for row in _rows(root / f'seed_{seed}' / f'fold_{fold}' / 'test_predictions.parquet'):
                key = str(row['graph_id'] if tile or cfg['dataset'] == 'sicapv2' else row['patient_id'])
                grouped[key].append(row)
    expected = len(seeds) * len(selected_folds)
    output = []
    for key, values in sorted(grouped.items()):
        if len(values) != expected:
            raise RuntimeError(f'official geometry coverage mismatch: {encoder}/{arm}/{key}')
        if cfg['dataset'] == 'sicapv2':
            probability = np.mean([_softmax(np.asarray(row['logits'], float)[None])[0] for row in values], axis=0)
            output.append({'graph_id': key, 'patient_id': str(values[0]['patient_id']), 'probability': probability})
        elif tile:
            output.append({'graph_id': key, 'patient_id': str(values[0]['patient_id']), 'probability': float(np.mean([row['score'] for row in values]))})
        else:
            output.append({'patient_id': key, 'probability': float(np.mean([row['probability'] for row in values]))})
    return output

def verify_official_staging(cfg: Mapping[str, Any]) -> dict[str, Any]:
    gate = require_test_gate(cfg['paths']['result_root'])
    root = Path(cfg['paths']['result_root'])
    train_manifest = Path(cfg['paths']['data_root']) / '01_rgb224_cache' / 'manifest.json'
    test_manifest = Path(cfg['paths']['data_root']) / '01_rgb224_cache' / 'official_test' / 'manifest.json'
    train_cache = json.loads(train_manifest.read_text(encoding='utf-8'))
    cache = json.loads(test_manifest.read_text(encoding='utf-8'))
    if train_cache.get('status') != 'PASS' or train_cache.get('include_test') is not False:
        raise RuntimeError('frozen TRAIN RGB cache is absent or no longer TRAIN-only')
    if cache.get('status') != 'PASS' or cache.get('include_test') is not True or cache.get('cache_phase') != 'official_test_only':
        raise RuntimeError('independent official TEST RGB cache is incomplete')
    evidence = [{'path': str(train_manifest), 'sha256': sha256(train_manifest)}, {'path': str(test_manifest), 'sha256': sha256(test_manifest)}]
    for seed in SEEDS:
        path = root / 'official_test' / 'paper_baseline' / f'seed_{seed}' / 'job.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') != 'PASS' or value.get('test_labels_read_during_prediction') is not False:
            raise RuntimeError(f'official paper prediction is incomplete or label-tainted: {path}')
        _verify_job_artifacts(value, str(path))
        evidence.append({'path': str(path), 'sha256': sha256(path)})
    folds = range(int(cfg['split']['validation_folds']))
    for fold in folds:
        path = Path(cfg['paths']['data_root']) / '04_residual3d_tokens' / 'official_test' / f'fold_{fold:02d}' / 'manifest.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') != 'PASS' or value.get('labels_read') is not False:
            raise RuntimeError(f'official token cache is incomplete or label-tainted: {path}')
        for file in value.get('files', []):
            _verify_artifact(file['path'], file['sha256'], f'{path}/token_shard')
        for name in ('probe', 'statistics', 'residual_manifest'):
            _verify_artifact(value[name], value[f'{name}_sha256'], f'{path}/{name}')
        evidence.append({'path': str(path), 'sha256': sha256(path)})
    for encoder in ENCODERS:
        for arm in [key for key in ARMS if key.startswith('O')]:
            for seed in SEEDS:
                for fold in folds:
                    routes = ['geometry_patient'] + (['geometry_tile'] if cfg['dataset'] == 'tcga_crc_msi' else [])
                    for route in routes:
                        path = root / 'official_test' / route / encoder / arm / f'seed_{seed}' / f'fold_{fold}' / 'job.json'
                        value = json.loads(path.read_text(encoding='utf-8'))
                        if value.get('status') != 'PASS' or value.get('test_labels_read') is not False:
                            raise RuntimeError(f'official geometry prediction is incomplete or label-tainted: {path}')
                        _verify_job_artifacts(value, str(path))
                        evidence.append({'path': str(path), 'sha256': sha256(path)})
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'selection_gate_sha256': sha256(root / 'selection_frozen.json'), 'arms': list(ARMS), 'encoders': list(ENCODERS), 'seeds': list(SEEDS), 'labels_read': False, 'evidence_count': len(evidence), 'evidence': evidence}
    atomic_json(root / 'official_test' / 'transaction_ready.json', payload)
    return payload

def _test_labels(cfg: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    path = Path(cfg['paths']['model_input_root']) / '04_labels_splits' / 'labels_splits.parquet'
    rows = _rows(path, columns=['graph_id', 'patient_id', 'label_id', 'official_split'], filters=[('official_split', 'in', ['TEST', 'test', 'Test'])])
    labels = {str(row['graph_id']): int(row['label_id']) for row in rows}
    patients = {str(row['graph_id']): str(row['patient_id']) for row in rows}
    return (labels, patients)

def _sicap_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    prediction = probability.argmax(1)
    confusion = confusion_matrix(labels, prediction, labels=[0, 1, 2, 3])
    return {'patches': int(len(labels)), 'QWK': float(_qwk_from_confusion(confusion)[0]), 'accuracy': float(accuracy_score(labels, prediction)), 'macro_F1': float(f1_score(labels, prediction, average='macro', zero_division=0)), 'balanced_accuracy': float(balanced_accuracy_score(labels, prediction)), 'per_class_F1': f1_score(labels, prediction, labels=[0, 1, 2, 3], average=None, zero_division=0).tolist(), 'per_class_AUROC': [float(roc_auc_score(labels == value, probability[:, value])) for value in range(4)], 'confusion_matrix': confusion.tolist()}

def _validation_thresholds(cfg: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    from sklearn.metrics import balanced_accuracy_score
    rows = _rows(Path(cfg['paths']['result_root']) / 'metrics' / 'validation' / 'fused_predictions.parquet')
    output = {}
    for encoder in ('paper', *ENCODERS):
        arms = ['R0'] if encoder == 'paper' else list(ARMS)
        for arm in arms:
            values = [row for row in rows if row['encoder'] == encoder and row['arm_id'] == arm]
            if not values:
                continue
            labels = np.asarray([row['label_id'] for row in values])
            scores = np.asarray([row['score'] for row in values])
            candidates = np.unique(scores)
            best, threshold = (-1.0, 0.5)
            for candidate in candidates:
                metric = balanced_accuracy_score(labels, scores >= candidate)
                if metric > best + 1e-12:
                    best, threshold = (float(metric), float(candidate))
            output[encoder, arm] = threshold
    return output

def _validation_tile_thresholds(cfg: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    from sklearn.metrics import balanced_accuracy_score
    rows = _rows(Path(cfg['paths']['result_root']) / 'metrics' / 'validation_tile' / 'predictions.parquet')
    output = {}
    for encoder in ('paper', *ENCODERS):
        arms = ['R0'] if encoder == 'paper' else [key for key in ARMS if key.startswith('R') and key != 'R0']
        for arm in arms:
            values = [row for row in rows if row['encoder'] == encoder and row['arm_id'] == arm]
            labels = np.asarray([row['label_id'] for row in values])
            scores = np.asarray([row['score'] for row in values])
            best, threshold = (-1.0, 0.5)
            for candidate in np.unique(scores):
                metric = balanced_accuracy_score(labels, scores >= candidate)
                if metric > best + 1e-12:
                    best, threshold = (float(metric), float(candidate))
            output[encoder, arm] = threshold
    return output

def _crc_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    prediction = scores >= threshold
    positive, negative = (labels == 1, labels == 0)
    return {'patients': int(len(labels)), 'patient_AUROC': float(roc_auc_score(labels, scores)), 'patient_AUPRC': float(average_precision_score(labels, scores)), 'balanced_accuracy': float(balanced_accuracy_score(labels, prediction)), 'sensitivity': float(prediction[positive].mean()), 'specificity': float((~prediction[negative]).mean()), 'validation_frozen_threshold': float(threshold)}

def _final_sicap_calibration(cfg: Mapping[str, Any], encoder: str, o_arm: str):
    paper = _paper_seed_average(cfg)
    pmap = {(row['fold'], row['graph_id']): row for row in paper}
    keys = sorted(pmap)
    geometry = _geometry_seed_average(cfg, encoder, o_arm)
    gmap = {(row['fold'], row['key']): row for row in geometry}
    labels = np.asarray([pmap[key]['label_id'] for key in keys])
    rgb = np.log(np.clip(np.stack([pmap[key]['probabilities'] for key in keys]), 1e-07, 1))
    geo = np.log(np.clip(np.stack([gmap[key]['probability'] for key in keys]), 1e-07, 1))
    return fit_sicap_alpha(rgb, geo, labels)

def _final_crc_patient_calibration(cfg: Mapping[str, Any], encoder: str, o_arm: str):
    eligible = _geometry_eligible_ids(cfg)
    paper = [row for row in _paper_seed_average(cfg) if row['graph_id'] in eligible]
    pmap = {(row['fold'], row['patient_id']): row for row in _crc_paper_patients(paper)}
    keys = sorted(pmap)
    geometry = _geometry_seed_average(cfg, encoder, o_arm)
    gmap = {(row['fold'], row['patient_id']): row for row in geometry}
    labels = np.asarray([pmap[key]['label_id'] for key in keys])
    rgb = np.asarray([pmap[key]['logit'] for key in keys])
    geo = probability_logit(np.asarray([gmap[key]['probability'] for key in keys]))
    return fit_crc_calibration(rgb, geo, labels)

def _official_statistics(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    replicates = int(cfg.get('runtime', {}).get('bootstrap_replicates', 10000))
    output = []
    paper = [row for row in rows if row['encoder'] == 'paper' and row['arm_id'] == 'R0']
    for encoder in ENCODERS:
        maps = {}
        for arm in ARMS:
            values = paper if arm == 'R0' else [row for row in rows if row['encoder'] == encoder and row['arm_id'] == arm]
            key = 'graph_id' if cfg['dataset'] == 'sicapv2' else 'patient_id'
            maps[arm] = {str(row[key]): row for row in values}
        keys = sorted(set.intersection(*(set(value) for value in maps.values())))
        labels = np.asarray([maps['R0'][key]['label_id'] for key in keys], np.int64)
        if cfg['dataset'] == 'sicapv2':
            patients = np.asarray([maps['R0'][key]['patient_id'] for key in keys], object)
            scores = {arm: np.stack([maps[arm][key]['probabilities'] for key in keys]) for arm in ARMS}
            observed, distributions = _sicap_distributions(labels, patients, scores, replicates, 20260831)
        else:
            scores = {arm: np.asarray([maps[arm][key]['score'] for key in keys]) for arm in ARMS}
            rng = np.random.default_rng(20260831)
            weights = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=replicates)
            observed = {arm: float(_weighted_auc(labels, value, np.ones(len(keys)))[0]) for arm, value in scores.items()}
            distributions = {arm: _weighted_auc(labels, value, weights) for arm, value in scores.items()}
        for family, comparisons in PAIR_FAMILIES.items():
            current = []
            for first, second in comparisons:
                row = {'encoder': encoder, 'family': family, 'comparison': f'{first}-{second}', **_result_from_distribution(observed[first] - observed[second], distributions[first] - distributions[second])}
                if cfg['dataset'] == 'tcga_crc_msi':
                    z, p = _delong_fold(labels, scores[first], scores[second])
                    row.update({'delong_z': z, 'delong_p_one_sided': p})
                current.append(row)
            adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in current})
            for row in current:
                row['holm_p'] = adjusted[row['comparison']]
                row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
                output.append(row)
        current = []
        for name, terms in INTERACTIONS.items():
            current.append({'encoder': encoder, 'family': 'interaction', 'comparison': name, **_result_from_distribution(sum((weight * observed[arm] for arm, weight in terms)), sum((weight * distributions[arm] for arm, weight in terms)))})
        adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in current})
        for row in current:
            row['holm_p'] = adjusted[row['comparison']]
            row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
            output.append(row)
    root = Path(cfg['paths']['result_root']) / 'statistics' / 'official_test'
    atomic_parquet(root / 'comparisons.parquet', output)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'replicates': replicates, 'comparisons': output}
    atomic_json(root / 'summary.json', payload)
    return payload

def _official_crc_tile_statistics(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    replicates = int(cfg.get('runtime', {}).get('bootstrap_replicates', 10000))
    available = ('R0',) + tuple((arm for arm in ARMS if arm.startswith('R') and arm != 'R0'))
    paper = [row for row in rows if row['encoder'] == 'paper' and row['arm_id'] == 'R0']
    output = []
    coverage = {}
    for encoder in ENCODERS:
        maps = {}
        for arm in available:
            selected = paper if arm == 'R0' else [row for row in rows if row['encoder'] == encoder and row['arm_id'] == arm]
            maps[arm] = {str(row['patient_id']): row for row in selected}
        keys = sorted(set.intersection(*(set(value) for value in maps.values())))
        if not keys:
            raise RuntimeError(f'empty official tile-fusion comparison universe: {encoder}')
        coverage[encoder] = {'common_patients': len(keys), 'per_arm': {arm: len(value) for arm, value in maps.items()}}
        labels = np.asarray([int(maps['R0'][key]['label_id']) for key in keys], np.int64)
        scores = {arm: np.asarray([float(maps[arm][key]['score']) for key in keys]) for arm in available}
        rng = np.random.default_rng(20260831)
        weights = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=replicates)
        observed = {arm: float(_weighted_auc(labels, value, np.ones(len(keys)))[0]) for arm, value in scores.items()}
        distributions = {arm: _weighted_auc(labels, value, weights) for arm, value in scores.items()}
        for family, comparisons in PAIR_FAMILIES.items():
            registered = [(first, second) for first, second in comparisons if first in scores and second in scores]
            if not registered:
                continue
            current = []
            for first, second in registered:
                z, p = _delong_fold(labels, scores[first], scores[second])
                current.append({'encoder': encoder, 'family': family, 'comparison': f'{first}-{second}', **_result_from_distribution(observed[first] - observed[second], distributions[first] - distributions[second]), 'delong_z': z, 'delong_p_one_sided': p})
            adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in current})
            for row in current:
                row['holm_p'] = adjusted[row['comparison']]
                row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
                output.append(row)
        current = []
        for name, terms in INTERACTIONS.items():
            if not all((arm in scores for arm, _ in terms)):
                continue
            current.append({'encoder': encoder, 'family': 'interaction', 'comparison': name, **_result_from_distribution(sum((weight * observed[arm] for arm, weight in terms)), sum((weight * distributions[arm] for arm, weight in terms)))})
        adjusted = holm_adjust({row['comparison']: row['p_one_sided'] for row in current})
        for row in current:
            row['holm_p'] = adjusted[row['comparison']]
            row['positive'] = row['ci_low'] > 0 and row['holm_p'] < 0.05
            output.append(row)
    root = Path(cfg['paths']['result_root']) / 'statistics' / 'official_test_tile'
    atomic_parquet(root / 'comparisons.parquet', output)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'route': 'tile_temperature_scalar_then_patient_hard_vote', 'replicates': replicates, 'coverage': coverage, 'comparisons': output}
    atomic_json(root / 'summary.json', payload)
    return payload

def _component_metrics(cfg: Mapping[str, Any], labels_by_graph: Mapping[str, int], patient_by_graph: Mapping[str, str], thresholds: Mapping[tuple[str, str], float] | None) -> list[dict[str, Any]]:
    from sklearn.metrics import roc_auc_score
    output: list[dict[str, Any]] = []
    paper_by_seed = {seed: _paper_test(cfg, (seed,)) for seed in SEEDS}
    if cfg['dataset'] == 'sicapv2':
        keys = sorted(labels_by_graph)
        labels = np.asarray([labels_by_graph[key] for key in keys])
        for seed, rows in paper_by_seed.items():
            mapping = {row['graph_id']: row for row in rows}
            probability = np.stack([mapping[key]['probabilities'] for key in keys])
            output.append({'component': 'seed', 'seed': seed, 'fold': None, 'encoder': 'paper', 'arm_id': 'R0', **_sicap_metrics(labels, probability)})
        for encoder in ENCODERS:
            for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                calibration = _final_sicap_calibration(cfg, encoder, o_arm)
                r_arm = 'R' + o_arm[1:]
                for seed in SEEDS:
                    paper = {row['graph_id']: row for row in paper_by_seed[seed]}
                    geometry = {row['graph_id']: row for row in _geometry_test(cfg, encoder, o_arm, seeds=(seed,))}
                    rgb = np.stack([paper[key]['probabilities'] for key in keys])
                    gp = np.stack([geometry[key]['probability'] for key in keys])
                    fused = _softmax(apply_sicap(np.log(np.clip(rgb, 1e-07, 1)), np.log(np.clip(gp, 1e-07, 1)), calibration))
                    output.extend([{'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels, gp)}, {'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': r_arm, **_sicap_metrics(labels, fused)}])
                    for fold in range(int(cfg['split']['validation_folds'])):
                        value = {row['graph_id']: row for row in _geometry_test(cfg, encoder, o_arm, seeds=(seed,), folds=(fold,))}
                        probability = np.stack([value[key]['probability'] for key in keys])
                        output.append({'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels, probability)})
    else:
        assert thresholds is not None
        paper_patient: dict[int, list[dict[str, Any]]] = {}
        for seed, values in paper_by_seed.items():
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in values:
                groups[str(row['patient_id'])].append(row)
            rows = []
            for patient, tiles in sorted(groups.items()):
                if len(tiles) < 10:
                    continue
                positive = sum((row['probabilities'][1] >= 0.5 for row in tiles))
                rows.append({'patient_id': patient, 'label_id': labels_by_graph[tiles[0]['graph_id']], 'score': positive / len(tiles)})
            paper_patient[seed] = rows
            label = np.asarray([row['label_id'] for row in rows])
            score = np.asarray([row['score'] for row in rows])
            output.append({'component': 'seed', 'seed': seed, 'fold': None, 'encoder': 'paper', 'arm_id': 'R0', **_crc_metrics(label, score, thresholds['paper', 'R0'])})
        geometry_keys = {row['graph_id'] for row in _geometry_test(cfg, 'meanpool', 'O1', tile=True)}
        common_tiles: dict[str, list[str]] = defaultdict(list)
        for graph in geometry_keys:
            common_tiles[patient_by_graph[graph]].append(graph)
        for encoder in ENCODERS:
            for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                calibration = _final_crc_patient_calibration(cfg, encoder, o_arm)
                r_arm = 'R' + o_arm[1:]
                for seed in SEEDS:
                    geometry = {row['patient_id']: row for row in _geometry_test(cfg, encoder, o_arm, seeds=(seed,))}
                    paper_map = {row['graph_id']: row for row in paper_by_seed[seed]}
                    common_rgb = {}
                    for patient, graphs in common_tiles.items():
                        if len(graphs) < 10:
                            continue
                        positive = sum((paper_map[graph]['probabilities'][1] >= 0.5 for graph in graphs))
                        common_rgb[patient] = {'logit': float(continuity_logit(np.array([positive]), np.array([len(graphs)]))[0]), 'label_id': labels_by_graph[graphs[0]]}
                    keys = sorted(set(common_rgb) & set(geometry))
                    label = np.asarray([common_rgb[key]['label_id'] for key in keys])
                    gp = np.asarray([geometry[key]['probability'] for key in keys])
                    rgb = np.asarray([common_rgb[key]['logit'] for key in keys])
                    fused = _sigmoid(apply_crc(rgb, probability_logit(gp), calibration))
                    output.extend([{'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': o_arm, **_crc_metrics(label, gp, thresholds[encoder, o_arm])}, {'component': 'seed', 'seed': seed, 'fold': None, 'encoder': encoder, 'arm_id': r_arm, **_crc_metrics(label, fused, thresholds[encoder, r_arm])}])
                    for fold in range(int(cfg['split']['validation_folds'])):
                        value = {row['patient_id']: row for row in _geometry_test(cfg, encoder, o_arm, seeds=(seed,), folds=(fold,))}
                        fkeys = sorted(set(value) & set(common_rgb))
                        flabel = np.asarray([common_rgb[key]['label_id'] for key in fkeys])
                        fscore = np.asarray([value[key]['probability'] for key in fkeys])
                        output.append({'component': 'fold_seed', 'seed': seed, 'fold': fold, 'encoder': encoder, 'arm_id': o_arm, 'patient_AUROC': float(roc_auc_score(flabel, fscore))})
    return output

def evaluate_official_test(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ready = verify_official_staging(cfg)
    from .qc import compute_ncr_qc
    ncr_qc = compute_ncr_qc(cfg, include_test=True)
    root = Path(cfg['paths']['result_root'])
    labels_by_graph, patient_by_graph = _test_labels(cfg)
    paper = _paper_test(cfg)
    metrics, predictions = ([], [])
    tile_statistics = None
    thresholds = None
    if cfg['dataset'] == 'sicapv2':
        pmap = {row['graph_id']: row for row in paper}
        keys = sorted(pmap)
        labels = np.asarray([labels_by_graph[key] for key in keys])
        rgb = np.stack([pmap[key]['probabilities'] for key in keys])
        metrics.append({'encoder': 'paper', 'arm_id': 'R0', **_sicap_metrics(labels, rgb)})
        predictions.extend(({'encoder': 'paper', 'arm_id': 'R0', 'graph_id': key, 'patient_id': patient_by_graph[key], 'label_id': int(label), 'probabilities': probability.tolist()} for key, label, probability in zip(keys, labels, rgb)))
        for encoder in ENCODERS:
            for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                geometry = {row['graph_id']: row for row in _geometry_test(cfg, encoder, o_arm)}
                gp = np.stack([geometry[key]['probability'] for key in keys])
                r_arm = 'R' + o_arm[1:]
                calibration = _final_sicap_calibration(cfg, encoder, o_arm)
                fused = _softmax(apply_sicap(np.log(np.clip(rgb, 1e-07, 1)), np.log(np.clip(gp, 1e-07, 1)), calibration))
                metrics.extend([{'encoder': encoder, 'arm_id': o_arm, **_sicap_metrics(labels, gp)}, {'encoder': encoder, 'arm_id': r_arm, 'alpha': calibration.alpha, **_sicap_metrics(labels, fused)}])
                for arm, score in ((o_arm, gp), (r_arm, fused)):
                    predictions.extend(({'encoder': encoder, 'arm_id': arm, 'graph_id': key, 'patient_id': patient_by_graph[key], 'label_id': int(label), 'probabilities': value.tolist()} for key, label, value in zip(keys, labels, score)))
    else:
        thresholds = _validation_thresholds(cfg)
        paper_map = {row['graph_id']: row for row in paper}
        groups: dict[str, list[str]] = defaultdict(list)
        for graph in paper_map:
            groups[patient_by_graph[graph]].append(graph)
        r0 = []
        for patient, graphs in sorted(groups.items()):
            if len(graphs) < 10:
                continue
            positive = sum((paper_map[graph]['probabilities'][1] >= 0.5 for graph in graphs))
            r0.append({'patient_id': patient, 'label_id': labels_by_graph[graphs[0]], 'score': positive / len(graphs), 'tiles': len(graphs)})
        l = np.asarray([row['label_id'] for row in r0])
        s = np.asarray([row['score'] for row in r0])
        metrics.append({'encoder': 'paper', 'arm_id': 'R0', 'tile_universe': 'all_source', **_crc_metrics(l, s, thresholds['paper', 'R0'])})
        predictions.extend(({'encoder': 'paper', 'arm_id': 'R0', **row} for row in r0))
        first_geometry = _geometry_test(cfg, 'meanpool', 'O1', tile=True)
        eligible = {row['graph_id'] for row in first_geometry}
        common_groups: dict[str, list[str]] = defaultdict(list)
        for graph in eligible:
            common_groups[patient_by_graph[graph]].append(graph)
        common_rgb = {}
        for patient, graphs in common_groups.items():
            if len(graphs) < 10:
                continue
            positive = sum((paper_map[graph]['probabilities'][1] >= 0.5 for graph in graphs))
            common_rgb[patient] = {'logit': float(continuity_logit(np.array([positive]), np.array([len(graphs)]))[0]), 'score': positive / len(graphs), 'tiles': len(graphs), 'label_id': labels_by_graph[graphs[0]]}
        for encoder in ENCODERS:
            for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                geometry = {row['patient_id']: row for row in _geometry_test(cfg, encoder, o_arm)}
                missing, extra = (set(common_rgb) - set(geometry), set(geometry) - set(common_rgb))
                if missing:
                    raise RuntimeError(f'official endpoint patients missing geometry: {encoder}/{o_arm}: {len(missing)}')
                if any((len(common_groups.get(patient, [])) >= 10 for patient in extra)):
                    raise RuntimeError(f'official geometry extra patient is not explained by <10 tiles: {encoder}/{o_arm}')
                keys = sorted(common_rgb)
                r_arm = 'R' + o_arm[1:]
                label = np.asarray([common_rgb[key]['label_id'] for key in keys])
                gp = np.asarray([geometry[key]['probability'] for key in keys])
                rgb_logit = np.asarray([common_rgb[key]['logit'] for key in keys])
                calibration = _final_crc_patient_calibration(cfg, encoder, o_arm)
                fused = _sigmoid(apply_crc(rgb_logit, probability_logit(gp), calibration))
                metrics.extend([{'encoder': encoder, 'arm_id': o_arm, 'route': 'patient', 'geometry_patients_excluded_under_10_tiles': len(extra), **_crc_metrics(label, gp, thresholds[encoder, o_arm])}, {'encoder': encoder, 'arm_id': r_arm, 'route': 'patient', 'geometry_patients_excluded_under_10_tiles': len(extra), 'alpha': calibration.alpha, 'rgb_temperature': calibration.rgb_temperature, 'geometry_temperature': calibration.geometry_temperature, **_crc_metrics(label, fused, thresholds[encoder, r_arm])}])
                for arm, score in ((o_arm, gp), (r_arm, fused)):
                    predictions.extend(({'encoder': encoder, 'arm_id': arm, 'patient_id': key, 'label_id': int(value), 'score': float(probability), 'tiles': common_rgb[key]['tiles'], 'route': 'patient'} for key, value, probability in zip(keys, label, score)))
        tile_metrics, tile_predictions = ([], [])
        tile_thresholds = _validation_tile_thresholds(cfg)
        tile_reference = [{'patient_id': patient, 'label_id': int(value['label_id']), 'score': float(value['score']), 'tiles': int(value['tiles'])} for patient, value in sorted(common_rgb.items())]
        tile_ref_labels = np.asarray([row['label_id'] for row in tile_reference])
        tile_ref_scores = np.asarray([row['score'] for row in tile_reference])
        tile_metrics.append({'encoder': 'paper', 'arm_id': 'R0', 'route': 'tile_geometry_eligible_reference', **_crc_metrics(tile_ref_labels, tile_ref_scores, tile_thresholds['paper', 'R0'])})
        tile_predictions.extend(({'encoder': 'paper', 'arm_id': 'R0', 'route': 'tile', **row} for row in tile_reference))
        validation_paper = [row for row in _paper_seed_average(cfg) if row['graph_id'] in _geometry_eligible_ids(cfg)]
        validation_map = {(row['fold'], row['graph_id']): row for row in validation_paper}
        for encoder in ENCODERS:
            for o_arm in [arm for arm in ARMS if arm.startswith('O')]:
                test_geo = {row['graph_id']: row for row in _geometry_test(cfg, encoder, o_arm, tile=True)}
                val_geo = {(row['fold'], row['graph_id']): row for row in _tile_geometry_seed_average(cfg, encoder, o_arm)}
                vkeys = sorted(validation_map)
                val_labels = np.asarray([validation_map[key]['label_id'] for key in vkeys])
                val_rgb = probability_logit(np.asarray([validation_map[key]['probabilities'][1] for key in vkeys]))
                val_g = probability_logit(np.asarray([val_geo[key]['score'] for key in vkeys]))
                fit_rows = [{'patient_id': validation_map[key]['patient_id'], 'label_id': int(label)} for key, label in zip(vkeys, val_labels)]
                calibration = fit_crc_calibration(val_rgb, val_g, val_labels, sample_weight=_tile_calibration_weight(fit_rows))
                graphs = sorted(test_geo)
                fused_tile = apply_crc(probability_logit(np.asarray([paper_map[key]['probabilities'][1] for key in graphs])), probability_logit(np.asarray([test_geo[key]['probability'] for key in graphs])), calibration) >= 0
                pg: dict[str, list[int]] = defaultdict(list)
                for index, graph in enumerate(graphs):
                    pg[patient_by_graph[graph]].append(index)
                rows = []
                for patient, indices in sorted(pg.items()):
                    if len(indices) < 10:
                        continue
                    rows.append({'patient_id': patient, 'label_id': labels_by_graph[graphs[indices[0]]], 'score': float(np.mean(fused_tile[indices])), 'tiles': len(indices)})
                label = np.asarray([row['label_id'] for row in rows])
                score = np.asarray([row['score'] for row in rows])
                r_arm = 'R' + o_arm[1:]
                tile_metrics.append({'encoder': encoder, 'arm_id': r_arm, 'route': 'tile', 'alpha': calibration.alpha, 'rgb_temperature': calibration.rgb_temperature, 'geometry_temperature': calibration.geometry_temperature, **_crc_metrics(label, score, tile_thresholds[encoder, r_arm])})
                tile_predictions.extend(({'encoder': encoder, 'arm_id': r_arm, 'route': 'tile', **row} for row in rows))
        atomic_parquet(root / 'official_test' / 'evaluation' / 'tile_metrics.parquet', tile_metrics)
        atomic_parquet(root / 'official_test' / 'evaluation' / 'tile_predictions.parquet', tile_predictions)
        tile_statistics = _official_crc_tile_statistics(cfg, tile_predictions)
    output = root / 'official_test' / 'evaluation'
    atomic_parquet(output / 'metrics.parquet', metrics)
    atomic_parquet(output / 'predictions.parquet', predictions)
    components = _component_metrics(cfg, labels_by_graph, patient_by_graph, thresholds)
    atomic_parquet(output / 'component_metrics.parquet', components)
    primary_name = 'QWK' if cfg['dataset'] == 'sicapv2' else 'patient_AUROC'
    grouped_components: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in components:
        if row['component'] == 'seed' and primary_name in row:
            grouped_components[str(row['encoder']), str(row['arm_id'])].append(float(row[primary_name]))
    seed_summary = []
    for (encoder, arm), values in sorted(grouped_components.items()):
        if len(values) != len(SEEDS):
            raise RuntimeError(f'official five-seed metric coverage mismatch: {encoder}/{arm}')
        seed_summary.append({'encoder': encoder, 'arm_id': arm, 'metric': primary_name, 'seed_values': values, 'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)), 'seeds': list(SEEDS)})
    atomic_parquet(output / 'seed_mean_std.parquet', seed_summary)
    statistics = _official_statistics(cfg, predictions)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'transaction_ready_sha256': sha256(root / 'official_test' / 'transaction_ready.json'), 'metrics': metrics, 'component_metrics_rows': len(components), 'seed_summary_rows': len(seed_summary), 'ncr_qc': ncr_qc, 'statistics': statistics, 'tile_statistics': tile_statistics}
    atomic_json(output / 'summary.json', payload)
    return payload
