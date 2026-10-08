from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .fuse import pred_path
from .io import job_dir
from .jobs import job_matrix, baseline_job_matrix
from .metrics import macro_f1, patient_auroc, qwk, weighted_f1
from .paths import load_config, baseline_result_root, result_root
from .splits import load_units
EXPECTED_TEST_GRAPHS = {'sicapv2': 2122, 'bracs': 1160, 'tcga_crc_msi': 32360}
EXPECTED_TEST_BAGS = {'sicapv2': 2122, 'bracs': 570, 'tcga_crc_msi': 140}

def _bootstrap(values: np.ndarray, rng: np.random.Generator, n: int=1000) -> tuple[float, float]:
    if values.size == 0:
        return (float('nan'), float('nan'))
    stats = []
    for _ in range(n):
        idx = rng.integers(0, values.size, values.size)
        stats.append(float(values[idx].mean()))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return (float(lo), float(hi))

def _metric_pack(dataset: str, labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    probs = np.asarray(probs, np.float64)
    if dataset == 'sicapv2':
        pred = probs.argmax(1)
        return {'qwk': qwk(labels, pred), 'macro_f1': macro_f1(labels, pred, probs.shape[1])}
    if dataset == 'bracs':
        pred = probs.argmax(1)
        return {'macro_f1': macro_f1(labels, pred, 7), 'weighted_f1': weighted_f1(labels, pred, 7)}
    score = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.reshape(-1)
    return {'patient_auroc': patient_auroc(labels, score)}

def _test_ids(dataset: str) -> list[str]:
    units = load_units(dataset)
    if dataset == 'sicapv2':
        full = next((unit for unit in units if unit['kind'] == 'sicap_full_train'))
        return list(map(str, full.get('test') or []))
    if dataset == 'tcga_crc_msi':
        from .crc_score import qualified_patient_ids
        return qualified_patient_ids('test')
    return list(map(str, units[0].get('test') or []))

def _load_pred(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from .fuse import _as_probability
    rows = pq.read_table(path).to_pylist()
    out = []
    for row in rows:
        item = dict(row)
        item['probability'] = _as_probability(item)
        out.append(item)
    return out

def _apply_locked_fusion(b_rows, e_rows, weights: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    b_map = {str(row['sample_id']): row for row in b_rows}
    e_map = {str(row['sample_id']): row for row in e_rows}
    keys = sorted(set(b_map) & set(e_map))
    a = float(weights.get('a', 0.0))
    temperature = float(weights.get('temperature', 1.0))
    out = []
    for key in keys:
        b = np.clip(np.asarray(b_map[key]['probability'], np.float64), 1e-06, 1)
        e = np.clip(np.asarray(e_map[key]['probability'], np.float64), 1e-06, 1)
        if b.shape[0] == 2:
            logit_b = np.log(b[1]) - np.log(b[0])
            logit_e = np.log(e[1]) - np.log(e[0])
            centered = logit_e / temperature
            fused = 1 / (1 + np.exp(-(logit_b + a * (centered - 0.0))))
            prob = [1 - float(fused), float(fused)]
        else:
            logit_b = np.log(b) - np.log(b).mean()
            logit_e = np.log(e) - np.log(e).mean()
            z = logit_b + a * logit_e
            z = z - z.max()
            p = np.exp(z)
            prob = (p / p.sum()).tolist()
        out.append({'sample_id': key, 'label_id': int(b_map[key]['label_id']), 'probability': prob})
    return out

def evaluate_job_on_test(job: dict[str, Any], device: str) -> dict[str, Any] | None:
    destination = job_dir(job)
    manifest_path = destination / 'job.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') not in {'PASS', 'REUSED'}:
        return None
    test_path = destination / 'test_predictions.parquet'
    if test_path.is_file():
        rows = _load_pred(test_path)
        probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0) if rows else np.zeros((0, 1))
        labels = np.asarray([int(row['label_id']) for row in rows]) if rows else np.zeros((0,), np.int64)
        return {'job_id': job['job_id'], 'n': len(rows), 'metrics': _metric_pack(job['dataset'], labels, probs), 'path': str(test_path)}
    if job['family'] == 'fusion':
        return _evaluate_fusion_test(job, destination)
    from .predict import predict_test
    rows = predict_test(job, device)
    if rows is None:
        return {'job_id': job['job_id'], 'status': 'SKIP', 'reason': 'predictor_unavailable'}
    from .io import atomic_parquet
    return None

def _resolve_test_pred(job_like: dict[str, Any]) -> Path | None:
    prefer_baseline = bool(job_like.get('baseline_qualification'))
    order = (True, False) if prefer_baseline else (False, True)
    for flag in order:
        path = job_dir({**job_like, 'baseline_qualification': flag}) / 'test_predictions.parquet'
        if path.is_file():
            return path
    return None

def _evaluate_fusion_test(job: dict[str, Any], destination: Path) -> dict[str, Any] | None:
    weights_path = destination / 'weights.json'
    if not weights_path.is_file():
        return None
    weights = json.loads(weights_path.read_text(encoding='utf-8'))
    dataset = job['dataset']
    if dataset == 'sicapv2':
        b_test = _resolve_test_pred({'dataset': dataset, 'family': 'baseline', 'arm': 'B', 'unit': 'full_train', 'seed': 42, 'fold': 0, 'baseline_qualification': job.get('baseline_qualification')})
        e_test = _resolve_test_pred({'dataset': dataset, 'family': job['expert_family'], 'arm': job['expert_arm'], 'unit': 'full_train', 'seed': 42, 'fold': 0})
        if b_test is None or e_test is None:
            return {'job_id': job['job_id'], 'status': 'WAIT', 'reason': 'sicap_retrain_test'}
        fused = _apply_locked_fusion(_load_pred(b_test), _load_pred(e_test), weights, dataset)
    else:
        partner_baseline = bool(job.get('baseline_qualification') and job.get('partner_family') == 'baseline')
        b_test = _resolve_test_pred({'dataset': dataset, 'family': job['partner_family'], 'arm': job['partner_arm'], 'unit': job['unit'], 'seed': job['seed'], 'fold': int(job.get('fold') or 0), 'baseline_qualification': partner_baseline})
        e_test = _resolve_test_pred({'dataset': dataset, 'family': job['expert_family'], 'arm': job['expert_arm'], 'unit': job['unit'], 'seed': job['seed'], 'fold': int(job.get('fold') or 0)})
        if b_test is None or e_test is None:
            return {'job_id': job['job_id'], 'status': 'WAIT', 'reason': 'component_test'}
        fused = _apply_locked_fusion(_load_pred(b_test), _load_pred(e_test), weights, dataset)
        if dataset == 'tcga_crc_msi':
            from .crc_score import filter_to_qualified
            fused = filter_to_qualified(fused, 'test')
    from celllift.morphology_interaction.io_utils import atomic_parquet
    atomic_parquet(destination / 'test_predictions.parquet', fused)
    probs = np.stack([np.asarray(row['probability'], np.float64) for row in fused], 0)
    labels = np.asarray([int(row['label_id']) for row in fused])
    metrics = _metric_pack(dataset, labels, probs)
    (destination / 'test_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return {'job_id': job['job_id'], 'n': len(fused), 'metrics': metrics, 'fusion_locked': True, 'test_used_for_fit': False}

def test_cache_ready(dataset: str) -> tuple[bool, str]:
    from celllift.morphology_interaction.dataset import cache_index_path, cache_root
    cfg = load_config(dataset)
    root = cache_root(cfg, dataset) / 'official_test'
    expected_shards = {'sicapv2': 2, 'bracs': 2, 'tcga_crc_msi': 4}[dataset]
    expected = EXPECTED_TEST_GRAPHS[dataset]
    graphs = 0
    for shard in range(expected_shards):
        manifest = root / f'shard_{shard:03d}' / 'manifest.json'
        if not manifest.is_file():
            return (False, f'missing_manifest:{shard}')
        graphs += int(json.loads(manifest.read_text(encoding='utf-8')).get('graphs') or 0)
    if graphs < expected:
        return (False, f'packed_{graphs}_lt_{expected}')
    index_path = cache_index_path(cfg, dataset)
    if not index_path.is_file():
        return (False, 'missing_scene_index')
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    tagged = sum((1 for row in payload.get('graphs') or [] if (row.get('metadata') or {}).get('official_split') == 'test' or (row.get('metadata') or {}).get('split') == 'test'))
    recorded = int(payload.get('graphs_official_test') or 0)
    if max(tagged, recorded) < expected:
        return (False, f'not_merged:{max(tagged, recorded)}')
    return (True, 'ok')

def ensure_test_merged(dataset: str) -> dict[str, Any]:
    ready, why = test_cache_ready(dataset)
    if ready:
        return {'dataset': dataset, 'status': 'READY'}
    if why.startswith('not_merged'):
        from .pack_official_test import merge_test
        return {'dataset': dataset, 'status': 'MERGED', **merge_test(dataset)}
    return {'dataset': dataset, 'status': 'WAIT', 'reason': why}

def _complete_test_rows(job: dict[str, Any], rows: list[dict[str, Any]], exact: bool=False) -> tuple[bool, str]:
    expected_ids = _test_ids(job['dataset'])
    got = {str(row['sample_id']) for row in rows}
    missing = [sid for sid in expected_ids if sid not in got]
    if job['dataset'] == 'tcga_crc_msi' and (not exact):
        if missing:
            return (False, f'missing_{len(missing)}_of_{len(expected_ids)}')
        return (True, 'ok')
    if missing:
        return (False, f'missing_{len(missing)}_of_{len(expected_ids)}')
    extra = sorted(got - set(expected_ids))
    if extra:
        return (False, f'extra_{len(extra)}_not_in_qualified')
    return (True, 'ok')

def evaluate_all(device: str='cuda', datasets: list[str] | None=None) -> dict[str, Any]:
    from .predict import predict_test
    from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet
    wanted = set(datasets) if datasets else None
    merge_status = {}
    for dataset in ('sicapv2', 'bracs', 'tcga_crc_msi'):
        if wanted and dataset not in wanted:
            continue
        merge_status[dataset] = ensure_test_merged(dataset)
    jobs = job_matrix()
    results = []
    for job in jobs:
        if wanted and job['dataset'] not in wanted:
            continue
        if job['family'] == 'fusion':
            continue
        destination = job_dir(job)
        if not (destination / 'job.json').is_file():
            continue
        test_path = destination / 'test_predictions.parquet'
        if not test_path.is_file():
            if job['family'] != 'baseline':
                cache_ok, cache_why = test_cache_ready(job['dataset'])
                if not cache_ok:
                    results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': cache_why})
                    continue
            try:
                rows = predict_test(job, device)
            except Exception as error:
                results.append({'job_id': job['job_id'], 'status': 'FAILED', 'error': repr(error)})
                continue
            if not rows:
                results.append({'job_id': job['job_id'], 'status': 'SKIP'})
                continue
            complete, why = _complete_test_rows(job, rows)
            if not complete:
                results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': why, 'n': len(rows)})
                continue
            atomic_parquet(test_path, rows)
        rows = _load_pred(test_path)
        complete, why = _complete_test_rows(job, rows)
        if not complete:
            results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': f'existing_{why}', 'n': len(rows)})
            continue
        probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0)
        labels = np.asarray([int(row['label_id']) for row in rows])
        metrics = _metric_pack(job['dataset'], labels, probs)
        atomic_json(destination / 'test_metrics.json', metrics)
        results.append({'job_id': job['job_id'], 'status': 'PASS', 'metrics': metrics, 'n': len(rows)})
    for job in jobs:
        if wanted and job['dataset'] not in wanted:
            continue
        if job['family'] != 'fusion':
            continue
        results.append(_evaluate_fusion_test(job, job_dir(job)) or {'job_id': job['job_id'], 'status': 'WAIT'})
    tables = write_tables(results)
    return {'n': len(results), 'merge': merge_status, 'tables': tables, 'results': results}

def write_tables(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row.get('job_id'): row for row in results if row}
    jobs = job_matrix()
    for job in jobs:
        payload = by_id.get(job['job_id']) or {}
        if payload.get('metrics'):
            continue
        metrics_path = job_dir(job) / 'test_metrics.json'
        if metrics_path.is_file():
            by_id[job['job_id']] = {'job_id': job['job_id'], 'status': 'PASS', 'metrics': json.loads(metrics_path.read_text(encoding='utf-8'))}
    complete_rows = []
    geom_rows = []
    input_type = {'B': 'image', 'B_rgb': 'image', 'G2': '2D-mask-only', 'G3': 'raw3D-only', 'GR': 'residual3D-only', 'G2R': '2D+raw3D', 'G2S': '2D+residual3D', 'MP-G2': '2D-mask-only', 'MP-G3': 'raw3D-only', 'E2': '2D-mask-only', 'ES': '2D+raw3D', 'ER': '2D+residual3D', 'A2': 'image+geometry', 'AS': 'image+geometry', 'C2': 'image+geometry', 'C3': 'image+geometry', 'H2': 'image+geometry', 'H3': 'image+geometry', 'D2': 'train-time-3D-supervision', 'D3': 'train-time-3D-supervision', 'B2': 'image+geometry', 'B3': 'image+geometry', 'C1': 'image+geometry', 'C4': 'image+geometry'}

    def metric_of(job):
        payload = by_id.get(job['job_id']) or {}
        return payload.get('metrics') or {}

    def pick(dataset, family, arm, unit=None):
        for job in jobs:
            if job['dataset'] != dataset or job['family'] != family:
                continue
            if (job.get('arm_tag') or job['arm']) != arm:
                continue
            if unit and job.get('unit') != unit:
                continue
            if dataset == 'sicapv2' and job.get('unit') != 'full_train' and (family != 'fusion'):
                continue
            return (job, metric_of(job))
        return (None, {})
    for dataset, b_arm, pairs in (('sicapv2', 'B', [('A2', 'AS', 'S-A'), ('C2', 'C3', 'S-C')]), ('bracs', 'B_rgb', [('H2', 'H3', 'B-H'), ('D2', 'D3', 'B-D')]), ('tcga_crc_msi', 'B', [('B2', 'B3', 'C-P'), ('C1', 'C4', 'C-H')])):
        unit = 'full_train' if dataset == 'sicapv2' else 'official'
        _, b_m = pick(dataset, 'baseline', b_arm, unit)
        for f2, f3, route in pairs:
            family = 'route' if f2 in {'A2', 'AS', 'C2', 'C3', 'D2', 'D3', 'B2', 'B3'} else 'feature'
            _, m2 = pick(dataset, family, f2, unit)
            _, m3 = pick(dataset, family, f3, unit)
            complete_rows.append({'dataset': dataset, 'route': route, 'B': b_m, 'F2': m2, 'F3': m3, 'F3>B': _gt(m3, b_m, dataset), 'F3>F2': _gt(m3, m2, dataset), 'input_F2': input_type.get(f2), 'input_F3': input_type.get(f3), 'b_name': 'B_rgb' if dataset == 'bracs' else 'B'})
        for arm in ('G2', 'G3', 'GR'):
            fam = 'fusion'
            tag = f'{b_arm}+{arm}'
            _, fused = pick(dataset, fam, tag, 'fold_00' if dataset == 'sicapv2' else 'official')
            complete_rows.append({'dataset': dataset, 'route': 'fusion', 'arm': tag, 'metrics': fused, 'input': 'image+geometry-scores', 'fit_on_test': False})
    for dataset, family, arms in (('sicapv2', 'geometry', ('G2', 'G3', 'GR', 'G2R', 'G2S')), ('bracs', 'geometry', ('G2', 'G3', 'GR')), ('bracs', 'expert', ('E2', 'ES', 'ER')), ('tcga_crc_msi', 'geometry', ('G2', 'G3', 'GR', 'MP-G2', 'MP-G3'))):
        unit = 'full_train' if dataset == 'sicapv2' else 'official'
        metrics = {}
        for arm in arms:
            _, metrics[arm] = pick(dataset, family, arm, unit)
        g3 = metrics.get('G3') or metrics.get('ER') or metrics.get('MP-G3') or {}
        g2 = metrics.get('G2') or metrics.get('E2') or metrics.get('MP-G2') or {}
        geom_rows.append({'dataset': dataset, 'family': family, 'metrics': metrics, 'G3>G2': _gt(g3, g2, dataset), 'below_B_allowed': True, 'input': {arm: input_type.get(arm) for arm in arms}})
    payload = {'complete': complete_rows, 'geometry': geom_rows, 'seed': 42, 'protocol': 'benchmark_prediction_seed42'}
    out = result_root() / 'tables' / 'main_tables.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')
    return payload

def _gt(a: dict, b: dict, dataset: str) -> bool | None:
    key = 'qwk' if dataset == 'sicapv2' else 'macro_f1' if dataset == 'bracs' else 'patient_auroc'
    if not a or not b or a.get(key) is None or (b.get(key) is None):
        return None
    return float(a[key]) > float(b[key])
DISPLAY_NAME = {'E2': 'Geo2D', 'ES': 'Geo2D+raw3D', 'ER': 'Geo2D+res3D', 'G2': 'G2 / Geo2D', 'G3': 'G3 / raw3D', 'GR': 'GR / res3D (2D-conditioned residual)', 'G2R': 'G2+raw3D', 'G2S': 'G2+residual3D', 'B_rgb': 'B_rgb (local RGB)', 'B_rgb+E2': 'B_rgb+Geo2D', 'B_rgb+ES': 'B_rgb+Geo2D+raw3D', 'B_rgb+ER': 'B_rgb+Geo2D+res3D', 'B_rgb+G2': 'B_rgb+Geo2D', 'B_rgb+G3': 'B_rgb+raw3D', 'B_rgb+GR': 'B_rgb+res3D', 'B_paper200': 'B (paper 200-epoch recipe)', 'P2': 'P2 (B2)', 'P2+G2': 'P2+G2', 'P2+G3': 'P2+G3', 'P2+GR': 'P2+GR'}

def _score(metrics: dict[str, Any], dataset: str) -> float | None:
    key = 'qwk' if dataset == 'sicapv2' else 'macro_f1' if dataset == 'bracs' else 'patient_auroc'
    if not metrics or metrics.get(key) is None:
        return None
    return float(metrics[key])

def _metrics_from_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    if dataset == 'tcga_crc_msi':
        from .crc_score import filter_to_qualified
        rows = filter_to_qualified(rows, 'test')
    if not rows:
        return {}
    probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0)
    labels = np.asarray([int(row['label_id']) for row in rows])
    return _metric_pack(dataset, labels, probs)

def _lookup_test_metrics(job: dict[str, Any]) -> dict[str, float]:
    baseline_dest = job_dir({**job, 'baseline_qualification': True})
    compatibility_dest = job_dir({**job, 'baseline_qualification': False})
    for dest in (baseline_dest, compatibility_dest):
        path = dest / 'test_predictions.parquet'
        if path.is_file():
            return _metrics_from_rows(job['dataset'], _load_pred(path))
    for dest in (baseline_dest, compatibility_dest):
        metrics_path = dest / 'test_metrics.json'
        if metrics_path.is_file():
            return json.loads(metrics_path.read_text(encoding='utf-8'))
    return {}

def write_baseline_tables(extra: dict[str, Any] | None=None) -> dict[str, Any]:
    extra = extra or {}
    jobs = job_matrix() + baseline_job_matrix()

    def pick(dataset, family, arm, unit=None, r2=None):
        for job in jobs:
            if job['dataset'] != dataset or job['family'] != family:
                continue
            if (job.get('arm_tag') or job['arm']) != arm:
                continue
            if unit and job.get('unit') != unit:
                continue
            if r2 is True and (not job.get('baseline_qualification')):
                continue
            if dataset == 'sicapv2' and job.get('unit') != 'full_train' and (family not in {'fusion'}):
                continue
            metrics = extra.get(job['job_id']) or _lookup_test_metrics(job)
            return (job, metrics)
        return (None, {})
    table_a = []
    for dataset, family, arms in (('sicapv2', 'geometry', ('G2', 'G3', 'GR', 'G2R')), ('bracs', 'geometry', ('G2', 'G3', 'GR')), ('bracs', 'expert', ('E2', 'ES', 'ER')), ('tcga_crc_msi', 'geometry', ('G2', 'G3', 'GR'))):
        unit = 'full_train' if dataset == 'sicapv2' else 'official'
        row = {'dataset': dataset, 'family': family, 'below_B_allowed': True, 'methods': {}}
        for arm in arms:
            _, metrics = pick(dataset, family, arm, unit)
            row['methods'][arm] = {'display': DISPLAY_NAME.get(arm, arm), 'metrics': metrics, 'score': _score(metrics, dataset)}
        g3 = row['methods'].get('G3') or row['methods'].get('ER') or {}
        g2 = row['methods'].get('G2') or row['methods'].get('E2') or {}
        row['G3_minus_G2'] = None
        if g3.get('score') is not None and g2.get('score') is not None:
            row['G3_minus_G2'] = float(g3['score']) - float(g2['score'])
            row['G3>G2'] = float(g3['score']) > float(g2['score'])
        if 'GR' in row['methods']:
            row['methods']['GR']['note'] = '2D-conditioned residual; not a raw-3D replacement'
        table_a.append(row)
    table_b = []
    sicap_b, sicap_b_m = pick('sicapv2', 'baseline', 'B', 'full_train')
    _, sicap_paper = pick('sicapv2', 'baseline', 'B_paper200', 'full_train', r2=True)
    _, a2 = pick('sicapv2', 'route', 'A2', 'full_train')
    _, as_ = pick('sicapv2', 'route', 'AS', 'full_train')
    table_b.append({'dataset': 'sicapv2', 'B': {'display': 'B (local official-split)', 'metrics': sicap_b_m, 'score': _score(sicap_b_m, 'sicapv2')}, 'B_paper200': {'display': DISPLAY_NAME['B_paper200'], 'metrics': sicap_paper, 'score': _score(sicap_paper, 'sicapv2')}, 'F2': {'display': 'A2 (strongest 2D image+cell)', 'metrics': a2, 'score': _score(a2, 'sicapv2')}, 'F3': {'display': 'AS', 'metrics': as_, 'score': _score(as_, 'sicapv2')}, 'F3>B': _gt(as_, sicap_b_m, 'sicapv2'), 'F3>F2': _gt(as_, a2, 'sicapv2'), 'claim_task_increment': False})
    _, bracs_b = pick('bracs', 'baseline', 'B_rgb', 'official')
    _, bracs_e2 = pick('bracs', 'fusion', 'B_rgb+E2', 'official')
    _, bracs_er = pick('bracs', 'fusion', 'B_rgb+ER', 'official')
    _, bracs_h2 = pick('bracs', 'feature', 'H2', 'official')
    _, bracs_h3 = pick('bracs', 'feature', 'H3', 'official')
    table_b.append({'dataset': 'bracs', 'B': {'display': DISPLAY_NAME['B_rgb'], 'metrics': bracs_b, 'score': _score(bracs_b, 'bracs'), 'source': 'local_rgb'}, 'F2': {'display': DISPLAY_NAME['B_rgb+E2'], 'metrics': bracs_e2, 'score': _score(bracs_e2, 'bracs')}, 'F3': {'display': DISPLAY_NAME['B_rgb+ER'], 'metrics': bracs_er, 'score': _score(bracs_er, 'bracs')}, 'H2': {'display': 'H2', 'metrics': bracs_h2, 'score': _score(bracs_h2, 'bracs')}, 'H3': {'display': 'H3', 'metrics': bracs_h3, 'score': _score(bracs_h3, 'bracs')}, 'official_hact': extra.get('hact') or {'status': 'Incomplete', 'reason': 'source_weights_unmatched_or_unchecked'}, 'F3>B': _gt(bracs_er, bracs_b, 'bracs'), 'F3>F2': _gt(bracs_er, bracs_e2, 'bracs')})
    crc_b_job, crc_b = pick('tcga_crc_msi', 'baseline', 'B', 'official', r2=True)
    if not crc_b:
        _, crc_b = pick('tcga_crc_msi', 'baseline', 'B', 'official')
    _, crc_b2 = pick('tcga_crc_msi', 'route', 'B2', 'official')
    p2_rows = {}
    for arm in ('P2', 'P2+G2', 'P2+G3', 'P2+GR'):
        _, p2_rows[arm] = pick('tcga_crc_msi', 'fusion', arm, 'official', r2=True)
    if not p2_rows['P2']:
        p2_rows['P2'] = crc_b2
    bg = {}
    for arm in ('G2', 'G3', 'GR'):
        _, bg[arm] = pick('tcga_crc_msi', 'fusion', f'B+{arm}', 'official', r2=True)
        if not bg[arm]:
            _, bg[arm] = pick('tcga_crc_msi', 'fusion', f'B+{arm}', 'official')
    p2_s = _score(p2_rows['P2'], 'tcga_crc_msi')
    p2g3_s = _score(p2_rows['P2+G3'], 'tcga_crc_msi')
    p2g2_s = _score(p2_rows['P2+G2'], 'tcga_crc_msi')
    strong_3d = p2g3_s is not None and p2_s is not None and (p2g2_s is not None) and (p2g3_s > p2_s) and (p2g3_s > p2g2_s)
    table_b.append({'dataset': 'tcga_crc_msi', 'B': {'display': 'B (hard-vote, qualified ≥10 tiles)', 'metrics': crc_b, 'score': _score(crc_b, 'tcga_crc_msi')}, 'P2': {'display': DISPLAY_NAME['P2'], 'metrics': p2_rows['P2'], 'score': p2_s}, 'P2+G2': {'display': DISPLAY_NAME['P2+G2'], 'metrics': p2_rows['P2+G2'], 'score': p2g2_s}, 'P2+G3': {'display': DISPLAY_NAME['P2+G3'], 'metrics': p2_rows['P2+G3'], 'score': p2g3_s}, 'P2+GR': {'display': DISPLAY_NAME['P2+GR'], 'metrics': p2_rows['P2+GR'], 'score': _score(p2_rows['P2+GR'], 'tcga_crc_msi')}, 'B+G2': {'metrics': bg['G2'], 'score': _score(bg['G2'], 'tcga_crc_msi')}, 'B+G3': {'metrics': bg['G3'], 'score': _score(bg['G3'], 'tcga_crc_msi')}, 'B+GR': {'metrics': bg['GR'], 'score': _score(bg['GR'], 'tcga_crc_msi')}, 'strong_3d_on_P2': strong_3d, 'claim': 'strong 3D increment on P2' if strong_3d else 'keep B2/P2 stronger; no 3D-on-strong-2D claim', 'seed': 42})
    payload = {'table_a': table_a, 'table_b': table_b, 'seed': 42, 'protocol': 'baseline_qualification_seed42', 'qualification': extra.get('qualification'), 'before_after': extra.get('before_after')}
    out = baseline_result_root() / 'tables' / 'main_tables.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')
    return payload

def evaluate_baseline(device: str='cuda') -> dict[str, Any]:
    from .crc_score import qualification_diff
    from .predict import materialize_val_predictions, predict_test
    from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet
    qualification = qualification_diff()
    results = []
    for job in baseline_job_matrix():
        if job['family'] == 'fusion':
            continue
        dest = job_dir(job)
        if not (dest / 'best.pt').is_file() and (not (dest / 'job.json').is_file()):
            results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': 'no_r2_ckpt'})
            continue
        test_path = dest / 'test_predictions.parquet'
        if not test_path.is_file():
            rows = predict_test(job, device)
            if rows:
                complete, why = _complete_test_rows(job, rows)
                if complete:
                    atomic_parquet(test_path, rows)
                    atomic_json(dest / 'test_metrics.json', _metric_pack(job['dataset'], np.asarray([int(r['label_id']) for r in rows]), np.stack([np.asarray(r['probability'], np.float64) for r in rows], 0)))
                    results.append({'job_id': job['job_id'], 'status': 'PASS', 'n': len(rows)})
                else:
                    results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': why, 'n': len(rows)})
            else:
                results.append({'job_id': job['job_id'], 'status': 'SKIP'})
        else:
            results.append({'job_id': job['job_id'], 'status': 'EXISTS'})
    b2_job = {'dataset': 'tcga_crc_msi', 'family': 'route', 'arm': 'B2', 'unit': 'official', 'seed': 42, 'fold': 0, 'baseline_qualification': True, 'route': 'b', 'official_val': None}
    results.append({'b2_val': materialize_val_predictions({**b2_job, 'refresh_val': False}, device)})
    for job in baseline_job_matrix():
        if job['family'] != 'fusion':
            continue
        dest = job_dir(job)
        if not (dest / 'weights.json').is_file():
            results.append({'job_id': job['job_id'], 'status': 'WAIT', 'reason': 'fusion_not_fit'})
            continue
        results.append(_evaluate_fusion_test(job, dest) or {'job_id': job['job_id'], 'status': 'WAIT'})
    tables = write_baseline_tables({'qualification': qualification})
    return {'qualification': qualification, 'results': results, 'tables': tables}
