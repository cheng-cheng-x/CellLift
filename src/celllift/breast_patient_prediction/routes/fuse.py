from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
import numpy as np
from ..io_utils import atomic_json, atomic_parquet, read_json
from .config import COMPLETE_3D, FUSION_EXPERTS, MATCHING_2D, TASK_SPEC, TASKS, active_docs, arm_dir, fuse_dir, is_conditional_geometry, protocol_meta, resolved_arm_dir, tables_dir
PAIR_DIFFS = (('H3', 'H2'), ('P3', 'P2'), ('G3', 'G2'), ('GR', 'G2'), ('G23', 'G2'), ('G2R', 'G2'), ('ES', 'E2'), ('ER', 'ES'), ('B+G3', 'B+G2'), ('B+GR', 'B+G2'), ('B+G23', 'B+G2'), ('B+G2R', 'B+G2'), ('B+ES', 'B+E2'), ('B+ER', 'B+ES'))
from .evaluate import better, paired_bootstrap, primary_from_probs, binary_threshold

def _load_preds(path: Path) -> dict[str, dict]:
    import pandas as pd
    frame = pd.read_parquet(path)
    rows = {}
    for raw in frame.to_dict('records'):
        rows[str(raw['patient_id'])] = {'patient_id': str(raw['patient_id']), 'split': str(raw['split']).lower(), 'label': int(raw['label']), 'prob': np.asarray(raw['prob'], np.float64)}
    return rows

def _aligned(left: dict, right: dict, split: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    left_ids = {pid for pid, row in left.items() if row['split'] == split}
    right_ids = {pid for pid, row in right.items() if row['split'] == split}
    if left_ids != right_ids:
        raise RuntimeError(f'fusion {split} ids differ: {len(left_ids)} vs {len(right_ids)}')
    ids = sorted(left_ids)
    for pid in ids:
        if int(left[pid]['label']) != int(right[pid]['label']) or left[pid]['split'] != right[pid]['split']:
            raise RuntimeError(f'fusion label/split mismatch for {pid}')
    y = np.asarray([left[pid]['label'] for pid in ids], np.int64)
    a = np.stack([left[pid]['prob'] for pid in ids], 0)
    b = np.stack([right[pid]['prob'] for pid in ids], 0)
    return (ids, y, a, b)

def _select_alpha(task: str, y: np.ndarray, p_b: np.ndarray, p_g: np.ndarray) -> tuple[float, float]:
    best_a, best = (0.0, float('-inf'))
    for step in range(21):
        alpha = step / 20.0
        fused = (1.0 - alpha) * p_b + alpha * p_g
        score, _ = primary_from_probs(task, y, fused)
        if better(score, best) or (np.isfinite(score) and abs(score - best) <= 1e-12 and (alpha < best_a)):
            best, best_a = (float(score), float(alpha))
    return (best_a, best)

def fuse_one(task: str, expert: str) -> dict:
    dest = fuse_dir(task, expert)
    marker = dest / 'metrics.json'
    if marker.is_file():
        return {'status': 'skip', 'path': str(marker)}
    dest.mkdir(parents=True, exist_ok=True)
    base_b = resolved_arm_dir(task, 'B') / 'predictions.parquet'
    base_g = resolved_arm_dir(task, expert) / 'predictions.parquet'
    if not base_b.is_file() or not base_g.is_file():
        raise RuntimeError(f'missing predictions for {task} B+{expert}')
    pb = _load_preds(base_b)
    pg = _load_preds(base_g)
    ids_val, y_val, b_val, g_val = _aligned(pb, pg, 'val')
    alpha, val_score = _select_alpha(task, y_val, b_val, g_val)
    spec = TASK_SPEC[task]
    fused_by_split = {}
    labels = {}
    ids_by_split = {}
    for split in ('fit', 'val', 'test'):
        ids, y, b, g = _aligned(pb, pg, split)
        fused_by_split[split] = (1.0 - alpha) * b + alpha * g
        labels[split] = y
        ids_by_split[split] = ids
    threshold = None
    if spec['classes'] == 2:
        threshold = binary_threshold(labels['val'], fused_by_split['val'][:, 1])
    rows_out = []
    metrics = {}
    for split in ('fit', 'val', 'test'):
        primary, aux = primary_from_probs(task, labels[split], fused_by_split[split], threshold)
        metrics[split] = {'primary': primary, **aux}
        for pid, prob, label in zip(ids_by_split[split], fused_by_split[split], labels[split]):
            rows_out.append({'patient_id': pid, 'split': split, 'label': int(label), 'prob': prob.astype(np.float64).tolist(), 'alpha': alpha})
    atomic_parquet(dest / 'predictions.parquet', rows_out)
    payload = {**protocol_meta(), 'status': 'PASS', 'task': task, 'method': f'B+{expert}', 'alpha': alpha, 'geometry_used': bool(alpha > 0), 'val_primary_at_alpha': val_score, 'threshold': threshold, 'metrics': metrics, 'n_val': len(ids_val)}
    atomic_json(marker, payload)
    return payload

def fuse_ready(task: str | None=None) -> list[dict]:
    from .schedule import finish_claim, try_claim
    tasks = [task] if task else list(TASKS)
    out = []
    for name in tasks:
        for expert in FUSION_EXPERTS[name]:
            if not (resolved_arm_dir(name, 'B') / 'metrics.json').is_file():
                continue
            if not (resolved_arm_dir(name, expert) / 'metrics.json').is_file():
                continue
            if (fuse_dir(name, expert) / 'metrics.json').is_file():
                continue
            job_id = f'fuse:{name}:{expert}'
            if not try_claim(job_id):
                continue
            try:
                payload = fuse_one(name, expert)
                finish_claim(job_id, 'PASS')
                out.append(payload)
                print(f"fused {job_id} alpha={payload.get('alpha')}", flush=True)
            except Exception as exc:
                finish_claim(job_id, 'fail', {'error': str(exc)})
                print(f'fail {job_id}: {exc}', flush=True)
    return out

def fuse_watch(*, idle_s: float=30.0) -> dict:
    import time
    from .schedule import catalog_map, execute, finish_claim, try_claim, write_jobs, _done
    write_jobs()
    fused = []
    while True:
        fused.extend(fuse_ready())
        catalog = catalog_map()
        fuse_jobs = [job for job in catalog.values() if job['kind'] == 'fuse']
        if fuse_jobs and all((_done(job) for job in fuse_jobs)):
            summary = catalog.get('summarize')
            if summary is not None and (not _done(summary)) and try_claim(summary['id']):
                try:
                    execute(summary)
                    finish_claim(summary['id'], 'PASS')
                except Exception as exc:
                    finish_claim(summary['id'], 'fail', {'error': str(exc)})
                    raise
            break
        time.sleep(idle_s)
    return {'fused': len(fused), 'status': 'PASS'}

def _read_metrics(path: Path) -> dict:
    return read_json(path) if path.is_file() else {}

def collect_tables() -> dict:
    complete = []
    geometry = []
    recommended = []
    for task in TASKS:
        methods = [('B', resolved_arm_dir(task, 'B') / 'metrics.json', None)]
        for arm in ('G2', 'G3', 'GR', 'G23', 'G2R', 'H2', 'H3', 'P2', 'P3', 'E2', 'ES', 'ER'):
            path = resolved_arm_dir(task, arm) / 'metrics.json'
            if path.is_file():
                methods.append((arm, path, None))
        for expert in FUSION_EXPERTS[task]:
            path = fuse_dir(task, expert) / 'metrics.json'
            if path.is_file():
                methods.append((f'B+{expert}', path, expert))
        best_name, best_score, match = (None, float('-inf'), None)
        for name, path, _expert in methods:
            payload = _read_metrics(path)
            metrics = payload.get('metrics') or {}
            row = {'task': task, 'method': name, 'VAL_primary': (metrics.get('val') or {}).get('primary'), 'TEST_primary': (metrics.get('test') or {}).get('primary'), 'alpha': payload.get('alpha'), 'best_epoch': payload.get('best_epoch'), 'geometry_used': payload.get('geometry_used')}
            complete.append(row)
            if name in {'G2', 'G3', 'GR', 'G23', 'G2R', 'E2', 'ES', 'ER'}:
                geometry.append(row)
            val = row['VAL_primary']
            if val is not None and better(float(val), best_score) and (name in COMPLETE_3D | {'B', 'H2', 'H3', 'P2', 'P3'} | {f'B+{e}' for e in FUSION_EXPERTS[task]}):
                best_score = float(val)
                best_name = name
                match = MATCHING_2D.get(name, 'B' if name == 'B' else None)
        recommended.append({'task': task, 'recommended': best_name, 'matching_2d': match, 'VAL_primary': best_score if np.isfinite(best_score) else None})
    indexed = {(row['task'], row['method']): row for row in complete}
    pairs = []
    for task in TASKS:
        for left, right in PAIR_DIFFS:
            a = indexed.get((task, left))
            b = indexed.get((task, right))
            if not a or not b or a.get('VAL_primary') is None or (b.get('VAL_primary') is None):
                continue
            pairs.append({'task': task, 'pair': f'{left}-{right}', 'VAL_delta': float(a['VAL_primary']) - float(b['VAL_primary']), 'TEST_delta': float(a['TEST_primary']) - float(b['TEST_primary']), 'kind': 'point'})
    recalls = []
    for row in complete:
        if row['task'] != 'subtype4':
            continue
        payload = _read_metrics(resolved_arm_dir(row['task'], row['method']) / 'metrics.json') if not str(row['method']).startswith('B+') else _read_metrics(fuse_dir(row['task'], row['method'][2:]) / 'metrics.json')
        for split in ('fit', 'val'):
            recall = ((payload.get('metrics') or {}).get(split) or {}).get('recall')
            if recall:
                recalls.append({'task': row['task'], 'method': row['method'], 'split': split, 'recall': recall})
    deltas = []
    for task in TASKS:
        b = _read_metrics(resolved_arm_dir(task, 'B') / 'metrics.json')
        if not b:
            continue
        import pandas as pd
        b_pred = pd.read_parquet(resolved_arm_dir(task, 'B') / 'predictions.parquet')
        b_pred = b_pred[b_pred['split'].astype(str).str.lower() == 'test']
        y = b_pred['label'].to_numpy(np.int64)
        pb = np.stack(b_pred['prob'].to_list(), 0)
        for method in ('H3', 'G3', 'GR'):
            path = resolved_arm_dir(task, method) / 'predictions.parquet'
            if not path.is_file():
                continue
            other = pd.read_parquet(path)
            other = other[other['split'].astype(str).str.lower() == 'test']
            merged = b_pred.merge(other, on='patient_id', suffixes=('_b', '_o'))
            if merged.empty:
                continue
            left = np.stack(merged['prob_o'].to_list(), 0)
            right = np.stack(merged['prob_b'].to_list(), 0)
            deltas.append({'task': task, 'pair': f'{method}-B', **paired_bootstrap(merged['label_b'].to_numpy(np.int64), left, right, task)})
    tables_dir().mkdir(parents=True, exist_ok=True)
    payload = {**protocol_meta(), 'complete': complete, 'geometry': geometry, 'recommended': recommended, 'pairs': pairs, 'subtype4_recall': recalls, 'bootstrap': deltas}
    atomic_json(tables_dir() / 'summary.json', payload)
    return payload

def render_results(docs: Path | None, tables: dict) -> Path:
    remote, local = active_docs()
    dest = Path(docs) if docs else remote if remote.parent.is_dir() else local
    dest.parent.mkdir(parents=True, exist_ok=True)
    version = 'conditional_geometry' if is_conditional_geometry() else 'set_encoding'
    lines = [f'# Downstream routes {version} results', '', f"Status: **PASS**. Batch `{tables.get('batch_id', '')}`.", 'Design: [downstream-training-routes.md](../design/downstream-training-routes.md).', f'Execution: [downstream-routes-{version}.md](../execution/downstream-routes-{version}.md).', 'Audit: [all-routes-audit-set_encoding.md](./all-routes-audit-set_encoding.md).', '', 'TEST is reported once; it was not used to select α or checkpoints.', '', '## Main comparisons', '', '1. Does a complete 3D route beat B on the task primary metric?', '2. Does it beat the matching 2D route on the same mechanism?', '3. Does independent 3D (G3 / GR) beat independent 2D (G2)?', '', 'α=0 means fusion did not use geometry and is not a 3D-fusion success.', '', '## Complete prediction table', '', '| task | method | VAL primary | TEST primary | α | notes |', '|---|---|---:|---:|---:|---|']
    for row in tables.get('complete') or []:
        alpha = row.get('alpha')
        note = ''
        if row['method'].startswith('B+') and alpha == 0:
            note = 'α=0, fusion unused'
        lines.append(f"| {row['task']} | {row['method']} | {_fmt(row.get('VAL_primary'))} | {_fmt(row.get('TEST_primary'))} | {_fmt(alpha)} | {note} |")
    lines.extend(['', '## Image-free geometry table', '', '| task | method | VAL primary | TEST primary | notes |', '|---|---|---:|---:|---|'])
    for row in tables.get('geometry') or []:
        lines.append(f"| {row['task']} | {row['method']} | {_fmt(row.get('VAL_primary'))} | {_fmt(row.get('TEST_primary'))} | |")
    lines.extend(['', '## Recommended complete route (VAL only)', '', '| task | recommended | matching 2D | VAL primary |', '|---|---|---|---:|'])
    for row in tables.get('recommended') or []:
        lines.append(f"| {row['task']} | {row.get('recommended')} | {row.get('matching_2d')} | {_fmt(row.get('VAL_primary'))} |")
    if tables.get('pairs'):
        lines.extend(['', '## Paired route differences (point estimate)', '', 'Bootstrap means are listed separately and are not the primary delta.', '', '| task | pair | VAL Δ | TEST Δ |', '|---|---|---:|---:|'])
        for row in tables['pairs']:
            lines.append(f"| {row['task']} | {row['pair']} | {_fmt(row.get('VAL_delta'))} | {_fmt(row.get('TEST_delta'))} |")
    if tables.get('subtype4_recall'):
        lines.extend(['', '## subtype4 per-class recall', '', '| method | split | recall |', '|---|---|---|'])
        for row in tables['subtype4_recall']:
            recall = ', '.join((_fmt(value) for value in row['recall']))
            lines.append(f"| {row['method']} | {row['split']} | {recall} |")
    if tables.get('bootstrap'):
        lines.extend(['', '## TEST paired bootstrap (single seed, descriptive)', '', '| task | pair | mean Δ | 95% CI |', '|---|---|---:|---|'])
        for row in tables['bootstrap']:
            ci = row.get('ci95') or [None, None]
            lines.append(f"| {row['task']} | {row['pair']} | {_fmt(row.get('mean'))} | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dest

def _fmt(value) -> str:
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return '—'
    return f'{number:.4f}'
