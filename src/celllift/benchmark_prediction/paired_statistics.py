from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
import numpy as np
from .crc_score import filter_to_qualified, qualified_patient_ids
from .evaluate import _load_pred
from .fuse import _as_probability
from .io import job_dir
from .paired_ci import paired_delta
from .paths import baseline_result_root, comparison_result_root, result_root, split_root
from .splits import load_units
from .train_geometry import _channels

def _pred_file(dataset, family, arm, unit='official', fold=0, r2=False, r3=False) -> Path:
    job = {'dataset': dataset, 'family': family, 'arm': arm, 'unit': unit, 'seed': 42, 'fold': fold, 'baseline_qualification': r2, 'r3': r3}
    for flag in ((True, False, False), (False, True, False), (False, False, False)):
        path = job_dir({**job, 'r3': flag[0], 'baseline_qualification': flag[1]}) / 'test_predictions.parquet'
        if path.is_file():
            return path
    return job_dir(job) / 'test_predictions.parquet'

def _rows(path: Path) -> list[dict]:
    return _load_pred(path) if path.is_file() else []

def _stack(rows, n_class: int | None=None):
    ids = [str(row['sample_id']) for row in rows]
    labels = np.asarray([int(row['label_id']) for row in rows], np.int64)
    probs = np.stack([np.asarray(_as_probability(row), np.float64) for row in rows], 0)
    pred = probs.argmax(1)
    scores = probs[:, 1] if probs.shape[1] > 1 else probs.reshape(-1)
    return (ids, labels, pred, scores, probs)

def crc_isolation() -> dict:
    units = load_units('tcga_crc_msi')[0]
    fit = set(map(str, units.get('fit') or []))
    val = set(map(str, units.get('val') or []))
    test_q = set(qualified_patient_ids('test'))
    leak = {'val_in_fit': sorted(val & fit), 'test_in_fit': sorted(test_q & fit), 'test_in_val': sorted(test_q & val), 'n_fit': len(fit), 'n_val': len(val), 'n_test_qualified': len(test_q)}
    for family, arm in (('route', 'B2'), ('geometry', 'G2'), ('geometry', 'G3'), ('geometry', 'GR'), ('residual', 'probe')):
        dest = job_dir({'dataset': 'tcga_crc_msi', 'family': family, 'arm': arm, 'unit': 'official', 'seed': 42, 'fold': 0})
        manifest = dest / 'job.json'
        payload = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        pred_ids = set()
        pred = dest / 'predictions.parquet'
        if pred.is_file():
            import pyarrow.parquet as pq
            pred_ids = {str(row.get('sample_id') or row.get('patient_id') or '') for row in pq.read_table(pred).to_pylist()}
        leak[f'{family}/{arm}'] = {'has_job': manifest.is_file(), 'test_used': bool(payload.get('test_used') or payload.get('official_test_touched')), 'fit_split': payload.get('fit_split') or payload.get('wrapped', {}).get('official_fit_n'), 'val_pred_in_test': sorted(pred_ids & test_q), 'n_pred': len(pred_ids)}
    leak['val_disjoint_fit'] = not leak['val_in_fit']
    leak['test_disjoint'] = not leak['test_in_fit'] and (not leak['test_in_val'])
    return leak

def _align(rows_a, rows_b):
    map_a = {str(r['sample_id']): r for r in rows_a}
    map_b = {str(r['sample_id']): r for r in rows_b}
    keys = sorted(set(map_a) & set(map_b))
    return ([map_a[k] for k in keys], [map_b[k] for k in keys], keys)

def crc_p2_ci() -> dict:
    pairs = {}
    loaded = {}
    for arm in ('P2', 'P2+G2', 'P2+G3', 'P2+GR'):
        path = _pred_file('tcga_crc_msi', 'fusion', arm, r2=True)
        rows = filter_to_qualified(_rows(path), 'test')
        loaded[arm] = rows
        pairs[arm] = {'n': len(rows), 'path': str(path)}
    out = {'rows': pairs, 'deltas': {}, 'exploratory_after_seeing_test': True}
    for left, right in (('P2', 'P2+G3'), ('P2+G2', 'P2+G3'), ('P2', 'P2+GR'), ('P2+G2', 'P2+GR')):
        a, b, keys = _align(loaded[left], loaded[right])
        if not keys:
            out['deltas'][f'{right}-{left}'] = {'status': 'MISSING'}
            continue
        _, ya, _, sa, _ = _stack(a)
        _, yb, _, sb, _ = _stack(b)
        out['deltas'][f'{right}-{left}'] = paired_delta(ya, None, None, scores_a=sa, scores_b=sb, cluster_ids=keys, kind='auroc')
    out['primary'] = 'P2+G3-P2+G2'
    return out

def _cluster_ids(dataset: str, rows: list[dict]) -> list[str]:
    if dataset == 'bracs':
        meta = {}
        path = split_root() / 'bracs' / 'test_rows.json'
        if path.is_file():
            for row in json.loads(path.read_text(encoding='utf-8')):
                sid = str(row.get('sample_id') or row.get('roi_id') or '')
                wsi = str(row.get('wsi_id') or row.get('slide_id') or '')
                meta[sid] = wsi if wsi not in {'', 'None'} else sid
        return [meta.get(str(r['sample_id']), str(r['sample_id'])) for r in rows]
    if dataset == 'sicapv2':
        path = split_root() / 'sicapv2' / 'test_rows.json'
        pid = {}
        if path.is_file():
            for row in json.loads(path.read_text(encoding='utf-8')):
                pid[str(row.get('sample_id') or row.get('graph_id'))] = str(row.get('patient_id') or row.get('sample_id'))
        return [pid.get(str(r['sample_id']), str(r['sample_id'])) for r in rows]
    return [str(r['sample_id']) for r in rows]

def geometry_ci() -> dict:
    specs = (('sicapv2', 'geometry', 'G2', 'G3', 'full_train', 0, 'qwk', 4), ('sicapv2', 'geometry', 'G2', 'GR', 'full_train', 0, 'qwk', 4), ('bracs', 'geometry', 'G2', 'G3', 'official', 0, 'macro_f1', 7), ('bracs', 'geometry', 'G2', 'GR', 'official', 0, 'macro_f1', 7), ('tcga_crc_msi', 'geometry', 'G2', 'G3', 'official', 0, 'auroc', 2), ('tcga_crc_msi', 'geometry', 'G2', 'GR', 'official', 0, 'auroc', 2))
    out = {'channels_image_free': {'G2': 'node2d only', 'G3': 'raw node3d only', 'GR': 'residual3d only', 'reads_rgb_or_dino': False}, 'deltas': {}}
    dummy = np.zeros((2, 38 + 12), np.float32)
    dummy[:, :38] = 1
    dummy[:, 38:] = 2
    a, _ = _channels('G2', dummy, np.zeros((2, 12), np.float32))
    out['channels_image_free']['G2_ok'] = bool(float(a.mean()) == 1.0)
    for dataset, family, left, right, unit, fold, kind, classes in specs:
        ra = _rows(_pred_file(dataset, family, left, unit, fold))
        rb = _rows(_pred_file(dataset, family, right, unit, fold))
        if dataset == 'tcga_crc_msi':
            ra, rb = (filter_to_qualified(ra, 'test'), filter_to_qualified(rb, 'test'))
        a, b, keys = _align(ra, rb)
        if not keys:
            out['deltas'][f'{dataset}:{right}-{left}'] = {'status': 'MISSING'}
            continue
        ids, ya, pa, sa, _ = _stack(a)
        _, _, pb, sb, _ = _stack(b)
        clusters = _cluster_ids(dataset, a)
        kwargs = dict(cluster_ids=clusters, kind=kind, classes=classes)
        if kind == 'auroc':
            payload = paired_delta(ya, None, None, scores_a=sa, scores_b=sb, **kwargs)
        else:
            payload = paired_delta(ya, pa, pb, **kwargs)
        payload['cluster_kind'] = 'wsi' if dataset == 'bracs' else 'patient'
        out['deltas'][f'{dataset}:{right}-{left}'] = payload
        del ids
    return out

def run() -> dict:
    report = {'crc_isolation': crc_isolation(), 'crc_p2': crc_p2_ci(), 'geometry': geometry_ci()}
    out = comparison_result_root() / 'analysis' / 'stats.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    report['written'] = str(out)
    return report
if __name__ == '__main__':
    print(json.dumps(run(), indent=2, default=str))
