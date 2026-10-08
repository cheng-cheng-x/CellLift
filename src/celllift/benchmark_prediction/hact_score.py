from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
import numpy as np
HACT_DATA = Path(_resource_path('artifact_0006'))
OUT = Path(_resource_path('artifact_0007'))
PRIMARY = 'best_val_weighted_f1_score'

def _f1(labels, pred, classes=7):
    labels = np.asarray(labels, np.int64)
    pred = np.asarray(pred, np.int64)
    macros, weights, scores = ([], [], [])
    for cls in range(classes):
        support = float((labels == cls).sum())
        tp = float(((pred == cls) & (labels == cls)).sum())
        fp = float(((pred == cls) & (labels != cls)).sum())
        fn = float(((pred != cls) & (labels == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        scores.append(f1)
        weights.append(support)
        macros.append(f1)
    total = float(sum(weights))
    return (float(np.mean(macros)), float(np.dot(scores, weights) / total) if total else float('nan'))

def coverage() -> dict:
    manifest = json.loads((HACT_DATA / 'manifest.json').read_text(encoding='utf-8')) if (HACT_DATA / 'manifest.json').is_file() else []
    test = [row for row in manifest if row.get('split') == 'test']
    have, missing = ([], [])
    for row in test:
        stem = Path(row['hact_name']).stem
        ok = all(((HACT_DATA / 'graphs' / kind / 'test' / f'{stem}{ext}').is_file() for kind, ext in (('cell_graphs', '.bin'), ('tissue_graphs', '.bin'), ('assignment_matrices', '.h5'))))
        (have if ok else missing).append(row['sample_id'])
    return {'n_test_manifest': len(test), 'n_graphs': len(have), 'n_missing': len(missing), 'complete': len(have) == 570 and (not missing)}

def _metrics_from_dump(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding='utf-8'))
    labels = payload.get('labels') or []
    preds = payload.get('preds') or []
    macro, weighted = _f1(labels, preds)
    return {'path': str(path), 'n': int(payload.get('n') or len(labels)), 'macro_f1': macro, 'weighted_f1': weighted, 'ckpt': payload.get('ckpt')}

def try_infer() -> dict:
    out = HACT_DATA / 'out'
    dumps = {}
    if out.is_dir():
        for path in sorted(out.glob('test_*.json')):
            dumps[path.stem.replace('test_', '')] = _metrics_from_dump(path)
    models = list((HACT_DATA / 'models').rglob('*.pt')) if (HACT_DATA / 'models').exists() else []
    return {'dumps': dumps, 'models': [str(p) for p in models[:20]]}

def main() -> int:
    cov = coverage()
    infer = try_infer()
    primary = (infer.get('dumps') or {}).get(PRIMARY) or {}
    complete = bool(cov['complete'] and primary.get('n') == 570)
    payload = {'status': 'PASS' if complete else 'Incomplete', 'coverage': cov, 'infer': infer, 'primary_ckpt': PRIMARY, 'macro_f1': primary.get('macro_f1'), 'weighted_f1': primary.get('weighted_f1'), 'n_scored': primary.get('n')}
    if not complete:
        if not cov['complete']:
            payload['reason'] = f"test_graphs_{cov['n_graphs']}_lt_570"
        else:
            payload['reason'] = f"scored_{primary.get('n')}_lt_570"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
