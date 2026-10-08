from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from scipy.optimize import minimize
from .constants import RESULT_ROOT
from .evaluate import score_arvaniti_predictions, score_lizard_predictions

def _center(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    return value - value.mean(axis=-1, keepdims=True)

def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    value = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(value)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)

def fit_alpha(base: np.ndarray, geom: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, np.int64)
    valid = labels >= 0
    if valid.sum() < 2:
        return 0.0
    base_c = _center(base[valid])
    geom_c = _center(geom[valid])
    y = labels[valid]

    def nll(alpha):
        fused = base_c + float(alpha[0]) * geom_c
        fused = fused - fused.max(axis=1, keepdims=True)
        exp = np.exp(fused)
        prob = exp / exp.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-07, 1)).mean())
    result = minimize(nll, x0=np.array([0.25]), bounds=[(0.0, 4.0)], method='L-BFGS-B')
    return float(result.x[0])

def _keys(payload) -> np.ndarray:
    graph = np.asarray(payload['graph_id']).astype(str)
    if 'nucleus_id' in payload.files or 'nucleus_id' in payload:
        nucleus = np.asarray(payload['nucleus_id']).astype(str)
        return np.char.add(np.char.add(graph, ':'), nucleus)
    return graph

def _align(base, geom) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    base_key = _keys(base)
    geom_key = _keys(geom)
    index = {key: i for i, key in enumerate(geom_key)}
    keep = [i for i, key in enumerate(base_key) if key in index]
    if not keep:
        return None
    gi = [index[base_key[i]] for i in keep]
    graphs = np.asarray(base['graph_id'])[keep]
    if 'core_id' in base.files or 'core_id' in base:
        cores = np.asarray(base['core_id'])[keep]
    else:
        cores = np.asarray([str(g).rsplit('_patch_', 1)[0] for g in graphs])
    if 'group_id' in base.files or 'group_id' in base:
        groups = np.asarray(base['group_id'])[keep]
    else:
        groups = cores
    return (base['logits'][keep], geom['logits'][gi], base['label'][keep], graphs, cores, groups)

def _copy_base_official(dataset: str, base_name: str) -> dict[str, Any]:
    matches = list((RESULT_ROOT / dataset).glob(f'**/{base_name}/seed42/official_metrics.json'))
    if not matches:
        return {}
    try:
        official = json.loads(matches[0].read_text(encoding='utf-8'))
    except Exception:
        return {}
    root = matches[0].parent
    if 'val' not in official and (root / 'val_metrics.json').is_file():
        official['val'] = json.loads((root / 'val_metrics.json').read_text(encoding='utf-8'))
    return official

def _find_logits(dataset: str, name: str, split: str) -> Path | None:
    matches = list((RESULT_ROOT / dataset).glob(f'**/{name}/seed42/{split}_logits.npz'))
    return matches[0] if matches else None

def fuse_pair(dataset: str, base_name: str, geom_name: str) -> dict[str, Any]:
    val_base = _find_logits(dataset, base_name, 'val')
    val_geom = _find_logits(dataset, geom_name, 'val')
    if val_base is None or val_geom is None:
        return {'status': 'WAIT', 'base': base_name, 'geom': geom_name}
    aligned = _align(np.load(val_base), np.load(val_geom))
    if aligned is None:
        return {'status': 'WAIT', 'base': base_name, 'geom': geom_name, 'reason': 'no_id_overlap'}
    base, geom, labels, graphs, cores, groups = aligned
    alpha = fit_alpha(base, geom, labels)
    fused_val = _center(base) + alpha * _center(geom)
    reuse_base = dataset == 'arvaniti' and abs(alpha) < 1e-12
    out = RESULT_ROOT / dataset / 'fusion' / f'{base_name}+{geom_name}' / 'seed42'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'val_logits.npz', graph_id=np.asarray(graphs), core_id=np.asarray(cores), logits=fused_val.astype(np.float32), label=np.asarray(labels, np.int64), group_id=np.asarray(groups))
    payload: dict[str, Any] = {'alpha': alpha, 'base': base_name, 'geom': geom_name, 'n_val': int(len(labels))}
    if dataset == 'lizard':
        payload['val'] = score_lizard_predictions([{'label': int(y), 'pred': int(p), 'group_id': g} for y, p, g in zip(labels, fused_val.argmax(-1), groups)])
    else:
        infer_base = _find_logits(dataset, base_name, 'val_infer')
        infer_geom = _find_logits(dataset, geom_name, 'val_infer')
        aligned_v = None
        fused_infer = vgraphs = vc = vy = None
        if infer_base is not None and infer_geom is not None:
            aligned_v = _align(np.load(infer_base), np.load(infer_geom))
            if aligned_v is not None:
                vb, vg, vy, vgraphs, vc, vgids = aligned_v
                fused_infer = _center(vb) + alpha * _center(vg)
                np.savez_compressed(out / 'val_infer_logits.npz', graph_id=np.asarray(vgraphs), core_id=np.asarray(vc), logits=fused_infer.astype(np.float32), label=np.asarray(vy, np.int64), group_id=np.asarray(vgids))
        if reuse_base:
            copied = _copy_base_official(dataset, base_name)
            if copied.get('val_infer'):
                payload['val'] = copied['val_infer']
            if copied.get('val'):
                payload['val_supervised'] = copied['val']
        if 'val' not in payload:
            rows = [{'graph_id': str(g), 'core_id': str(c), 'probs': _softmax(logit), 'label': int(y)} for g, c, logit, y in zip(graphs, cores, fused_val, labels)]
            payload['val_supervised'] = {k: v for k, v in score_arvaniti_predictions(rows, 'val').items() if k != 'cores'}
            if infer_base is not None and infer_geom is not None and (aligned_v is not None):
                rows = [{'graph_id': str(g), 'core_id': str(c), 'probs': _softmax(logit), 'label': int(y)} for g, c, logit, y in zip(vgraphs, vc, fused_infer, vy)]
                payload['val'] = {k: v for k, v in score_arvaniti_predictions(rows, 'val').items() if k != 'cores'}
        if 'val' not in payload:
            payload['val'] = payload.get('val_supervised')
    test_base = _find_logits(dataset, base_name, 'test_infer') or _find_logits(dataset, base_name, 'test')
    test_geom = _find_logits(dataset, geom_name, 'test_infer') or _find_logits(dataset, geom_name, 'test')
    if test_base is not None and test_geom is not None:
        aligned_t = _align(np.load(test_base), np.load(test_geom))
        if aligned_t is not None:
            tb, tg, ty, tgraphs, tc, tgids = aligned_t
            fused_test = _center(tb) + alpha * _center(tg)
            np.savez_compressed(out / 'test_logits.npz', graph_id=np.asarray(tgraphs), core_id=np.asarray(tc), logits=fused_test.astype(np.float32), label=np.asarray(ty, np.int64), group_id=np.asarray(tgids))
            if dataset == 'arvaniti':
                if reuse_base:
                    copied = _copy_base_official(dataset, base_name)
                    if copied.get('test_infer'):
                        payload['test'] = copied['test_infer']
                if 'test' not in payload:
                    rows = [{'graph_id': str(g), 'core_id': str(c), 'probs': _softmax(logit), 'label': int(y)} for g, c, logit, y in zip(tgraphs, tc, fused_test, ty)]
                    payload['test'] = {k: v for k, v in score_arvaniti_predictions(rows, 'test').items() if k != 'cores'}
            else:
                payload['test'] = score_lizard_predictions([{'label': int(y), 'pred': int(p), 'group_id': g} for y, p, g in zip(ty, fused_test.argmax(-1), tgids)])
    if dataset == 'arvaniti':
        official = {}
        if 'val' in payload:
            official['val_infer'] = payload['val']
            (out / 'val_infer_metrics.json').write_text(json.dumps(payload['val'], indent=2), encoding='utf-8')
        if 'test' in payload:
            official['test_infer'] = payload['test']
            (out / 'test_infer_metrics.json').write_text(json.dumps(payload['test'], indent=2), encoding='utf-8')
        if official:
            (out / 'official_metrics.json').write_text(json.dumps(official, indent=2), encoding='utf-8')
    else:
        official = {k: payload[k] for k in ('val', 'test') if k in payload}
        if official:
            (out / 'official_metrics.json').write_text(json.dumps(official, indent=2), encoding='utf-8')
        if 'val' in payload:
            (out / 'val_metrics.json').write_text(json.dumps(payload['val'], indent=2), encoding='utf-8')
        if 'test' in payload:
            (out / 'test_metrics.json').write_text(json.dumps(payload['test'], indent=2), encoding='utf-8')
    (out / 'fusion.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload
