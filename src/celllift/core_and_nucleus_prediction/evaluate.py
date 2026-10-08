from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import sys
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
_MODEL_INPUT = Path(__file__).resolve().parents[1] / 'model_inputs'
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
from celllift.model_inputs.arvaniti_lizard.io_utils import atomic_json, read_parquet
from celllift.model_inputs.arvaniti_lizard.official import core_label_from_mask, pool_official_from_alphas, pool_window_alphas
from celllift.model_inputs.arvaniti_lizard.audit import _decode_mask
from .constants import ARVANITI_DATA, LIZARD_DATA, RESULT_ROOT, SEED

def _qwk(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return float('nan')
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
_GT_CACHE: dict[str, dict[str, dict[str, int]]] | None = None
_WINDOW_META_CACHE: dict[str, dict[str, Any]] | None = None
_TISSUE_CACHE: dict[str, np.ndarray] = {}
_POOL_ALPHA_CACHE: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

def _core_gt_map() -> dict[str, dict[str, dict[str, int]]]:
    global _GT_CACHE
    if _GT_CACHE is not None:
        return _GT_CACHE
    cores = read_parquet(ARVANITI_DATA / '04_labels_splits' / 'cores.parquet')
    masks = read_parquet(ARVANITI_DATA / '00_manifest' / 'corrected_set_encoding' / 'masks.parquet')
    by = {(row['core_id'], row['reader']): row for row in masks}
    out = {}
    for core in cores:
        readers = ['pathologist1', 'pathologist2'] if core['role'] == 'TEST' else ['pathologist1'] if (core['core_id'], 'pathologist1') in by else ['train']
        out[core['core_id']] = {'role': core['role']}
        for reader in readers:
            row = by.get((core['core_id'], reader))
            if row is None:
                continue
            out[core['core_id']][reader] = core_label_from_mask(_decode_mask(Path(row['path'])))
        if 'pathologist1' not in out[core['core_id']] and 'train' in out[core['core_id']]:
            out[core['core_id']]['pathologist1'] = out[core['core_id']]['train']
    _GT_CACHE = out
    return out

def _window_meta_map() -> dict[str, dict[str, Any]]:
    global _WINDOW_META_CACHE
    if _WINDOW_META_CACHE is not None:
        return _WINDOW_META_CACHE
    windows: dict[str, dict[str, Any]] = {}
    for name in ('inference_window_manifest_conditional_geometry.parquet', 'window_manifest.parquet'):
        path = ARVANITI_DATA / '00_manifest' / name
        if path.is_file():
            for row in read_parquet(path):
                windows[row['graph_id']] = row
    _WINDOW_META_CACHE = windows
    return windows

def _tissue_1550(core_id: str) -> np.ndarray:
    cached = _TISSUE_CACHE.get(core_id)
    if cached is not None:
        return cached
    tissue_path = ARVANITI_DATA / '02_nucleus_masks' / 'tissue_author' / f'{core_id}_1550.npy'
    tissue = np.load(tissue_path) if tissue_path.is_file() else np.zeros((1550, 1550), np.uint8)
    _TISSUE_CACHE[core_id] = tissue
    return tissue

def _alphas_for_core(core_id: str, meta: list[dict[str, Any]]) -> np.ndarray:
    key = (core_id, tuple((str(row['graph_id']) for row in meta)))
    cached = _POOL_ALPHA_CACHE.get(key)
    if cached is not None:
        return cached
    alphas = pool_window_alphas(meta, _tissue_1550(core_id))
    _POOL_ALPHA_CACHE[key] = alphas
    return alphas

def score_arvaniti_predictions(pred_rows: list[dict[str, Any]], split: str, *, bootstrap: bool=True) -> dict[str, Any]:
    windows = _window_meta_map()
    by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        by_core[row['core_id']].append(row)
    gt = _core_gt_map()
    cores = []
    for core_id, items in by_core.items():
        kept = [item for item in items if item['graph_id'] in windows]
        meta = [windows[item['graph_id']] for item in kept]
        if not meta:
            continue
        probs = np.stack([np.asarray(item['probs'], np.float32) for item in kept])
        pooled = pool_official_from_alphas(probs, _alphas_for_core(core_id, meta))
        cores.append({'core_id': core_id, **pooled, 'n_windows': len(items), 'gt': gt.get(core_id, {})})

    def kappa(reader: str) -> float:
        y_true, y_pred = ([], [])
        for row in cores:
            if reader not in row.get('gt', {}):
                continue
            y_true.append(row['gt'][reader]['qwk_label'])
            y_pred.append(row['qwk_label'])
        return _qwk(y_true, y_pred)
    local_y, local_p = ([], [])
    for row in pred_rows:
        if int(row.get('label', -1)) >= 0:
            local_y.append(int(row['label']))
            local_p.append(int(np.argmax(row['probs'])))
    qwk_p1 = kappa('pathologist1')
    if qwk_p1 != qwk_p1:
        qwk_p1 = kappa('train')
    payload = {'split': split, 'n_cores': len(cores), 'n_windows': len(pred_rows), 'core_qwk_p1': qwk_p1, 'core_qwk_p2': kappa('pathologist2') if split == 'test' else None, 'local_macro_f1': float(f1_score(local_y, local_p, average='macro')) if local_y else None, 'local_qwk': _qwk(local_y, local_p) if local_y else None, 'local_recall': recall_score(local_y, local_p, average=None, zero_division=0).tolist() if local_y else None}
    if payload['core_qwk_p1'] is not None and payload['core_qwk_p2'] is not None:
        payload['core_qwk_mean'] = 0.5 * (payload['core_qwk_p1'] + payload['core_qwk_p2'])
    if bootstrap:
        payload['bootstrap'] = _bootstrap_cores(cores, split)
    payload['cores'] = cores
    return payload

def _bootstrap_cores(cores: list[dict[str, Any]], split: str, draws: int=1000) -> dict[str, Any]:
    if len(cores) < 2:
        return {}
    rng = np.random.RandomState(SEED)
    readers = ['pathologist1'] if split != 'test' else ['pathologist1', 'pathologist2']
    out = {}
    for reader in readers:
        pairs = [(row['gt'][reader]['qwk_label'], row['qwk_label']) for row in cores if reader in row.get('gt', {})]
        if len(pairs) < 2:
            continue
        stats = []
        idx = np.arange(len(pairs))
        for _ in range(draws):
            take = rng.choice(idx, size=len(idx), replace=True)
            y = [pairs[i][0] for i in take]
            p = [pairs[i][1] for i in take]
            stats.append(_qwk(y, p))
        stats = np.asarray(stats, np.float64)
        out[reader] = {'mean': float(np.nanmean(stats)), 'lo': float(np.nanpercentile(stats, 2.5)), 'hi': float(np.nanpercentile(stats, 97.5))}
    return out

def score_lizard_predictions(pred_rows: list[dict[str, Any]], groups: list[str] | None=None, *, bootstrap: bool=True) -> dict[str, Any]:
    kept_y, kept_p, kept_g = ([], [], [])
    for index, row in enumerate(pred_rows):
        label = int(row.get('label', -1))
        if label < 0:
            continue
        kept_y.append(label)
        kept_p.append(int(row['pred']))
        if 'group_id' in row and row.get('group_id') is not None:
            kept_g.append(row.get('group_id'))
        elif groups is not None and index < len(groups):
            kept_g.append(groups[index])
    y, p = (kept_y, kept_p)
    payload = {'n': len(y), 'macro_f1': float(f1_score(y, p, average='macro')) if y else None, 'weighted_f1': float(f1_score(y, p, average='weighted')) if y else None, 'accuracy': float(np.mean(np.asarray(y) == np.asarray(p))) if y else None, 'precision': precision_score(y, p, average=None, zero_division=0).tolist() if y else None, 'recall': recall_score(y, p, average=None, zero_division=0).tolist() if y else None}
    if bootstrap and kept_g and (len(kept_g) == len(y)):
        payload['bootstrap'] = _bootstrap_groups(y, p, kept_g)
    return payload

def _bootstrap_groups(y, p, groups, draws: int=1000) -> dict[str, Any]:
    y = np.asarray(y)
    p = np.asarray(p)
    groups = np.asarray(groups)
    known = np.array([str(g) not in {'', 'None', 'unknown', 'UNKNOWN'} for g in groups])
    if known.sum() < 2:
        return {'note': 'unknown groups not treated as patient intervals'}
    buckets: dict[str, np.ndarray] = {}
    for index, group in enumerate(groups):
        if not known[index]:
            continue
        buckets.setdefault(str(group), []).append(index)
    buckets = {key: np.asarray(value, np.int64) for key, value in buckets.items()}
    unique = np.asarray(list(buckets))
    rng = np.random.RandomState(SEED)
    stats = []
    for _ in range(draws):
        take = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([buckets[str(group)] for group in take])
        if len(idx) < 2:
            continue
        stats.append(float(f1_score(y[idx], p[idx], average='macro')))
    if not stats:
        return {}
    arr = np.asarray(stats)
    return {'mean': float(arr.mean()), 'lo': float(np.percentile(arr, 2.5)), 'hi': float(np.percentile(arr, 97.5)), 'n_groups': int(len(unique))}

def _read(path: Path):
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    return None

def _finite_metric(payload: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not payload:
        return None
    value = payload.get(key)
    if value is None or value != value:
        return None
    return payload

def _arm_bundle(dataset: str, route: str, arm: str) -> dict[str, Any] | None:
    root = RESULT_ROOT / dataset / route / arm / 'seed42'
    if not (root / 'model.pt').is_file() and (not (root / 'fusion.json').is_file()) and (not (root / 'metrics.json').is_file()):
        return None
    official = _read(root / 'official_metrics.json')
    if dataset == 'arvaniti':
        val = _finite_metric((official or {}).get('val_infer'), 'core_qwk_p1') or _finite_metric(_read(root / 'val_infer_metrics.json'), 'core_qwk_p1') or _finite_metric(_read(root / 'val_metrics.json'), 'core_qwk_p1') or _read(root / 'val_infer_metrics.json') or _read(root / 'val_metrics.json')
        test = _finite_metric((official or {}).get('test_infer'), 'core_qwk_p1') or _finite_metric(_read(root / 'test_infer_metrics.json'), 'core_qwk_p1') or _finite_metric(_read(root / 'test_metrics.json'), 'core_qwk_p1') or _read(root / 'test_infer_metrics.json') or _read(root / 'test_metrics.json')
    else:
        val = (official or {}).get('val') or _read(root / 'val_metrics.json')
        test = (official or {}).get('test') or _read(root / 'test_metrics.json')
    bundle = {'metrics': _read(root / 'metrics.json'), 'official': official, 'val': val, 'test': test, 'fusion': _read(root / 'fusion.json')}
    return bundle

def _delta(a, b, key: str):
    if not a or not b:
        return None
    left = (a.get('test') or a.get('official') or {}).get('test') or a.get('test') or {}
    right = (b.get('test') or b.get('official') or {}).get('test') or b.get('test') or {}
    if key not in left or key not in right or left[key] is None or (right[key] is None):
        va = (a.get('metrics') or {}).get('best')
        vb = (b.get('metrics') or {}).get('best')
        if va is None or vb is None:
            return None
        return float(va) - float(vb)
    return float(left[key]) - float(right[key])

def recommended_lizard_b() -> str:
    crop = _arm_bundle('lizard', 'baseline', 'B_crop')
    dino = _arm_bundle('lizard', 'baseline', 'B_dino')
    crop_s = ((crop or {}).get('val') or {}).get('macro_f1') or ((crop or {}).get('metrics') or {}).get('best_macro_f1') or -1
    dino_s = ((dino or {}).get('val') or {}).get('macro_f1') or ((dino or {}).get('metrics') or {}).get('best_macro_f1') or -1
    return 'B_crop' if float(crop_s) >= float(dino_s) else 'B_dino'

def write_tables() -> dict[str, Any]:
    rec_b = recommended_lizard_b()
    complete = {'arvaniti': {'B_paper': _arm_bundle('arvaniti', 'baseline', 'B_paper'), 'H2': _arm_bundle('arvaniti', 'fusion', 'H2'), 'H23': _arm_bundle('arvaniti', 'fusion', 'H23'), 'A2': _arm_bundle('arvaniti', 'interact', 'A2'), 'AS': _arm_bundle('arvaniti', 'interact', 'AS'), 'C2': _arm_bundle('arvaniti', 'spatial', 'C2'), 'C3': _arm_bundle('arvaniti', 'spatial', 'C3'), 'B+G2': _arm_bundle('arvaniti', 'fusion', 'B_paper+G2'), 'B+G3': _arm_bundle('arvaniti', 'fusion', 'B_paper+G3'), 'B+GR': _arm_bundle('arvaniti', 'fusion', 'B_paper+GR'), 'B+G23': _arm_bundle('arvaniti', 'fusion', 'B_paper+G23'), 'B+G2R': _arm_bundle('arvaniti', 'fusion', 'B_paper+G2R'), 'official_fcn_released': _read(RESULT_ROOT / 'arvaniti' / 'official_fcn' / 'verify.json'), 'official_fcn_seed42': _read(RESULT_ROOT / 'arvaniti' / 'official_fcn_seed42' / 'verify.json')}, 'lizard': {'recommended_B': rec_b, 'B_crop': _arm_bundle('lizard', 'baseline', 'B_crop'), 'B_dino': _arm_bundle('lizard', 'baseline', 'B_dino'), 'LH2': _arm_bundle('lizard', 'fusion', 'LH2'), 'LH23': _arm_bundle('lizard', 'fusion', 'LH23'), 'LA2': _arm_bundle('lizard', 'interact', 'LA2'), 'LA3': _arm_bundle('lizard', 'interact', 'LA3')}, 'primary_metric': {'arvaniti': 'core_qwk_p1 on VAL; TEST p1/p2/mean', 'lizard': 'six-class macro-F1'}}
    for left, right, key in (('H23', 'B_paper', 'F3-B'), ('H23', 'H2', 'F3-F2'), ('AS', 'A2', 'F3-F2'), ('C3', 'C2', 'F3-F2')):
        complete['arvaniti'][key if left == 'H23' and right == 'B_paper' else f'{left}-{right}'] = _delta(complete['arvaniti'].get(left), complete['arvaniti'].get(right), 'core_qwk_p1')
    complete['arvaniti']['H23-B'] = _delta(complete['arvaniti']['H23'], complete['arvaniti']['B_paper'], 'core_qwk_mean')
    complete['arvaniti']['H23-H2'] = _delta(complete['arvaniti']['H23'], complete['arvaniti']['H2'], 'core_qwk_p1')
    complete['lizard']['LH23-B'] = _delta(complete['lizard']['LH23'], complete['lizard'][rec_b], 'macro_f1')
    complete['lizard']['LH23-LH2'] = _delta(complete['lizard']['LH23'], complete['lizard']['LH2'], 'macro_f1')
    complete['lizard']['LA3-LA2'] = _delta(complete['lizard']['LA3'], complete['lizard']['LA2'], 'macro_f1')
    for geom in ('N2', 'N3', 'NR', 'N23', 'N2R', 'J2', 'JS', 'JR'):
        complete['lizard'][f'{rec_b}+{geom}'] = _arm_bundle('lizard', 'fusion', f'{rec_b}+{geom}')
    geometry = {'arvaniti': {arm: _arm_bundle('arvaniti', 'geom', arm) for arm in ('G2', 'G3', 'GR', 'G23', 'G2R')}, 'lizard': {**{arm: _arm_bundle('lizard', 'geom', arm) for arm in ('N2', 'N3', 'NR', 'N23', 'N2R')}, **{arm: _arm_bundle('lizard', 'relation', arm) for arm in ('J2', 'JS', 'JR')}}}
    geometry['arvaniti']['G3-G2'] = _delta(geometry['arvaniti']['G3'], geometry['arvaniti']['G2'], 'core_qwk_p1')
    geometry['arvaniti']['G23-G2'] = _delta(geometry['arvaniti']['G23'], geometry['arvaniti']['G2'], 'core_qwk_p1')
    geometry['arvaniti']['G2R-G2'] = _delta(geometry['arvaniti']['G2R'], geometry['arvaniti']['G2'], 'core_qwk_p1')
    geometry['lizard']['N3-N2'] = _delta(geometry['lizard']['N3'], geometry['lizard']['N2'], 'macro_f1')
    geometry['lizard']['N23-N2'] = _delta(geometry['lizard']['N23'], geometry['lizard']['N2'], 'macro_f1')
    geometry['lizard']['JS-J2'] = _delta(geometry['lizard']['JS'], geometry['lizard']['J2'], 'macro_f1')
    geometry['lizard']['JR-JS'] = _delta(geometry['lizard']['JR'], geometry['lizard']['JS'], 'macro_f1')
    from .paired import attach_paired
    attach_paired(complete, geometry)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULT_ROOT / 'complete_prediction_table.json', complete)
    atomic_json(RESULT_ROOT / 'geometry_independent_table.json', geometry)
    return {'complete': complete, 'geometry': geometry, 'recommended_lizard_B': rec_b}

def _arm_metrics(dataset: str, route: str, arm: str) -> dict[str, Any] | None:
    return _arm_bundle(dataset, route, arm)
