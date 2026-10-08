from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
import numpy as np
from sklearn.metrics import f1_score
from .constants import RESULT_ROOT, SEED
from .evaluate import _qwk, score_arvaniti_predictions
from .predict import _route_arm, _softmax

def _logits(name: str, split: str):
    if '+' in name:
        dataset = 'lizard' if name.startswith(('B_crop', 'B_dino')) else 'arvaniti'
        path = RESULT_ROOT / dataset / 'fusion' / name / 'seed42' / f'{split}_logits.npz'
    else:
        dataset, route, arm = _route_arm(name)
        path = RESULT_ROOT / dataset / route / arm / 'seed42' / f'{split}_logits.npz'
    if not path.is_file():
        return None
    return np.load(path)

def _cores(name: str, split: str, cache: dict):
    key = (name, split)
    if key in cache:
        return cache[key]
    packed = _logits(name, split)
    if packed is None:
        cache[key] = None
        return None
    rows = [{'graph_id': str(gid), 'core_id': str(cid), 'probs': _softmax(logit), 'label': int(lab)} for gid, cid, logit, lab in zip(packed['graph_id'], packed['core_id'], packed['logits'], packed['label'])]
    scored = score_arvaniti_predictions(rows, 'val' if split.startswith('val') else 'test', bootstrap=False)
    cache[key] = scored['cores']
    return cache[key]

def _aligned_cores(left, right, reader: str):
    right_pred = {row['core_id']: row['qwk_label'] for row in right}
    y, pl, pr = ([], [], [])
    for row in left:
        gt = (row.get('gt') or {}).get(reader)
        if not gt or row['core_id'] not in right_pred:
            continue
        y.append(gt['qwk_label'])
        pl.append(row['qwk_label'])
        pr.append(right_pred[row['core_id']])
    return (np.asarray(y), np.asarray(pl), np.asarray(pr))

def _interval(values: list[float]) -> dict[str, float] | None:
    arr = np.asarray(values, np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return None
    return {'mean': float(arr.mean()), 'lo': float(np.percentile(arr, 2.5)), 'hi': float(np.percentile(arr, 97.5)), 'n': int(len(arr))}

def _core_delta(left_name: str, right_name: str, cache: dict, draws: int=1000) -> dict[str, Any]:
    rng = np.random.RandomState(SEED)
    out = {}
    for split, readers in (('val_infer', ('pathologist1',)), ('test_infer', ('pathologist1', 'pathologist2'))):
        left = _cores(left_name, split, cache)
        right = _cores(right_name, split, cache)
        if not left or not right:
            continue
        for reader in readers:
            y, pl, pr = _aligned_cores(left, right, reader)
            if len(y) < 2:
                continue
            idx = np.arange(len(y))
            stats = []
            for _ in range(draws):
                take = rng.choice(idx, size=len(idx), replace=True)
                stats.append(_qwk(y[take], pl[take]) - _qwk(y[take], pr[take]))
            out[f'{split}:{reader}'] = _interval(stats)
    return out

def _lizard_rows(name: str, split: str):
    packed = _logits(name, split)
    if packed is None or 'group_id' not in packed.files:
        return None
    pred = packed['logits'].argmax(-1)
    return (np.asarray(packed['label']), np.asarray(pred), np.asarray(packed['group_id']).astype(str), np.asarray(packed['graph_id']).astype(str), np.asarray(packed['nucleus_id']).astype(str) if 'nucleus_id' in packed.files else None)

def _lizard_delta(left_name: str, right_name: str, draws: int=1000) -> dict[str, Any]:
    out = {}
    rng = np.random.RandomState(SEED)
    for split in ('val', 'test'):
        left = _lizard_rows(left_name, split)
        right = _lizard_rows(right_name, split)
        if left is None or right is None or left[4] is None or (right[4] is None):
            continue
        ly, lp, lg, lgraph, lnid = left
        ry, rp, rg, rgraph, rnid = right
        right_index = {f'{g}:{n}': i for i, (g, n) in enumerate(zip(rgraph, rnid))}
        y, pl, pr, groups = ([], [], [], [])
        for i, (g, n) in enumerate(zip(lgraph, lnid)):
            j = right_index.get(f'{g}:{n}')
            if j is None or int(ly[i]) < 0:
                continue
            y.append(int(ly[i]))
            pl.append(int(lp[i]))
            pr.append(int(rp[j]))
            groups.append(str(lg[i]))
        if len(y) < 2:
            continue
        y = np.asarray(y)
        pl = np.asarray(pl)
        pr = np.asarray(pr)
        groups = np.asarray(groups)
        known = np.array([g not in {'', 'None', 'unknown', 'UNKNOWN'} for g in groups])
        buckets: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            if known[index]:
                buckets.setdefault(str(group), []).append(index)
        if len(buckets) < 2:
            out[split] = {'note': 'unknown groups not treated as patient intervals'}
            continue
        unique = np.asarray(list(buckets))
        stats = []
        for _ in range(draws):
            take = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([np.asarray(buckets[str(group)], np.int64) for group in take])
            stats.append(float(f1_score(y[idx], pl[idx], average='macro') - f1_score(y[idx], pr[idx], average='macro')))
        out[split] = _interval(stats)
    return out

def attach_paired(complete: dict[str, Any], geometry: dict[str, Any]) -> None:
    cache: dict = {}
    paired = {'H23-B_paper': _core_delta('H23', 'B_paper', cache), 'H23-H2': _core_delta('H23', 'H2', cache), 'AS-A2': _core_delta('AS', 'A2', cache), 'C3-C2': _core_delta('C3', 'C2', cache), 'G3-G2': _core_delta('G3', 'G2', cache), 'G23-G2': _core_delta('G23', 'G2', cache), 'G2R-G2': _core_delta('G2R', 'G2', cache)}
    for geom in ('G2', 'G3', 'GR', 'G23', 'G2R'):
        paired[f'B_paper+{geom}-B_paper'] = _core_delta(f'B_paper+{geom}', 'B_paper', cache)
    rec = (complete.get('lizard') or {}).get('recommended_B') or 'B_crop'
    paired['LH23-B'] = _lizard_delta('LH23', rec)
    paired['LH23-LH2'] = _lizard_delta('LH23', 'LH2')
    paired['LA3-LA2'] = _lizard_delta('LA3', 'LA2')
    paired['N3-N2'] = _lizard_delta('N3', 'N2')
    paired['N23-N2'] = _lizard_delta('N23', 'N2')
    paired['JS-J2'] = _lizard_delta('JS', 'J2')
    paired['JR-JS'] = _lizard_delta('JR', 'JS')
    for geom in ('N2', 'N3', 'NR', 'N23', 'N2R', 'J2', 'JS', 'JR'):
        paired[f'{rec}+{geom}-{rec}'] = _lizard_delta(f'{rec}+{geom}', rec)
    complete['paired_bootstrap'] = paired
    geometry['paired_bootstrap'] = {key: paired[key] for key in ('G3-G2', 'G23-G2', 'G2R-G2', 'N3-N2', 'N23-N2', 'JS-J2', 'JR-JS') if key in paired}
