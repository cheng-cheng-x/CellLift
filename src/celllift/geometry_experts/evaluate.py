from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .dataset import DATASET_SPEC, EXPERT_ARMS, FUSED_ARMS, batch_result_root
from .io_utils import atomic_json, atomic_parquet, read_parquet
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260909
ARMS = ('B',) + EXPERT_ARMS + FUSED_ARMS
PAIRS = (('FR', 'B'), ('F2', 'B'), ('FS', 'B'), ('FR', 'F2'), ('FS', 'F2'), ('FR', 'FS'), ('ER', 'B'), ('E2', 'B'), ('ER', 'E2'), ('ES', 'E2'), ('ER', 'ES'))

def _qwk(labels, probabilities):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(labels, np.asarray(probabilities).argmax(1), weights='quadratic'))

def _macro_f1(labels, probabilities):
    from sklearn.metrics import f1_score
    return float(f1_score(labels, np.asarray(probabilities).argmax(1), average='macro'))

def _mean_fold_auroc(labels, scores, folds):
    from sklearn.metrics import roc_auc_score
    values = []
    for fold in sorted(set(map(int, folds))):
        selected = folds == fold
        if len(np.unique(labels[selected])) < 2:
            return float('nan')
        values.append(float(roc_auc_score(labels[selected], scores[selected])))
    return float(np.mean(values)) if values else float('nan')

def _paired_cluster_bootstrap(labels, groups, scores, metric, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    observed = {arm: metric(labels, value) for arm, value in scores.items()}
    distributions = {arm: np.empty(replicates) for arm in scores}
    members = {group: np.flatnonzero(groups == group) for group in unique}
    for r in range(replicates):
        sampled = rng.choice(len(unique), len(unique), replace=True)
        ids = np.concatenate([members[unique[index]] for index in sampled])
        for arm, value in scores.items():
            distributions[arm][r] = metric(labels[ids], value[ids])
    return (observed, distributions)

def _fold_stratified_bootstrap(labels, folds, scores, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    observed = {arm: _mean_fold_auroc(labels, value, folds) for arm, value in scores.items()}
    distributions = {arm: np.empty(replicates) for arm in scores}
    fold_ids = {fold: np.flatnonzero(folds == fold) for fold in sorted(set(map(int, folds)))}
    for r in range(replicates):
        sampled = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in fold_ids.values()])
        for arm, value in scores.items():
            distributions[arm][r] = _mean_fold_auroc(labels[sampled], value[sampled], folds[sampled])
    return (observed, distributions)

def comparison(observed, distributions, first: str, second: str) -> dict:
    if first not in distributions or second not in distributions:
        return {'comparison': f'{first}-{second}', 'status': 'MISSING'}
    delta = np.asarray(distributions[first]) - np.asarray(distributions[second])
    finite = delta[np.isfinite(delta)]
    if not len(finite):
        return {'comparison': f'{first}-{second}', 'status': 'NO_FINITE_REPLICATES'}
    return {'comparison': f'{first}-{second}', 'delta': float(observed[first] - observed[second]), 'ci_low': float(np.quantile(finite, 0.025)), 'ci_high': float(np.quantile(finite, 0.975)), 'p_one_sided': float((1 + np.count_nonzero(finite <= 0)) / (len(finite) + 1)), 'finite_replicates': int(len(finite))}

def load_fold_predictions(result_root: Path, dataset: str, arm: str, seed: int) -> list[dict]:
    rows = []
    for fold in range(int(DATASET_SPEC[dataset]['folds'])):
        path = result_root / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{fold:02d}' / 'predictions.parquet'
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(read_parquet(path))
    return rows

def evaluate_dataset(cfg: Mapping[str, Any], dataset: str, seed: int, arms: Sequence[str]=ARMS) -> dict[str, Any]:
    result_root = batch_result_root(cfg)
    completed, missing, per_arm = ([], [], {})
    for arm in arms:
        if arm == 'B':
            continue
        try:
            per_arm[arm] = load_fold_predictions(result_root, dataset, arm, seed)
            completed.append(arm)
        except FileNotFoundError:
            missing.append(arm)
    if 'FR' not in completed and 'ER' not in completed:
        raise RuntimeError(f'no completed expert/fused arm for {dataset}: {completed}')
    reference = 'FR' if 'FR' in completed else completed[0]
    lookup = defaultdict(dict)
    for arm, rows in per_arm.items():
        for row in rows:
            lookup[arm][str(row['sample_id'])] = row
    keys = sorted(lookup[reference])
    for arm in completed:
        if set(lookup[arm]) != set(keys):
            raise RuntimeError(f'sample set mismatch for {arm}')
    labels = np.asarray([int(lookup[reference][key]['label_id']) for key in keys])
    folds = np.asarray([int(lookup[reference][key]['fold']) for key in keys])
    groups = np.asarray([str(lookup[reference][key]['group_id']) for key in keys], object)
    baseline = np.stack([np.asarray(lookup[reference][key]['baseline_probability'], float) for key in keys])
    scores = {arm: np.stack([np.asarray(lookup[arm][key]['final_probability'], float) for key in keys]) for arm in completed}
    scores['B'] = baseline
    completed = ['B', *completed]
    if dataset == 'sicapv2':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, scores, _qwk)
    elif dataset == 'bracs':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, scores, _macro_f1)
    else:
        scalar = {arm: value[:, 0] for arm, value in scores.items()}
        observed, distributions = _fold_stratified_bootstrap(labels, folds, scalar)
    payload = {'status': 'PASS', 'dataset': dataset, 'seed': seed, 'arms_completed': completed, 'arms_missing': missing, 'samples': len(keys), 'metric': DATASET_SPEC[dataset]['metric'], 'observed': {arm: float(value) for arm, value in observed.items()}, 'comparisons': [comparison(observed, distributions, first, second) for first, second in PAIRS], 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'development_only': True, 'official_test_touched': False, 'primary': ('FR-B', 'FR-F2')}
    destination = result_root / dataset / 'metrics'
    atomic_json(destination / f'summary_seed{seed}.json', payload)
    flat = []
    for arm in completed:
        for position, key in enumerate(keys):
            flat.append({'sample_id': key, 'arm': arm, 'fold': int(folds[position]), 'group_id': str(groups[position]), 'label_id': int(labels[position]), 'probability': np.asarray(scores[arm][position], float).tolist()})
    atomic_parquet(destination / f'oof_seed{seed}.parquet', flat)
    return payload
