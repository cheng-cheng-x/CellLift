from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .dataset import DATASET_SPEC
from .io_utils import atomic_json, atomic_parquet, read_parquet
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260909
ARMS = ('B', 'M', 'XY', 'G', 'G-Zperm', 'Recal')
JOB_ARMS = ARMS + ('G-beta', 'G-Het', 'G-Local', 'XY-Local')
PAIRS = (('M', 'B'), ('XY', 'B'), ('G', 'B'), ('G', 'M'), ('XY', 'M'), ('G', 'XY'), ('G-Zperm', 'G'), ('Recal', 'B'), ('G-beta', 'B'), ('G-beta', 'G'), ('G-beta', 'Recal'), ('G-Het', 'B'), ('G-Het', 'G'), ('G-Local', 'B'), ('G-Local', 'G'), ('XY-Local', 'B'), ('XY-Local', 'XY'), ('G-Local', 'XY-Local'))

def _qwk(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(labels, np.asarray(probabilities).argmax(1), weights='quadratic'))

def _macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(labels, np.asarray(probabilities).argmax(1), average='macro'))

def _mean_fold_auroc(labels: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float:
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

def fold_prediction_path(result_root: Path, dataset: str, arm: str, seed: int, fold: int) -> Path:
    return result_root / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{fold:02d}' / 'predictions.parquet'

def load_fold_predictions(result_root: Path, dataset: str, arm: str, seed: int) -> list[dict]:
    rows: list[dict] = []
    for fold in range(int(DATASET_SPEC[dataset]['folds'])):
        path = fold_prediction_path(result_root, dataset, arm, seed, fold)
        if not path.is_file():
            raise FileNotFoundError(f'missing fold predictions: {path}')
        rows.extend(read_parquet(path))
    return rows

def prediction_roots(cfg: Mapping[str, Any]) -> list[Path]:
    result_root = Path(cfg['paths']['result_root'])
    roots = [result_root]
    source = cfg['paths'].get('source_root')
    if source:
        source_root = Path(source)
        if source_root != result_root:
            roots.append(source_root)
    return roots

def load_arm_predictions(cfg: Mapping[str, Any], dataset: str, arm: str, seed: int) -> list[dict]:
    last_error: FileNotFoundError | None = None
    for root in prediction_roots(cfg):
        try:
            return load_fold_predictions(root, dataset, arm, seed)
        except FileNotFoundError as error:
            last_error = error
    raise last_error if last_error is not None else FileNotFoundError(arm)

def evaluate_dataset(cfg: Mapping[str, Any], dataset: str, seed: int, arms: Sequence[str]=ARMS) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    completed, missing = ([], [])
    per_arm: dict[str, list[dict]] = {}
    for arm in arms:
        try:
            per_arm[arm] = load_arm_predictions(cfg, dataset, arm, seed)
            completed.append(arm)
        except FileNotFoundError:
            missing.append(arm)
    if 'G' not in completed:
        raise RuntimeError(f'primary arm G is incomplete for {dataset}: completed={completed}')
    lookup: dict[str, dict[str, dict]] = defaultdict(dict)
    for arm, rows in per_arm.items():
        for row in rows:
            lookup[arm][str(row['sample_id'])] = row
    keys = sorted(lookup['G'])
    for arm in completed:
        if set(lookup[arm]) != set(keys):
            raise RuntimeError(f'sample set mismatch for arm {arm} on {dataset}')
    labels = np.asarray([int(lookup['G'][key]['label_id']) for key in keys])
    folds = np.asarray([int(lookup['G'][key]['fold']) for key in keys])
    groups = np.asarray([str(lookup['G'][key]['group_id']) for key in keys], object)
    baseline = np.stack([np.asarray(lookup['G'][key]['baseline_probability'], float) for key in keys])
    scores = {arm: np.stack([np.asarray(lookup[arm][key]['final_probability'], float) for key in keys]) for arm in completed}
    scores['B'] = baseline
    if 'B' not in completed:
        completed = ['B', *completed]
    missing = [arm for arm in missing if arm != 'B']
    if dataset == 'sicapv2':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, scores, _qwk)
    elif dataset == 'bracs':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, scores, _macro_f1)
    else:
        scalar = {arm: value[:, 0] for arm, value in scores.items()}
        observed, distributions = _fold_stratified_bootstrap(labels, folds, scalar)
    comparisons = [comparison(observed, distributions, first, second) for first, second in PAIRS]
    payload = {'status': 'PASS', 'dataset': dataset, 'seed': seed, 'arms_completed': completed, 'arms_missing': missing, 'samples': len(keys), 'metric': DATASET_SPEC[dataset]['metric'], 'observed': {arm: float(value) for arm, value in observed.items()}, 'comparisons': comparisons, 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'development_only': True, 'official_test_touched': False, 'nested_independence': "development screening only: other folds' frozen baseline checkpoints may have seen the evaluated fold"}
    destination = result_root / dataset / 'metrics'
    atomic_json(destination / f'summary_seed{seed}.json', payload)
    flat = []
    for arm in completed:
        for position, key in enumerate(keys):
            flat.append({'sample_id': key, 'arm': arm, 'fold': int(folds[position]), 'group_id': str(groups[position]), 'label_id': int(labels[position]), 'probability': np.asarray(scores[arm][position], float).tolist()})
    atomic_parquet(destination / f'oof_seed{seed}.parquet', flat)
    grid = evaluate_beta_grid(cfg, dataset, seed, keys, labels, folds, groups, scores.get('B'), scores.get('G'))
    if grid is not None:
        payload['beta_grid'] = grid
        atomic_json(destination / f'beta_grid_seed{seed}.json', grid)
        atomic_json(destination / f'summary_seed{seed}.json', payload)
    return payload

def evaluate_beta_grid(cfg: Mapping[str, Any], dataset: str, seed: int, keys: Sequence[str], labels: np.ndarray | None=None, folds: np.ndarray | None=None, groups: np.ndarray | None=None, baseline: np.ndarray | None=None, geometry: np.ndarray | None=None) -> dict[str, Any] | None:
    from .beta_scale import BETA_GRID
    result_root = Path(cfg['paths']['result_root'])
    rows: list[dict] = []
    for fold in range(int(DATASET_SPEC[dataset]['folds'])):
        path = fold_prediction_path(result_root, dataset, 'G-beta', seed, fold).with_name('grid.parquet')
        if not path.is_file():
            return None
        rows.extend(read_parquet(path))
    if not rows:
        return None
    keyed: dict[str, dict[float, dict]] = defaultdict(dict)
    for row in rows:
        keyed[str(row['sample_id'])][float(row['beta'])] = row
    if labels is None or groups is None or folds is None:
        raise RuntimeError('beta grid evaluation requires aligned OOF identities')
    if set(keyed) != set(keys):
        raise RuntimeError('beta grid sample set does not match G')
    grid_scores = {}
    for beta in BETA_GRID:
        stacked = np.stack([np.asarray(keyed[key][float(beta)]['probability'], float) for key in keys])
        grid_scores[f'G-beta-{beta:g}'] = stacked
    if baseline is not None:
        grid_scores['B'] = baseline
    if geometry is not None:
        grid_scores['G'] = geometry
    if dataset == 'sicapv2':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, grid_scores, _qwk)
    elif dataset == 'bracs':
        observed, distributions = _paired_cluster_bootstrap(labels, groups, grid_scores, _macro_f1)
    else:
        scalar = {arm: value[:, 0] for arm, value in grid_scores.items()}
        observed, distributions = _fold_stratified_bootstrap(labels, folds, scalar)
    names = [f'G-beta-{beta:g}' for beta in BETA_GRID]
    comparisons = []
    for name in names:
        if 'B' in grid_scores:
            comparisons.append(comparison(observed, distributions, name, 'B'))
        if 'G' in grid_scores:
            comparisons.append(comparison(observed, distributions, name, 'G'))
    return {'status': 'PASS', 'dataset': dataset, 'seed': seed, 'kind': 'global_beta_sensitivity', 'selection': 'none: every pre-fixed β is reported; this is not a nested result', 'betas': [float(beta) for beta in BETA_GRID], 'samples': len(keys), 'metric': DATASET_SPEC[dataset]['metric'], 'observed': {key: float(observed[key]) for key in names}, 'comparisons': comparisons, 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'development_only': True, 'official_test_touched': False}
