from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np

def qwk(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(probabilities).argmax(1), weights='quadratic'))

def macro_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(np.asarray(y_true), np.asarray(probabilities).argmax(1), average='macro'))

def mean_fold_auroc(y_true: np.ndarray, scores: np.ndarray, folds: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    folds = np.asarray(folds)
    values = []
    for fold in sorted(set(map(int, folds))):
        selected = folds == fold
        if len(np.unique(y_true[selected])) < 2:
            return float('nan')
        values.append(roc_auc_score(y_true[selected], scores[selected]))
    return float(np.mean(values))

def paired_cluster_bootstrap(*, labels, groups, scores: dict[str, np.ndarray], metric, replicates=10000, seed=20260909):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    observed = {arm: metric(labels, value) for arm, value in scores.items()}
    distributions = {arm: np.empty(replicates) for arm in scores}
    members = {group: np.flatnonzero(groups == group) for group in unique}
    for r in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ids = np.concatenate([members[group] for group in sampled])
        for arm, value in scores.items():
            distributions[arm][r] = metric(labels[ids], value[ids])
    return (observed, distributions)

def fold_stratified_patient_bootstrap(*, labels, folds, scores: dict[str, np.ndarray], replicates=10000, seed=20260909):
    labels = np.asarray(labels)
    folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    observed = {arm: mean_fold_auroc(labels, value, folds) for arm, value in scores.items()}
    distributions = {arm: np.empty(replicates) for arm in scores}
    fold_ids = {fold: np.flatnonzero(folds == fold) for fold in sorted(set(map(int, folds)))}
    for r in range(replicates):
        sampled = np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in fold_ids.values()])
        for arm, value in scores.items():
            distributions[arm][r] = mean_fold_auroc(labels[sampled], value[sampled], folds[sampled])
    return (observed, distributions)

def comparison(observed: dict[str, float], distributions: dict[str, np.ndarray], first: str, second: str):
    delta = np.asarray(distributions[first]) - np.asarray(distributions[second])
    finite = delta[np.isfinite(delta)]
    if not len(finite):
        raise RuntimeError(f'bootstrap comparison has no finite replicates: {first}-{second}')
    return {'comparison': f'{first}-{second}', 'delta': observed[first] - observed[second], 'ci_low': float(np.quantile(finite, 0.025)), 'ci_high': float(np.quantile(finite, 0.975)), 'p_one_sided': float((1 + np.count_nonzero(finite <= 0)) / (len(finite) + 1)), 'finite_replicates': int(len(finite))}

def registered_interpretation(observed, distributions):
    rows = [comparison(observed, distributions, a, b) for a, b in (('D', 'B0'), ('D', 'DS'), ('R', 'B0'), ('R', 'RS'), ('D', 'R'))]
    lookup = {row['comparison']: row for row in rows}
    return {'comparisons': rows, 'direct_positive': lookup['D-B0']['ci_low'] > 0 and lookup['D-DS']['ci_low'] > 0}

def t7_to_t3(probabilities: np.ndarray) -> np.ndarray:
    value = np.asarray(probabilities, float)
    if value.ndim != 2 or value.shape[1] != 7:
        raise ValueError('T7 probabilities must be [N,7]')
    return np.column_stack((value[:, 0:3].sum(1), value[:, 3:5].sum(1), value[:, 5:7].sum(1)))
