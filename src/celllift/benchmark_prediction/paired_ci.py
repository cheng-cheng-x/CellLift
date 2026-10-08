from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Sequence
import numpy as np
from .metrics import patient_auroc

def _confusion_batch(labels: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    labels = np.asarray(labels, np.int64)
    pred = np.asarray(pred, np.int64)
    batch, n = labels.shape
    flat = np.repeat(np.arange(batch), n) * (classes * classes) + labels.reshape(-1) * classes + pred.reshape(-1)
    valid = (labels.reshape(-1) >= 0) & (labels.reshape(-1) < classes) & (pred.reshape(-1) >= 0) & (pred.reshape(-1) < classes)
    counts = np.bincount(flat[valid], minlength=batch * classes * classes)
    return counts.reshape(batch, classes, classes)

def qwk_batch(labels: np.ndarray, pred: np.ndarray, classes: int=4) -> np.ndarray:
    matrix = _confusion_batch(labels, pred, classes)
    hist_y = matrix.sum(2)
    hist_p = matrix.sum(1)
    total = matrix.sum((1, 2)).clip(min=1.0)
    expected = hist_y[:, :, None] * hist_p[:, None, :] / total[:, None, None]
    idx = np.arange(classes, dtype=np.float64)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (classes - 1) ** 2
    num = (weights[None] * matrix).sum((1, 2))
    den = (weights[None] * expected).sum((1, 2))
    out = np.where(den > 0, 1.0 - num / den, 1.0)
    out[matrix.sum((1, 2)) == 0] = np.nan
    return out

def macro_f1_batch(labels: np.ndarray, pred: np.ndarray, classes: int) -> np.ndarray:
    matrix = _confusion_batch(labels, pred, classes)
    tp = np.diagonal(matrix, axis1=1, axis2=2)
    fp = matrix.sum(1) - tp
    fn = matrix.sum(2) - tp
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=prec + rec > 0)
    return f1.mean(1)

def auroc_loop(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    out = np.empty(labels.shape[0], np.float64)
    for i in range(labels.shape[0]):
        out[i] = patient_auroc(labels[i], scores[i])
    return out

def cluster_index(ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    keys = [str(v) if str(v) not in {'', 'None', 'nan'} else f'row_{i}' for i, v in enumerate(ids)]
    uniq = sorted(set(keys))
    loc = {key: i for i, key in enumerate(uniq)}
    return (np.asarray([loc[key] for key in keys], np.int64), uniq)

def resample_clusters(cluster: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n = cluster.size
    n_cluster = int(cluster.max()) + 1 if n else 0
    members = [np.flatnonzero(cluster == i) for i in range(n_cluster)]
    lengths = np.asarray([m.size for m in members], np.int64)
    drawn = rng.integers(0, n_cluster, size=(n_boot, n_cluster))
    out = np.empty((n_boot, n), np.int64)
    for b in range(n_boot):
        parts = [members[int(i)] for i in drawn[b]]
        cat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        if cat.size == 0:
            out[b] = np.arange(n)
            continue
        reps = int(np.ceil(n / cat.size))
        out[b] = np.tile(cat, reps)[:n]
    del lengths
    return out

def paired_delta(labels: np.ndarray, pred_a: np.ndarray | None, pred_b: np.ndarray | None, *, scores_a: np.ndarray | None=None, scores_b: np.ndarray | None=None, cluster_ids: Sequence[str], kind: str, classes: int=4, n_boot: int=10000, seed: int=42) -> dict[str, float]:
    cluster, names = cluster_index(cluster_ids)
    rng = np.random.default_rng(seed)
    idx = resample_clusters(cluster, n_boot, rng)
    y = np.asarray(labels)
    if kind == 'auroc':
        sa, sb = (np.asarray(scores_a, np.float64), np.asarray(scores_b, np.float64))
        boot_a = auroc_loop(y[idx], sa[idx])
        boot_b = auroc_loop(y[idx], sb[idx])
        point_a = patient_auroc(y, sa)
        point_b = patient_auroc(y, sb)
    else:
        pa, pb = (np.asarray(pred_a, np.int64), np.asarray(pred_b, np.int64))
        fn = qwk_batch if kind == 'qwk' else lambda yy, pp, c=classes: macro_f1_batch(yy, pp, c)
        boot_a = fn(y[idx], pa[idx])
        boot_b = fn(y[idx], pb[idx])
        if kind == 'qwk':
            from .metrics import qwk
            point_a, point_b = (qwk(y, pa, classes), qwk(y, pb, classes))
        else:
            from .metrics import macro_f1
            point_a, point_b = (macro_f1(y, pa, classes), macro_f1(y, pb, classes))
    delta = boot_b - boot_a
    lo, hi = np.nanquantile(delta, [0.025, 0.975])
    return {'a': float(point_a), 'b': float(point_b), 'delta': float(point_b - point_a), 'ci95_lo': float(lo), 'ci95_hi': float(hi), 'crosses_zero': bool(lo <= 0.0 <= hi), 'n': int(y.size), 'n_clusters': len(names), 'n_boot': n_boot, 'cluster': 'patient_or_wsi'}
