from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Sequence
import numpy as np
from .config import TASK_SPEC

def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    squeeze = False
    if value.ndim == 1:
        value = value.reshape(1, -1)
        squeeze = True
    shifted = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    out = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float64)
    return out[0] if squeeze else out

def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float('nan')

def confusion(y: np.ndarray, pred: np.ndarray, classes: int) -> list[list[int]]:
    table = np.zeros((classes, classes), np.int64)
    for truth, guess in zip(y.astype(int), pred.astype(int)):
        if 0 <= truth < classes and 0 <= guess < classes:
            table[truth, guess] += 1
    return table.tolist()

def per_class_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> list[float]:
    scores = []
    for index in range(classes):
        tp = int(((y == index) & (pred == index)).sum())
        fp = int(((y != index) & (pred == index)).sum())
        fn = int(((y == index) & (pred != index)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return scores

def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    return _safe(float(np.mean(per_class_f1(y, pred, classes))))

def recall_per_class(y: np.ndarray, pred: np.ndarray, classes: int) -> list[float]:
    out = []
    for index in range(classes):
        denom = int((y == index).sum())
        out.append(float('nan') if denom == 0 else float(((y == index) & (pred == index)).sum() / denom))
    return out

def balanced_accuracy(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    return _safe(float(np.nanmean(recall_per_class(y, pred, classes))))

def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, np.int64)
    score = np.asarray(score, np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    wins = 0.0
    for value in pos:
        wins += float((neg < value).sum()) + 0.5 * float((neg == value).sum())
    return _safe(wins / (len(pos) * len(neg)))

def macro_ovr_auroc(y: np.ndarray, probs: np.ndarray) -> float:
    classes = probs.shape[1]
    scores = []
    for index in range(classes):
        scores.append(roc_auc((y == index).astype(np.int64), probs[:, index]))
    finite = [value for value in scores if np.isfinite(value)]
    return _safe(float(np.mean(finite))) if finite else float('nan')

def argmax_pred(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1).astype(np.int64)

def binary_threshold(y: np.ndarray, p1: np.ndarray) -> float:
    y = np.asarray(y, np.int64)
    p1 = np.asarray(p1, np.float64)
    grid = np.unique(np.concatenate(([0.0, 1.0], np.quantile(p1, np.linspace(0, 1, 99)))))
    best_t, best = (0.5, -1.0)
    for threshold in grid:
        pred = (p1 >= threshold).astype(np.int64)
        score = macro_f1(y, pred, 2)
        if score > best + 1e-12 or (abs(score - best) <= 1e-12 and threshold < best_t):
            best, best_t = (score, float(threshold))
    return float(best_t)

def primary_from_probs(task: str, y: np.ndarray, probs: np.ndarray, threshold: float | None=None) -> tuple[float, dict]:
    spec = TASK_SPEC[task]
    classes = spec['classes']
    y = np.asarray(y, np.int64)
    probs = np.asarray(probs, np.float64)
    if classes == 2:
        pred = (probs[:, 1] >= (0.5 if threshold is None else threshold)).astype(np.int64)
        primary = roc_auc(y, probs[:, 1]) if spec['primary'] == 'auroc' else macro_f1(y, pred, 2)
        aux = {'auroc': roc_auc(y, probs[:, 1]), 'macro_f1': macro_f1(y, pred, 2), 'balanced_accuracy': balanced_accuracy(y, pred, 2), 'threshold': float(0.5 if threshold is None else threshold)}
    else:
        pred = argmax_pred(probs)
        primary = macro_f1(y, pred, classes) if spec['primary'] == 'macro_f1' else macro_ovr_auroc(y, probs)
        aux = {'macro_f1': macro_f1(y, pred, classes), 'macro_ovr_auroc': macro_ovr_auroc(y, probs), 'recall': recall_per_class(y, pred, classes), 'basal_vs_rest_auroc': roc_auc((y == 3).astype(np.int64), probs[:, 3]) if classes == 4 else float('nan')}
    aux['confusion'] = confusion(y, pred, classes)
    aux['n'] = int(len(y))
    return (float(primary), aux)

def better(left: float, right: float) -> bool:
    if not np.isfinite(left):
        return False
    if not np.isfinite(right):
        return True
    return left > right + 1e-12

def paired_bootstrap(y: np.ndarray, left: np.ndarray, right: np.ndarray, task: str, n: int=1000, seed: int=42) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, np.int64)
    deltas = []
    for _ in range(int(n)):
        index = rng.integers(0, len(y), len(y))
        a, _ = primary_from_probs(task, y[index], left[index])
        b, _ = primary_from_probs(task, y[index], right[index])
        deltas.append(a - b)
    values = np.asarray(deltas, np.float64)
    return {'mean': _safe(float(values.mean())), 'ci95': [_safe(float(np.percentile(values, 2.5))), _safe(float(np.percentile(values, 97.5)))], 'n': int(n)}

def stack_probs(rows: Sequence[dict], classes: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([int(row['label']) for row in rows], np.int64)
    probs = np.asarray([row['prob'] for row in rows], np.float64)
    if probs.ndim != 2 or probs.shape[1] != classes:
        raise ValueError(f'prob shape {probs.shape} != [N,{classes}]')
    return (y, probs)
