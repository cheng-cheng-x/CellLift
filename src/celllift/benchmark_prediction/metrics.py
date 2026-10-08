from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np

def qwk(labels: np.ndarray, pred: np.ndarray, classes: int=4) -> float:
    labels = np.asarray(labels, np.int64)
    pred = np.asarray(pred, np.int64)
    if labels.size == 0:
        return float('nan')
    matrix = np.zeros((classes, classes), np.float64)
    for y, p in zip(labels, pred):
        if 0 <= y < classes and 0 <= p < classes:
            matrix[y, p] += 1
    if matrix.sum() == 0:
        return float('nan')
    hist_y = matrix.sum(1)
    hist_p = matrix.sum(0)
    expected = np.outer(hist_y, hist_p) / matrix.sum()
    idx = np.arange(classes, dtype=np.float64)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (classes - 1) ** 2
    num = float((weights * matrix).sum())
    den = float((weights * expected).sum())
    return 1.0 - num / den if den else 1.0

def macro_f1(labels: np.ndarray, pred: np.ndarray, classes: int) -> float:
    labels = np.asarray(labels, np.int64)
    pred = np.asarray(pred, np.int64)
    scores = []
    for cls in range(classes):
        tp = float(((pred == cls) & (labels == cls)).sum())
        fp = float(((pred == cls) & (labels != cls)).sum())
        fn = float(((pred != cls) & (labels == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else float('nan')

def weighted_f1(labels: np.ndarray, pred: np.ndarray, classes: int) -> float:
    labels = np.asarray(labels, np.int64)
    pred = np.asarray(pred, np.int64)
    scores, weights = ([], [])
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
    total = float(sum(weights))
    return float(np.dot(scores, weights) / total) if total else float('nan')

def patient_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, np.int64)
    scores = np.asarray(scores, np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    order = np.argsort(np.concatenate([neg, pos]), kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    values = np.concatenate([neg, pos])
    sorted_v = values[order]
    start = 0
    while start < sorted_v.size:
        end = start + 1
        while end < sorted_v.size and sorted_v[end] == sorted_v[start]:
            end += 1
        ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    rank_pos = ranks[neg.size:]
    return float((rank_pos.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))
