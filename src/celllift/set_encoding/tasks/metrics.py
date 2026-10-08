from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Iterable
import numpy as np

def _as_numpy(values: object) -> np.ndarray:
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values)

def quadratic_weighted_kappa(y_true: object, y_pred: object, num_classes: int | None=None) -> float:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    prediction = _as_numpy(y_pred).astype(np.int64).ravel()
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError('y_true and y_pred must be non-empty arrays of equal shape')
    if num_classes is None:
        num_classes = int(max(truth.max(), prediction.max())) + 1
    if num_classes < 2:
        return 1.0
    if np.any((truth < 0) | (truth >= num_classes)) or np.any((prediction < 0) | (prediction >= num_classes)):
        raise ValueError('labels fall outside num_classes')
    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(observed, (truth, prediction), 1.0)
    truth_hist = np.bincount(truth, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(prediction, minlength=num_classes).astype(np.float64)
    expected = np.outer(truth_hist, pred_hist) / truth.size
    indices = np.arange(num_classes, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / (num_classes - 1)) ** 2
    denominator = np.sum(weights * expected)
    if denominator == 0:
        return 1.0 if np.array_equal(truth, prediction) else float('nan')
    return float(1.0 - np.sum(weights * observed) / denominator)

def binary_auroc(y_true: object, scores: object) -> float:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    values = _as_numpy(scores).astype(np.float64).ravel()
    if truth.shape != values.shape:
        raise ValueError('y_true and scores must have equal shape')
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError('binary labels must be 0 or 1')
    positive = truth == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _midranks(values)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def binary_auprc(y_true: object, scores: object) -> float:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    values = _as_numpy(scores).astype(np.float64).ravel()
    if truth.shape != values.shape:
        raise ValueError('y_true and scores must have equal shape')
    n_pos = int((truth == 1).sum())
    if n_pos == 0:
        return float('nan')
    order = np.argsort(-values, kind='mergesort')
    sorted_values = values[order]
    sorted_truth = truth[order]
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_values)), truth.size - 1]
    true_positive = np.cumsum(sorted_truth == 1)[group_ends]
    predicted_positive = group_ends + 1
    precision = true_positive / predicted_positive
    recall = true_positive / n_pos
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))

def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks

def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    return confusion

def multiclass_metrics(y_true: object, probabilities: object, class_names: Iterable[str] | None=None) -> dict[str, float]:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    probs = _as_numpy(probabilities).astype(np.float64)
    if probs.ndim != 2 or probs.shape[0] != truth.size:
        raise ValueError('probabilities must have shape [samples, classes]')
    num_classes = probs.shape[1]
    if class_names is None:
        names = [str(index) for index in range(num_classes)]
    else:
        names = list(class_names)
        if len(names) != num_classes:
            raise ValueError('class_names length does not match probability columns')
    prediction = probs.argmax(axis=1)
    confusion = _confusion_counts(truth, prediction, num_classes)
    true_positive = np.diag(confusion).astype(np.float64)
    recall_denominator = confusion.sum(axis=1)
    precision_denominator = confusion.sum(axis=0)
    recalls = np.divide(true_positive, recall_denominator, out=np.full(num_classes, np.nan), where=recall_denominator > 0)
    precisions = np.divide(true_positive, precision_denominator, out=np.full(num_classes, np.nan), where=precision_denominator > 0)
    f1 = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros(num_classes), where=precisions + recalls > 0)
    result = {'qwk': quadratic_weighted_kappa(truth, prediction, num_classes), 'macro_f1': float(np.nanmean(f1)), 'balanced_accuracy': float(np.nanmean(recalls))}
    for index, name in enumerate(names):
        result[f'auroc_{name}'] = binary_auroc((truth == index).astype(np.int64), probs[:, index])
    return result

def binary_metrics(y_true: object, scores: object, threshold: float=0.5) -> dict[str, float]:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    values = _as_numpy(scores).astype(np.float64).ravel()
    if truth.shape != values.shape or not np.all(np.isin(truth, (0, 1))):
        raise ValueError('binary labels and scores must be equal-shaped, with labels in {0,1}')
    prediction = values >= threshold
    positive = truth == 1
    negative = truth == 0
    tp = int(np.sum(prediction & positive))
    tn = int(np.sum(~prediction & negative))
    sensitivity = tp / positive.sum() if positive.any() else float('nan')
    specificity = tn / negative.sum() if negative.any() else float('nan')
    return {'auroc': binary_auroc(truth, values), 'auprc': binary_auprc(truth, values), 'balanced_accuracy': float(np.nanmean((sensitivity, specificity))), 'sensitivity': float(sensitivity), 'specificity': float(specificity), 'threshold': float(threshold)}

def select_youden_threshold(y_true: object, scores: object) -> float:
    truth = _as_numpy(y_true).astype(np.int64).ravel()
    values = _as_numpy(scores).astype(np.float64).ravel()
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))
    best_threshold = 0.5
    best_j = -np.inf
    for threshold in candidates:
        metrics = binary_metrics(truth, values, float(threshold))
        j = metrics['sensitivity'] + metrics['specificity'] - 1.0
        if j > best_j or (j == best_j and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_j = j
            best_threshold = float(threshold)
    return best_threshold
