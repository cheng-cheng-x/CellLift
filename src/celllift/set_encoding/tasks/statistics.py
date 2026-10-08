from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_resamples: int
    n_valid_resamples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

def paired_cluster_bootstrap(y_true: object, prediction_a: object, prediction_b: object, cluster_ids: object, metric: Callable[[np.ndarray, np.ndarray], float], *, n_resamples: int=10000, confidence: float=0.95, seed: int=20260811) -> PairedBootstrapResult:
    truth = np.asarray(y_true)
    first = np.asarray(prediction_a)
    second = np.asarray(prediction_b)
    clusters = np.asarray(cluster_ids)
    if not truth.shape[0] == first.shape[0] == second.shape[0] == clusters.shape[0]:
        raise ValueError('truth, predictions, and cluster_ids must be sample aligned')
    if n_resamples <= 0:
        raise ValueError('n_resamples must be positive')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be between zero and one')
    unique_clusters = np.unique(clusters)
    if unique_clusters.size < 2:
        raise ValueError('at least two clusters are required')
    members = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    observed = float(metric(truth, first) - metric(truth, second))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    deltas.fill(np.nan)
    for resample_index in range(n_resamples):
        sampled = rng.integers(0, unique_clusters.size, size=unique_clusters.size)
        indices = np.concatenate([members[index] for index in sampled])
        value_a = metric(truth[indices], first[indices])
        value_b = metric(truth[indices], second[indices])
        deltas[resample_index] = value_a - value_b
    valid = deltas[np.isfinite(deltas)]
    if valid.size < max(100, int(0.5 * n_resamples)):
        raise RuntimeError('too few finite bootstrap replicates')
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(valid, (alpha / 2.0, 1.0 - alpha / 2.0))
    left = (np.sum(valid <= 0.0) + 1) / (valid.size + 1)
    right = (np.sum(valid >= 0.0) + 1) / (valid.size + 1)
    p_value = min(1.0, 2.0 * min(left, right))
    return PairedBootstrapResult(observed_delta=observed, ci_low=float(ci_low), ci_high=float(ci_high), p_value=float(p_value), n_resamples=n_resamples, n_valid_resamples=int(valid.size))

def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return result

def _fast_delong(sorted_predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, examples = sorted_predictions.shape
    negative_count = examples - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError('DeLong requires both positive and negative examples')
    positive = sorted_predictions[:, :positive_count]
    negative = sorted_predictions[:, positive_count:]
    tx = np.empty_like(positive, dtype=np.float64)
    ty = np.empty_like(negative, dtype=np.float64)
    tz = np.empty_like(sorted_predictions, dtype=np.float64)
    for classifier in range(classifiers):
        tx[classifier] = _midrank(positive[classifier])
        ty[classifier] = _midrank(negative[classifier])
        tz[classifier] = _midrank(sorted_predictions[classifier])
    aucs = tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    auxiliary_938db8 = (tz[:, :positive_count] - tx) / negative_count
    shape_warmstart = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(auxiliary_938db8, bias=False))
    sy = np.atleast_2d(np.cov(shape_warmstart, bias=False))
    covariance = sx / positive_count + sy / negative_count
    return (aucs, covariance)

def paired_delong(y_true: object, scores_a: object, scores_b: object) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    first = np.asarray(scores_a, dtype=np.float64).ravel()
    second = np.asarray(scores_b, dtype=np.float64).ravel()
    if not truth.shape == first.shape == second.shape:
        raise ValueError('labels and scores must have equal shape')
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError('labels must be binary')
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives < 2 or negatives < 2:
        raise ValueError('DeLong requires at least two positive and two negative examples')
    order = np.argsort(-truth, kind='mergesort')
    predictions = np.vstack((first, second))[:, order]
    aucs, covariance = _fast_delong(predictions, positives)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[0] - aucs[1])
    if variance <= 0:
        z_score = math.copysign(float('inf'), delta) if delta else 0.0
    else:
        z_score = delta / math.sqrt(variance)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]), 'delta': delta, 'variance': variance, 'z': float(z_score), 'p_value': float(p_value)}

def holm_adjust(p_values: Sequence[float] | Mapping[str, float], alpha: float=0.05) -> list[dict[str, float | bool]] | dict[str, dict[str, float | bool]]:
    is_mapping = isinstance(p_values, Mapping)
    keys = list(p_values.keys()) if is_mapping else list(range(len(p_values)))
    values = np.asarray(list(p_values.values()) if is_mapping else p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('p-values must be finite numbers in [0, 1]')
    order = np.argsort(values, kind='mergesort')
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    records = [{'p_value': float(values[index]), 'p_adjusted': float(adjusted[index]), 'reject': bool(adjusted[index] < alpha)} for index in range(count)]
    if is_mapping:
        return {str(key): records[index] for index, key in enumerate(keys)}
    return records
