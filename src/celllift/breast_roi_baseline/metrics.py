from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class PairedBootstrapResult:
    observed_delta: float
    ci_low: float
    ci_high: float
    p_value_one_sided: float
    n_resamples: int
    n_valid_resamples: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

def _classes_from(truth: np.ndarray, *predictions: np.ndarray) -> np.ndarray:
    max_label = int(truth.max(initial=-1))
    for prediction in predictions:
        if prediction.ndim == 2:
            max_label = max(max_label, prediction.shape[1] - 1)
        elif prediction.size:
            max_label = max(max_label, int(prediction.max()))
    if max_label < 1:
        raise ValueError('macro-F1 requires at least two registered classes')
    return np.arange(max_label + 1, dtype=np.int64)

def _class_predictions(prediction: object) -> np.ndarray:
    values = np.asarray(prediction)
    if values.ndim == 2:
        if values.shape[1] < 2 or not np.all(np.isfinite(values)):
            raise ValueError('probabilities must be a finite N x K matrix')
        return values.argmax(axis=1).astype(np.int64, copy=False)
    if values.ndim == 1:
        if not np.all(np.isfinite(values)):
            raise ValueError('class predictions must be finite')
        return values.astype(np.int64, copy=False)
    raise ValueError('predictions must be class IDs or an N x K probability matrix')

def macro_f1(y_true: object, prediction: object, *, classes: object | None=None) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = _class_predictions(prediction)
    if truth.ndim != 1 or predicted.shape != truth.shape or truth.size == 0:
        raise ValueError('truth and predictions must be aligned non-empty vectors')
    labels = _classes_from(truth, np.asarray(prediction)) if classes is None else np.asarray(classes, dtype=np.int64)
    if labels.ndim != 1 or labels.size < 2 or np.unique(labels).size != labels.size:
        raise ValueError('classes must contain at least two unique class IDs')
    scores = np.empty(labels.size, dtype=np.float64)
    for index, label in enumerate(labels):
        true_positive = np.sum((truth == label) & (predicted == label))
        false_positive = np.sum((truth != label) & (predicted == label))
        false_negative = np.sum((truth == label) & (predicted != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores[index] = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
    return float(scores.mean())

def paired_wsi_cluster_bootstrap(y_true: object, candidate_prediction: object, baseline_prediction: object, wsi_ids: object, *, n_resamples: int=10000, confidence: float=0.95, seed: int=20260811) -> PairedBootstrapResult:
    truth = np.asarray(y_true, dtype=np.int64)
    candidate_raw = np.asarray(candidate_prediction)
    baseline_raw = np.asarray(baseline_prediction)
    candidate = _class_predictions(candidate_raw)
    baseline = _class_predictions(baseline_raw)
    clusters = np.asarray(wsi_ids)
    if truth.ndim != 1 or not len(truth) == len(candidate) == len(baseline) == len(clusters):
        raise ValueError('truth, predictions, and WSI IDs must be sample aligned')
    if n_resamples <= 0:
        raise ValueError('n_resamples must be positive')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must lie strictly between zero and one')
    unique_clusters = np.unique(clusters)
    if unique_clusters.size < 2:
        raise ValueError('at least two WSI clusters are required')
    members = [np.flatnonzero(clusters == cluster) for cluster in unique_clusters]
    classes = _classes_from(truth, candidate_raw, baseline_raw)
    observed = macro_f1(truth, candidate, classes=classes) - macro_f1(truth, baseline, classes=classes)
    rng = np.random.default_rng(seed)
    deltas = np.full(n_resamples, np.nan, dtype=np.float64)
    for resample_index in range(n_resamples):
        sampled = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in sampled])
        deltas[resample_index] = macro_f1(truth[indices], candidate[indices], classes=classes) - macro_f1(truth[indices], baseline[indices], classes=classes)
    valid = deltas[np.isfinite(deltas)]
    if valid.size < max(100, int(0.5 * n_resamples)):
        raise RuntimeError('too few finite WSI bootstrap replicates')
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(valid, [tail, 1.0 - tail])
    p_value = (np.sum(valid <= 0.0) + 1.0) / (valid.size + 1.0)
    return PairedBootstrapResult(observed_delta=float(observed), ci_low=float(low), ci_high=float(high), p_value_one_sided=float(p_value), n_resamples=int(n_resamples), n_valid_resamples=int(valid.size), seed=int(seed))

def holm_correction(p_values: Sequence[float] | Mapping[str, float], *, alpha: float=0.05) -> list[dict[str, float | bool]] | dict[str, dict[str, float | bool]]:
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must lie strictly between zero and one')
    is_mapping = isinstance(p_values, Mapping)
    keys = list(p_values.keys()) if is_mapping else list(range(len(p_values)))
    raw = list(p_values.values()) if is_mapping else list(p_values)
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('at least one p-value is required')
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError('p-values must be finite and lie in [0, 1]')
    order = np.argsort(values, kind='mergesort')
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        running = max(running, (count - rank) * values[original_index])
        adjusted[original_index] = min(1.0, running)
    records = [{'p_value': float(values[index]), 'p_adjusted': float(adjusted[index]), 'reject': bool(adjusted[index] < alpha)} for index in range(count)]
    if is_mapping:
        return {str(key): records[index] for index, key in enumerate(keys)}
    return records
