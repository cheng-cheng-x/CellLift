from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter
import numpy as np

def primary_secondary_from_counts(counts: np.ndarray) -> tuple[int, int]:
    values = np.asarray(counts[:4], dtype=np.int64)
    order = np.argsort(values)
    primary = int(order[-1])
    secondary = int(order[-2])
    if int((values == 0).sum()) == 3:
        secondary = primary
    return (primary, secondary)

def gleason_summary_wsum(y_pred, *, thres: float=0.25) -> tuple[int, int, int]:
    scores = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if scores.size < 4:
        scores = np.pad(scores, (0, 4 - scores.size))
    scores = scores[:4].copy()
    total = float(scores.sum())
    if total <= 0:
        return (0, 0, 0)
    scores = scores / total
    if thres is not None:
        scores[scores < float(thres)] = 0.0
    order = np.argsort(scores)[::-1]
    primary = int(order[0])
    secondary = int(order[1]) if scores[order[1]] > 0 else primary
    return (primary, secondary, assign_group(primary, secondary))

def assign_group(primary: int, secondary: int, survival_groups: bool=False) -> int:
    a, b = (int(primary), int(secondary))
    if a > 0 and b == 0:
        b = a
    if b > 0 and a == 0:
        a = b
    if survival_groups:
        a += 2
        b += 2
        total = a + b
        if total <= 6:
            return 1
        if total == 7:
            return 2
        return 3
    return a + b

def gleason_sum_to_qwk_label(value: int) -> int:
    mapping = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return mapping.get(int(value), 0)

def aggregate_patch_logits(logits: np.ndarray) -> dict[str, int]:
    labels = np.argmax(np.asarray(logits), axis=-1)
    counts = np.bincount(labels.astype(np.int64), minlength=4)
    primary, secondary = primary_secondary_from_counts(counts)
    return {'primary': primary, 'secondary': secondary, 'hist': {str(index): int(counts[index]) for index in range(4)}}

def majority_class(labels: list[int]) -> int:
    return Counter((int(value) for value in labels)).most_common(1)[0][0]
