from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from typing import Callable, Mapping, Sequence
import numpy as np

def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted

def paired_cluster_bootstrap(rows: Sequence[Mapping[str, object]], score_a: str, score_b: str, metric: Callable[[np.ndarray, np.ndarray], float], *, replicates: int=10000, seed: int=20260831, stratify_key: str | None=None) -> dict[str, float]:
    patients: dict[tuple[object, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        stratum = row[stratify_key] if stratify_key is not None else 'all'
        patients[stratum, str(row['patient_id'])].append(index)
    strata: dict[object, list[list[int]]] = defaultdict(list)
    for (stratum, _), indices in patients.items():
        strata[stratum].append(indices)
    labels = np.asarray([int(row['label_id']) for row in rows])
    a = np.asarray([float(row[score_a]) for row in rows])
    b = np.asarray([float(row[score_b]) for row in rows])
    observed = metric(labels, a) - metric(labels, b)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, np.float64)
    for replicate in range(replicates):
        indices: list[int] = []
        for clusters in strata.values():
            chosen = rng.integers(0, len(clusters), size=len(clusters))
            for index in chosen:
                indices.extend(clusters[int(index)])
        selected = np.asarray(indices, np.int64)
        try:
            deltas[replicate] = metric(labels[selected], a[selected]) - metric(labels[selected], b[selected])
        except ValueError:
            deltas[replicate] = np.nan
    valid = deltas[np.isfinite(deltas)]
    if len(valid) < 0.95 * replicates:
        raise RuntimeError('too many invalid bootstrap replicates')
    p_one_sided = float((1 + np.sum(valid <= 0)) / (1 + len(valid)))
    return {'delta': float(observed), 'ci_low': float(np.quantile(valid, 0.025)), 'ci_high': float(np.quantile(valid, 0.975)), 'p_one_sided': p_one_sided, 'replicates': int(len(valid))}
