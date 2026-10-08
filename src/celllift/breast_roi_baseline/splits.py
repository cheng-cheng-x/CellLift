from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def assign_grouped_folds(rows: Sequence[Mapping[str, Any]], *, folds: int=5, seed: int=20260830, group_key: str='wsi_id', label_key: str='label_7', search_starts: int=256) -> dict[str, int]:
    if folds < 2:
        raise ValueError('folds must be at least two')
    ordered = sorted(((str(row[group_key]), int(row[label_key])) for row in rows), key=lambda item: (item[0], item[1]))
    groups = np.asarray([item[0] for item in ordered], dtype=object)
    labels = np.asarray([item[1] for item in ordered], dtype=np.int64)
    unique_groups = set(groups.tolist())
    if len(unique_groups) < folds:
        raise ValueError('fewer WSI groups than folds')
    if search_starts < 1:
        raise ValueError('search_starts must be positive')
    expected_labels = set(labels.tolist())
    total_by_label = Counter(labels.tolist())
    target_fraction = 1.0 / folds
    dummy = np.zeros(len(labels), dtype=np.uint8)
    best: tuple[tuple[float, ...], dict[str, int]] | None = None
    for attempt in range(search_starts):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + attempt)
        candidate: dict[str, int] = {}
        fold_indices: list[np.ndarray] = []
        for fold, (_, heldout_indices) in enumerate(splitter.split(dummy, labels, groups)):
            fold_indices.append(heldout_indices)
            for group in set(groups[heldout_indices].tolist()):
                if group in candidate:
                    raise AssertionError(f'WSI {group!r} assigned to more than one fold')
                candidate[group] = fold
        missing_labels = 0
        label_error = 0.0
        roi_error = 0.0
        group_error = 0.0
        for heldout_indices in fold_indices:
            fold_labels = Counter(labels[heldout_indices].tolist())
            missing_labels += len(expected_labels - set(fold_labels))
            label_error += sum(((fold_labels[label] / total_by_label[label] - target_fraction) ** 2 for label in expected_labels))
            roi_error += (len(heldout_indices) / len(labels) - target_fraction) ** 2
            fold_group_count = len(set(groups[heldout_indices].tolist()))
            group_error += (fold_group_count / len(unique_groups) - target_fraction) ** 2
        score = (float(missing_labels), label_error + 0.25 * roi_error + 0.05 * group_error, float(attempt))
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    assignment = best[1]
    if set(assignment) != unique_groups:
        raise AssertionError('not every WSI received a fold')
    for fold in range(folds):
        fold_labels = {label for group, label in ordered if assignment[group] == fold}
        missing = expected_labels - fold_labels
        if missing:
            raise ValueError(f'fold {fold} is missing labels {sorted(missing)}; the grouped stratification is not valid')
    return assignment

def attach_development_folds(rows: Iterable[Mapping[str, Any]], *, folds: int=5, seed: int=20260830) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    training = [row for row in output if str(row['split_new']).lower() == 'train']
    assignment = assign_grouped_folds(training, folds=folds, seed=seed)
    for row in output:
        row['validation_fold'] = assignment[str(row['wsi_id'])] if str(row['split_new']).lower() == 'train' else None
    return output

def attach_final_folds(rows: Iterable[Mapping[str, Any]], *, folds: int=5, seed: int=20260831) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    development = [row for row in output if str(row['split_new']).lower() in {'train', 'val'}]
    assignment = assign_grouped_folds(development, folds=folds, seed=seed)
    for row in output:
        row['final_validation_fold'] = assignment[str(row['wsi_id'])] if str(row['split_new']).lower() in {'train', 'val'} else None
    return output
