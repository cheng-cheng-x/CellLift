from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections.abc import Hashable, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor
from torch.nn import functional as F
from .metrics import binary_metrics, select_youden_threshold

def crc_patient_loss(patient_logits: Tensor, patient_targets: Tensor) -> Tensor:
    if patient_logits.ndim != 1 or patient_targets.shape != patient_logits.shape:
        raise ValueError('patient logits and targets must both have shape [patients]')
    return F.binary_cross_entropy_with_logits(patient_logits, patient_targets.float())

def crc_patient_metrics(patient_targets: object, patient_scores: object, *, threshold: float) -> dict[str, float]:
    return binary_metrics(patient_targets, patient_scores, threshold)

def validation_threshold(patient_targets: object, patient_scores: object) -> float:
    return select_youden_threshold(patient_targets, patient_scores)

def historical_tile_hard_vote(tile_scores: object, patient_ids: Sequence[Hashable], *, tile_threshold: float=0.5) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(tile_scores, dtype=np.float64).ravel()
    ids = np.asarray(patient_ids, dtype=object).ravel()
    if scores.shape != ids.shape:
        raise ValueError('tile_scores and patient_ids must be tile aligned')
    ordered_ids = list(dict.fromkeys(ids.tolist()))
    patient_scores = np.asarray([np.mean(scores[ids == patient] >= tile_threshold) for patient in ordered_ids], dtype=np.float64)
    return (np.asarray(ordered_ids, dtype=object), patient_scores)
