from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
import numpy as np
from celllift.runtime import torch
from torch import Tensor
from torch.nn import functional as F
from .metrics import binary_auprc, binary_auroc, multiclass_metrics
SICAP_CLASS_NAMES = ('NC', 'G3', 'G4', 'G5')

def sicap_loss(outputs: dict[str, Tensor], grade_targets: Tensor, *, class_weights: Optional[Tensor]=None, cribriform_targets: Optional[Tensor]=None, cribriform_valid: Optional[Tensor]=None, cribriform_weight: float=1.0) -> dict[str, Tensor]:
    grade_logits = outputs['grade_logits']
    cribriform_logits = outputs['cribriform_logits']
    if grade_logits.ndim != 2 or grade_logits.shape[1] != 4:
        raise ValueError('grade_logits must have shape [B, 4]')
    if grade_targets.shape != grade_logits.shape[:1]:
        raise ValueError('grade_targets must have shape [B]')
    grade = F.cross_entropy(grade_logits, grade_targets.long(), weight=class_weights)
    auxiliary = cribriform_logits.sum() * 0.0
    valid_count = 0
    if cribriform_targets is not None:
        if cribriform_targets.shape != cribriform_logits.shape:
            raise ValueError('cribriform_targets must have shape [B]')
        if cribriform_valid is None:
            cribriform_valid = torch.isfinite(cribriform_targets)
        if cribriform_valid.shape != cribriform_logits.shape:
            raise ValueError('cribriform_valid must have shape [B]')
        valid = cribriform_valid.bool() & torch.isfinite(cribriform_targets)
        if torch.any(valid & (grade_targets != 2)):
            raise ValueError('cribriform supervision is valid only for G4 patches')
        valid_count = int(valid.sum().item())
        if valid_count:
            auxiliary = F.binary_cross_entropy_with_logits(cribriform_logits[valid], cribriform_targets[valid].float())
    total = grade + float(cribriform_weight) * auxiliary
    return {'loss': total, 'grade_loss': grade, 'cribriform_loss': auxiliary, 'cribriform_valid_count': grade.new_tensor(valid_count, dtype=torch.long)}

def sicap_metrics(grade_targets: object, grade_probabilities: object, *, cribriform_targets: object | None=None, cribriform_scores: object | None=None, cribriform_valid: object | None=None) -> dict[str, float]:
    result = multiclass_metrics(grade_targets, grade_probabilities, class_names=SICAP_CLASS_NAMES)
    if cribriform_targets is not None and cribriform_scores is not None:
        targets = np.asarray(cribriform_targets)
        scores = np.asarray(cribriform_scores)
        valid = np.isfinite(targets) if cribriform_valid is None else np.asarray(cribriform_valid).astype(bool)
        valid &= np.isfinite(targets) & np.isfinite(scores)
        result['cribriform_auroc'] = binary_auroc(targets[valid], scores[valid])
        result['cribriform_auprc'] = binary_auprc(targets[valid], scores[valid])
        result['cribriform_n'] = float(valid.sum())
    return result

def inverse_frequency_class_weights(training_labels: object, num_classes: int=4) -> Tensor:
    labels = np.asarray(training_labels, dtype=np.int64).ravel()
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError('every class must be represented in the training fold')
    weights = 1.0 / counts
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
