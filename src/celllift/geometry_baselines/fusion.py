from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class Calibration:
    alpha: float
    rgb_temperature: float = 1.0
    geometry_temperature: float = 1.0

def fit_sicap_alpha(rgb_logits: np.ndarray, geometry_logits: np.ndarray, labels: np.ndarray) -> Calibration:
    if rgb_logits.shape != geometry_logits.shape or rgb_logits.ndim != 2 or rgb_logits.shape[1] != 4:
        raise ValueError('SICAP fusion requires matched [N,4] logits')
    alpha_raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    rgb = torch.as_tensor(rgb_logits, dtype=torch.float64)
    geometry = torch.as_tensor(geometry_logits, dtype=torch.float64)
    target = torch.as_tensor(labels, dtype=torch.long)
    optimizer = torch.optim.LBFGS([alpha_raw], lr=0.25, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        alpha = 4.0 * torch.sigmoid(alpha_raw)
        loss = F.cross_entropy(rgb + alpha * geometry, target)
        loss.backward()
        return loss
    optimizer.step(closure)
    return Calibration(alpha=float((4.0 * torch.sigmoid(alpha_raw)).detach()))

def apply_sicap(rgb_logits: np.ndarray, geometry_logits: np.ndarray, calibration: Calibration) -> np.ndarray:
    return np.asarray(rgb_logits, np.float64) + calibration.alpha * np.asarray(geometry_logits, np.float64)

def continuity_logit(positive_tiles: np.ndarray, total_tiles: np.ndarray) -> np.ndarray:
    positive_tiles, total_tiles = (np.asarray(positive_tiles, float), np.asarray(total_tiles, float))
    if np.any(total_tiles <= 0) or np.any(positive_tiles < 0) or np.any(positive_tiles > total_tiles):
        raise ValueError('invalid hard-vote counts')
    return np.log((positive_tiles + 0.5) / (total_tiles - positive_tiles + 0.5))

def probability_logit(probability: np.ndarray, epsilon: float=1e-06) -> np.ndarray:
    value = np.clip(np.asarray(probability, float), epsilon, 1.0 - epsilon)
    return np.log(value / (1.0 - value))

def fit_crc_calibration(rgb_logits: np.ndarray, geometry_logits: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None=None) -> Calibration:
    rgb = torch.as_tensor(rgb_logits, dtype=torch.float64)
    geometry = torch.as_tensor(geometry_logits, dtype=torch.float64)
    target = torch.as_tensor(labels, dtype=torch.float64)
    weight = torch.ones_like(target) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float64)
    if weight.shape != target.shape or torch.any(weight <= 0):
        raise ValueError('CRC calibration weights must be positive and label-aligned')
    raw = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.2, max_iter=150, line_search_fn='strong_wolfe')

    def parameters():
        tr = 0.25 + 3.75 * torch.sigmoid(raw[0])
        tg = 0.25 + 3.75 * torch.sigmoid(raw[1])
        alpha = 4.0 * torch.sigmoid(raw[2])
        return (tr, tg, alpha)

    def closure():
        optimizer.zero_grad()
        tr, tg, alpha = parameters()
        element = F.binary_cross_entropy_with_logits(rgb / tr + alpha * geometry / tg, target, reduction='none')
        loss = torch.sum(element * weight) / torch.sum(weight)
        loss.backward()
        return loss
    optimizer.step(closure)
    tr, tg, alpha = (value.detach().item() for value in parameters())
    return Calibration(alpha=alpha, rgb_temperature=tr, geometry_temperature=tg)

def apply_crc(rgb_logits: np.ndarray, geometry_logits: np.ndarray, calibration: Calibration) -> np.ndarray:
    return np.asarray(rgb_logits, float) / calibration.rgb_temperature + calibration.alpha * np.asarray(geometry_logits, float) / calibration.geometry_temperature

def cross_fit_sicap(folds: np.ndarray, rgb_logits: np.ndarray, geometry_logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[int, Calibration]]:
    output = np.empty_like(rgb_logits, dtype=np.float64)
    calibrations: dict[int, Calibration] = {}
    for fold in sorted(set(map(int, folds))):
        held = folds == fold
        calibration = fit_sicap_alpha(rgb_logits[~held], geometry_logits[~held], labels[~held])
        output[held] = apply_sicap(rgb_logits[held], geometry_logits[held], calibration)
        calibrations[fold] = calibration
    return (output, calibrations)

def cross_fit_crc(folds: np.ndarray, rgb_logits: np.ndarray, geometry_logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[int, Calibration]]:
    output = np.empty_like(rgb_logits, dtype=np.float64)
    calibrations: dict[int, Calibration] = {}
    for fold in sorted(set(map(int, folds))):
        held = folds == fold
        calibration = fit_crc_calibration(rgb_logits[~held], geometry_logits[~held], labels[~held])
        output[held] = apply_crc(rgb_logits[held], geometry_logits[held], calibration)
        calibrations[fold] = calibration
    return (output, calibrations)

def aggregate_crc_hard_vote(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row['patient_id']), int(row['fold'])].append(row)
    output = []
    for (patient, fold), values in sorted(groups.items()):
        if len(values) < 10:
            continue
        scores = np.asarray([float(row['msih_probability']) for row in values])
        positive = int((scores >= 0.5).sum())
        output.append({'patient_id': patient, 'fold': fold, 'label_id': int(values[0]['label_id']), 'tiles': len(values), 'positive_tiles': positive, 'hard_vote_score': positive / len(values), 'continuity_logit': float(continuity_logit(np.array([positive]), np.array([len(values)]))[0])})
    return output
