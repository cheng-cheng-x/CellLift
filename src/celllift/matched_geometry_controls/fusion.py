from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Calibration:
    alpha: float
    baseline_temperature: float = 1.0
    geometry_temperature: float = 1.0

def center_logits(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    return value - value.mean(-1, keepdims=True)

def fit_multiclass_alpha(baseline: np.ndarray, geometry: np.ndarray, labels: np.ndarray) -> Calibration:
    import torch
    import torch.nn.functional as F
    b = torch.as_tensor(center_logits(baseline), dtype=torch.float64)
    g = torch.as_tensor(center_logits(geometry), dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    raw = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.2, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(b + 4.0 * torch.sigmoid(raw) * g, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return Calibration(float((4.0 * torch.sigmoid(raw)).detach()))

def fit_crc(baseline: np.ndarray, geometry: np.ndarray, labels: np.ndarray) -> Calibration:
    import torch
    import torch.nn.functional as F
    b = torch.as_tensor(baseline, dtype=torch.float64)
    g = torch.as_tensor(geometry, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    raw = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.2, max_iter=150, line_search_fn='strong_wolfe')

    def parameters():
        return (0.25 + 3.75 * torch.sigmoid(raw[0]), 0.25 + 3.75 * torch.sigmoid(raw[1]), 4.0 * torch.sigmoid(raw[2]))

    def closure():
        optimizer.zero_grad()
        tb, tg, alpha = parameters()
        loss = F.binary_cross_entropy_with_logits(b / tb + alpha * g / tg, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    tb, tg, alpha = (float(value.detach()) for value in parameters())
    return Calibration(alpha, tb, tg)

def apply(baseline: np.ndarray, geometry: np.ndarray, calibration: Calibration, multiclass: bool) -> np.ndarray:
    b = center_logits(baseline) if multiclass else np.asarray(baseline, float)
    g = center_logits(geometry) if multiclass else np.asarray(geometry, float)
    return b / calibration.baseline_temperature + calibration.alpha * g / calibration.geometry_temperature

def crossfit(folds: np.ndarray, baseline: np.ndarray, geometry: np.ndarray, labels: np.ndarray, *, crc: bool=False):
    folds = np.asarray(folds)
    output = np.empty_like(np.asarray(baseline), dtype=np.float64)
    parameters = {}
    for fold in sorted(set(map(int, folds))):
        held = folds == fold
        fit = fit_crc if crc else fit_multiclass_alpha
        calibration = fit(np.asarray(baseline)[~held], np.asarray(geometry)[~held], np.asarray(labels)[~held])
        output[held] = apply(np.asarray(baseline)[held], np.asarray(geometry)[held], calibration, not crc)
        parameters[fold] = calibration
    return (output, parameters)
