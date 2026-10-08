from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import numpy as np
RAY_DIM = 36
TAIL_DIM = 5
TOKEN_DIM = 41

@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    ncr_median: float

    @classmethod
    def fit(cls, geometry9: np.ndarray, valid_ncr: np.ndarray, training: np.ndarray) -> 'FeatureNormalizer':
        raw = np.asarray(geometry9, np.float32)
        valid = np.asarray(valid_ncr, bool)
        train = np.asarray(training, bool)
        if raw.ndim != 2 or raw.shape[1] != 9 or valid.shape != (len(raw),) or (train.shape != (len(raw),)):
            raise ValueError('invalid normalizer arrays')
        if not train.any() or not (train & valid).any():
            raise ValueError('normalizer training rows are empty')
        median = float(np.median(raw[train & valid, 8]))
        filled = raw.copy()
        filled[~valid, 8] = median
        mean = filled[train].mean(0)
        std = np.maximum(filled[train].std(0), 1e-06)
        return cls(mean.astype(np.float32), std.astype(np.float32), median)

    def transform(self, geometry9: np.ndarray, valid_ncr: np.ndarray) -> np.ndarray:
        raw = np.asarray(geometry9, np.float32).copy()
        valid = np.asarray(valid_ncr, bool)
        raw[~valid, 8] = self.ncr_median
        output = (raw - self.mean) / self.std
        output[~valid, 8] = 0.0
        if not np.all(np.isfinite(output)):
            raise ValueError('standardized geometry contains NaN/Inf')
        return output.astype(np.float32)

def branch_tails(geometry9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(geometry9, np.float32)
    if value.ndim != 2 or value.shape[1] != 9:
        raise ValueError('geometry must have shape [N,9]')
    return (np.column_stack((value[:, :4], value[:, 8])), np.column_stack((value[:, 4:8], value[:, 8])))

def make_tokens(rays36: np.ndarray, geometry9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rays = np.asarray(rays36, np.float32)
    if rays.ndim != 2 or rays.shape[1] != RAY_DIM:
        raise ValueError('rays must have shape [N,36]')
    nucleus, cell = branch_tails(geometry9)
    if len(nucleus) != len(rays):
        raise ValueError('ray/geometry row mismatch')
    return (np.concatenate((rays, nucleus), 1), np.concatenate((rays, cell), 1))

def preaggregate(tokens: np.ndarray) -> tuple[np.ndarray, int]:
    value = np.asarray(tokens, np.float32)
    if value.ndim != 2 or value.shape[1] != TOKEN_DIM or len(value) == 0:
        raise ValueError('tokens must be nonempty [N,41]')
    return (value.mean(0, keepdims=True, dtype=np.float64).astype(np.float32), len(value))
