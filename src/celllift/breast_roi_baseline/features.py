from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from typing import Literal
import numpy as np
RAY_DIM = 36
OBJECT_GEOMETRY_DIM = 4
TAIL_DIM = 5
TOKEN_DIM = RAY_DIM + TAIL_DIM
JOINT_GEOMETRY_DIM = 2 * OBJECT_GEOMETRY_DIM + 1
GeometryArm = Literal['mask2d', 'direct3d', 'shuffled_direct3d', 'residual3d', 'shuffled_residual3d']
GEOMETRY_ARMS = frozenset({'mask2d', 'direct3d', 'shuffled_direct3d', 'residual3d', 'shuffled_residual3d'})

@dataclass(frozen=True)
class FoldRayNormalizer:
    mean: np.ndarray
    std: np.ndarray
    training_anchor_count: int

    @classmethod
    def fit(cls, rays36: np.ndarray, training_mask: np.ndarray, *, min_std: float=1e-06) -> 'FoldRayNormalizer':
        rays = _float_matrix(rays36, RAY_DIM, 'rays36')
        training = _boolean_vector(training_mask, len(rays), 'training_mask')
        if not training.any():
            raise ValueError('training_mask selects no anchors')
        mean = rays[training].mean(axis=0).astype(np.float32)
        std = np.maximum(rays[training].std(axis=0), np.float32(min_std)).astype(np.float32)
        return cls(mean=mean, std=std, training_anchor_count=int(training.sum()))

    def transform(self, rays36: np.ndarray) -> np.ndarray:
        rays = _float_matrix(rays36, RAY_DIM, 'rays36')
        standardized = (rays - self.mean) / self.std
        if not np.isfinite(standardized).all():
            raise ValueError('standardized MASK2D rays contain NaN/Inf')
        return standardized.astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(), 'training_anchor_count': self.training_anchor_count}

def _float_matrix(value: np.ndarray, columns: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f'{name} must have shape [N, {columns}], got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return result

def _boolean_vector(value: np.ndarray, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.shape != (rows,):
        raise ValueError(f'{name} must have shape [{rows}], got {result.shape}')
    return result

def complete_geometry_from_ellipsoids(nucleus_axes: np.ndarray, nucleus_rotation: np.ndarray, cell_axes: np.ndarray, cell_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_axes = _float_matrix(nucleus_axes, 3, 'nucleus_axes')
    c_axes = _float_matrix(cell_axes, 3, 'cell_axes')
    if n_axes.shape != c_axes.shape:
        raise ValueError('nucleus_axes and cell_axes must have identical shape')
    if np.any(n_axes <= 0) or np.any(c_axes <= 0):
        raise ValueError('ellipsoid axes must be positive')
    if np.any(n_axes[:, :-1] < n_axes[:, 1:]) or np.any(c_axes[:, :-1] < c_axes[:, 1:]):
        raise ValueError('ellipsoid axes must be ordered a >= b >= c')
    n_rotation = np.asarray(nucleus_rotation, dtype=np.float32)
    c_rotation = np.asarray(cell_rotation, dtype=np.float32)
    expected = (len(n_axes), 3, 3)
    if n_rotation.shape != expected or c_rotation.shape != expected:
        raise ValueError(f'rotation matrices must have shape {expected}')
    if not np.isfinite(n_rotation).all() or not np.isfinite(c_rotation).all():
        raise ValueError('rotation matrices contain NaN/Inf')
    scale = np.float32(4.0 * np.pi / 3.0)
    n_volume = scale * np.prod(n_axes, axis=1)
    c_volume = scale * np.prod(c_axes, axis=1)

    def object_features(axes: np.ndarray, rotation: np.ndarray, volume: np.ndarray) -> np.ndarray:
        return np.column_stack((np.log(volume), np.log(axes[:, 0] / axes[:, 1]), np.log(axes[:, 1] / axes[:, 2]), np.square(rotation[:, 2, 0]))).astype(np.float32)
    valid_ncr = np.isfinite(n_volume) & np.isfinite(c_volume) & (n_volume > 0) & (c_volume > n_volume)
    log_ncr = np.full(len(n_axes), np.nan, dtype=np.float32)
    log_ncr[valid_ncr] = np.log(n_volume[valid_ncr]) - np.log(c_volume[valid_ncr] - n_volume[valid_ncr])
    raw = np.column_stack((object_features(n_axes, n_rotation, n_volume), object_features(c_axes, c_rotation, c_volume), log_ncr)).astype(np.float32)
    if not np.isfinite(raw[:, :8]).all():
        raise ValueError('derived non-NCR geometry contains NaN/Inf')
    return (raw, valid_ncr)

@dataclass(frozen=True)
class FoldFeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    ncr_median: float
    training_anchor_count: int
    training_valid_ncr_count: int

    @classmethod
    def fit(cls, raw_geometry: np.ndarray, valid_ncr: np.ndarray, training_mask: np.ndarray, *, min_std: float=1e-06) -> 'FoldFeatureNormalizer':
        raw = np.asarray(raw_geometry, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != JOINT_GEOMETRY_DIM:
            raise ValueError(f'raw_geometry must have shape [N, {JOINT_GEOMETRY_DIM}], got {raw.shape}')
        valid = _boolean_vector(valid_ncr, len(raw), 'valid_ncr')
        training = _boolean_vector(training_mask, len(raw), 'training_mask')
        if not training.any():
            raise ValueError('training_mask selects no anchors')
        if not np.isfinite(raw[training, :8]).all():
            raise ValueError('training non-NCR geometry contains NaN/Inf')
        train_valid_ncr = training & valid & np.isfinite(raw[:, 8])
        if not train_valid_ncr.any():
            raise ValueError('training fold contains no valid NCR3D anchors')
        ncr_median = float(np.median(raw[train_valid_ncr, 8]))
        mean = np.empty(JOINT_GEOMETRY_DIM, dtype=np.float32)
        std = np.empty(JOINT_GEOMETRY_DIM, dtype=np.float32)
        mean[:8] = raw[training, :8].mean(axis=0)
        std[:8] = raw[training, :8].std(axis=0)
        mean[8] = raw[train_valid_ncr, 8].mean()
        std[8] = raw[train_valid_ncr, 8].std()
        std = np.maximum(std, np.float32(min_std))
        return cls(mean=mean, std=std, ncr_median=ncr_median, training_anchor_count=int(training.sum()), training_valid_ncr_count=int(train_valid_ncr.sum()))

    def transform_direct(self, raw_geometry: np.ndarray, valid_ncr: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_geometry, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != JOINT_GEOMETRY_DIM:
            raise ValueError(f'raw_geometry must have shape [N, {JOINT_GEOMETRY_DIM}], got {raw.shape}')
        valid = _boolean_vector(valid_ncr, len(raw), 'valid_ncr')
        if not np.isfinite(raw[:, :8]).all():
            raise ValueError('non-NCR geometry contains NaN/Inf')
        if np.any(valid & ~np.isfinite(raw[:, 8])):
            raise ValueError('valid NCR3D entries must be finite')
        imputed = raw.copy()
        imputed[~valid, 8] = self.ncr_median
        standardized = (imputed - self.mean) / self.std
        if not np.isfinite(standardized).all():
            raise ValueError('standardized direct geometry contains NaN/Inf')
        return standardized.astype(np.float32)

    def residual(self, raw_geometry: np.ndarray, valid_ncr: np.ndarray, standardized_prediction: np.ndarray) -> np.ndarray:
        direct = self.transform_direct(raw_geometry, valid_ncr)
        prediction = _float_matrix(standardized_prediction, JOINT_GEOMETRY_DIM, 'standardized_prediction')
        if prediction.shape != direct.shape:
            raise ValueError('standardized_prediction row count does not match geometry')
        residual = (direct - prediction).astype(np.float32)
        valid = np.asarray(valid_ncr, dtype=bool)
        residual[~valid, 8] = np.float32(0.0)
        if not np.isfinite(residual).all():
            raise ValueError('residual geometry contains NaN/Inf')
        return residual

    def as_dict(self) -> dict[str, object]:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(), 'ncr_median': self.ncr_median, 'training_anchor_count': self.training_anchor_count, 'training_valid_ncr_count': self.training_valid_ncr_count}

def split_object_tails(joint_geometry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joint = _float_matrix(joint_geometry, JOINT_GEOMETRY_DIM, 'joint_geometry')
    nucleus = np.concatenate((joint[:, :4], joint[:, 8:9]), axis=1)
    cell = np.concatenate((joint[:, 4:8], joint[:, 8:9]), axis=1)
    return (nucleus.astype(np.float32), cell.astype(np.float32))

def build_token_pair(rays36: np.ndarray, arm: GeometryArm, *, direct3d: np.ndarray | None=None, residual3d: np.ndarray | None=None, shuffled_direct3d: np.ndarray | None=None, shuffled_residual3d: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    if arm not in GEOMETRY_ARMS:
        raise ValueError(f'unknown geometry arm: {arm!r}')
    rays = _float_matrix(rays36, RAY_DIM, 'rays36')
    provided = {'direct3d': direct3d, 'residual3d': residual3d, 'shuffled_direct3d': shuffled_direct3d, 'shuffled_residual3d': shuffled_residual3d}
    expected = None if arm == 'mask2d' else arm
    supplied = [name for name, value in provided.items() if value is not None]
    if expected is None:
        if supplied:
            raise ValueError('mask2d tokens must not receive 3D geometry')
        nucleus_tail = np.zeros((len(rays), TAIL_DIM), dtype=np.float32)
        cell_tail = np.zeros_like(nucleus_tail)
    else:
        if supplied != [expected]:
            raise ValueError(f"arm {arm!r} requires only {expected}; received {supplied or 'none'}")
        geometry = _float_matrix(provided[expected], JOINT_GEOMETRY_DIM, expected)
        if len(geometry) != len(rays):
            raise ValueError('geometry and rays must contain the same anchors')
        nucleus_tail, cell_tail = split_object_tails(geometry)
    nucleus = np.concatenate((rays, nucleus_tail), axis=1).astype(np.float32)
    cell = np.concatenate((rays, cell_tail), axis=1).astype(np.float32)
    if nucleus.shape[1] != TOKEN_DIM or cell.shape != nucleus.shape:
        raise AssertionError('BRACS token schema changed')
    if not np.array_equal(nucleus[:, :RAY_DIM], cell[:, :RAY_DIM]):
        raise AssertionError('nucleus/cell mask-ray prefixes differ')
    if not np.array_equal(nucleus[:, -1], cell[:, -1]):
        raise AssertionError('nucleus/cell shared NCR channels differ')
    return (nucleus, cell)
