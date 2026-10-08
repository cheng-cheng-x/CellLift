from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import math
from functools import lru_cache
import numpy as np

@dataclass(frozen=True)
class MomentBatch:
    instance_id: np.ndarray
    count: np.ndarray
    center_xy_um: np.ndarray
    covariance_xy_um2: np.ndarray
    root_xy_um: np.ndarray
    precision_xy_um2_inv: np.ndarray
    raw_area_um2: np.ndarray
    ellipse_area_um2: np.ndarray

@lru_cache(maxsize=4)
def _coordinate_terms(height: int, width: int):
    x = np.tile(np.arange(width, dtype=np.float64) + 0.5, height)
    y = np.repeat(np.arange(height, dtype=np.float64) + 0.5, width)
    return (x, y, x * x, y * y, x * y)

def _spd_root_2x2(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, np.float64)
    determinant_root = np.sqrt(np.linalg.det(covariance))
    denominator = np.sqrt(np.trace(covariance, axis1=-2, axis2=-1) + 2.0 * determinant_root)
    return 2.0 * (covariance + determinant_root[:, None, None] * np.eye(2)) / denominator[:, None, None]

def all_instance_moments(label_image: np.ndarray, mpp: float, instance_ids: np.ndarray | None=None) -> MomentBatch:
    labels = np.asarray(label_image)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('label_image must be a 2-D integer array')
    if not math.isfinite(mpp) or mpp <= 0:
        raise ValueError('mpp must be positive and finite')
    flat = labels.reshape(-1).astype(np.int64, copy=False)
    if np.any(flat < 0):
        raise ValueError('negative instance labels are unsupported')
    height, width = labels.shape
    x, y, x2, y2, xy = _coordinate_terms(height, width)
    length = int(flat.max(initial=0)) + 1
    count_all = np.bincount(flat, minlength=length).astype(np.int64)
    sx = np.bincount(flat, weights=x, minlength=length)
    sy = np.bincount(flat, weights=y, minlength=length)
    sxx = np.bincount(flat, weights=x2, minlength=length)
    syy = np.bincount(flat, weights=y2, minlength=length)
    sxy = np.bincount(flat, weights=xy, minlength=length)
    ids = np.flatnonzero(count_all[1:]) + 1 if instance_ids is None else np.asarray(instance_ids, np.int64)
    if ids.ndim != 1 or np.any(ids <= 0) or np.any(ids >= length) or np.any(count_all[ids] == 0):
        raise ValueError('instance_ids contains absent/non-positive labels')
    count = count_all[ids].astype(np.float64)
    mx, my = (sx[ids] / count, sy[ids] / count)
    cxx = sxx[ids] / count - mx * mx + 1.0 / 12.0
    cyy = syy[ids] / count - my * my + 1.0 / 12.0
    cxy = sxy[ids] / count - mx * my
    covariance_px = np.stack((cxx, cxy, cxy, cyy), axis=1).reshape(-1, 2, 2)
    root_px = _spd_root_2x2(covariance_px)
    precision_px = np.linalg.inv(4.0 * covariance_px)
    scale2 = float(mpp) ** 2
    return MomentBatch(instance_id=ids, count=count.astype(np.int64), center_xy_um=np.column_stack((mx, my)) * mpp, covariance_xy_um2=covariance_px * scale2, root_xy_um=root_px * mpp, precision_xy_um2_inv=precision_px / scale2, raw_area_um2=count * scale2, ellipse_area_um2=np.pi * np.linalg.det(root_px) * scale2)

def align_moments(batch: MomentBatch, nucleus_ids: np.ndarray) -> MomentBatch:
    wanted = np.asarray(nucleus_ids, np.int64)
    order = np.argsort(batch.instance_id)
    position = np.searchsorted(batch.instance_id[order], wanted)
    if np.any(position >= len(order)) or not np.array_equal(batch.instance_id[order[position]], wanted):
        raise ValueError('graph nucleus IDs do not match label-mask instances')
    take = order[position]
    return MomentBatch(**{name: getattr(batch, name)[take] for name in batch.__dataclass_fields__})
