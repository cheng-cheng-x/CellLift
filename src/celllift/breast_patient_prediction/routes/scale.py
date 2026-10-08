from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NodePooledScale:
    ray_mean: np.ndarray
    ray_std: np.ndarray
    geom_mean: np.ndarray
    geom_std: np.ndarray

    def as_dict(self) -> dict:
        return {'ray_mean': self.ray_mean.tolist(), 'ray_std': self.ray_std.tolist(), 'geom_mean': self.geom_mean.tolist(), 'geom_std': self.geom_std.tolist(), 'estimator': 'fit_node_pooled'}

    @classmethod
    def from_dict(cls, payload: dict) -> 'NodePooledScale':
        return cls(ray_mean=np.asarray(payload['ray_mean'], np.float32), ray_std=np.asarray(payload['ray_std'], np.float32), geom_mean=np.asarray(payload['geom_mean'], np.float32), geom_std=np.asarray(payload['geom_std'], np.float32))

    def transform_rays(self, rays: np.ndarray, include: np.ndarray) -> np.ndarray:
        out = (np.asarray(rays, np.float32) - self.ray_mean) / self.ray_std
        out = np.where(np.isfinite(out), out, 0).astype(np.float32)
        out[~np.asarray(include, bool)] = 0.0
        return out

    def transform_geometry(self, geometry9: np.ndarray, valid: np.ndarray) -> np.ndarray:
        raw = np.asarray(geometry9, np.float32)
        out = (raw - self.geom_mean) / self.geom_std
        keep = np.asarray(valid, bool) & np.isfinite(raw).all(axis=1) & np.isfinite(out).all(axis=1)
        out = np.where(keep[:, None], out, 0).astype(np.float32)
        return out

def fit_node_pooled_scale(rays: np.ndarray, geometry9: np.ndarray, include: np.ndarray, valid: np.ndarray) -> NodePooledScale:
    rays = np.asarray(rays, np.float32)
    geom = np.asarray(geometry9, np.float32)
    include = np.asarray(include, bool).reshape(-1)
    valid = np.asarray(valid, bool).reshape(-1) & include & np.isfinite(geom).all(axis=1)
    ray_ok = include & np.isfinite(rays).all(axis=1)
    if not ray_ok.any():
        raise RuntimeError('ray scale rows are empty')
    if not valid.any():
        raise RuntimeError('geometry scale rows are empty')
    ray_mean = rays[ray_ok].mean(0).astype(np.float32)
    ray_std = np.maximum(rays[ray_ok].std(0), 1e-06).astype(np.float32)
    geom_mean = geom[valid].mean(0).astype(np.float32)
    geom_std = np.maximum(geom[valid].std(0), 1e-06).astype(np.float32)
    return NodePooledScale(ray_mean, ray_std, geom_mean, geom_std)

def select_best_epoch(losses: list[float]) -> int:
    if not losses:
        raise ValueError('empty loss history')
    best_index = 0
    best = float(losses[0])
    for index, value in enumerate(losses):
        if float(value) < best - 1e-07:
            best = float(value)
            best_index = index
    return best_index + 1

def assert_nucleus_identity(left: np.ndarray, right: np.ndarray, graph_id: str='') -> None:
    got = np.asarray(left).reshape(-1)
    expected = np.asarray(right).reshape(-1)
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise RuntimeError(f'residual nucleus_id mismatch for {graph_id}: {got.shape} vs {expected.shape}')
