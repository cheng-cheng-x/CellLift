from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.geometry_baselines.models import DualGeometryExpert
from celllift.matched_geometry_controls.tokens import TOKEN_DIM, branch_tails
from .config import DROPOUT, GEOM_EMBED, MASK_DIM, THREE_D_DIM
from .scale import NodePooledScale

def _as_f32(value, cols: int) -> np.ndarray:
    array = np.asarray(value, np.float32)
    if array.ndim != 2 or array.shape[1] != cols:
        if array.size == 0:
            return np.zeros((0, cols), np.float32)
        raise ValueError(f'expected [N,{cols}], got {array.shape}')
    return array

def _as_bool(value, n: int) -> np.ndarray:
    array = np.asarray(value, bool).reshape(-1)
    if array.shape != (n,):
        raise ValueError(f'mask length {array.shape} != {n}')
    return array

def truncate_by_nucleus_id(nucleus_id: np.ndarray, cap: int | None) -> np.ndarray:
    ids = np.asarray(nucleus_id).reshape(-1)
    if cap is None or len(ids) <= int(cap):
        return np.ones(len(ids), bool)
    order = np.argsort(ids, kind='stable')[:int(cap)]
    keep = np.zeros(len(ids), bool)
    keep[order] = True
    return keep

def sanitize_finite(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.size == 0:
        return value
    return np.where(np.isfinite(value), value, 0).astype(value.dtype, copy=False)

def geometry9(node3d: np.ndarray) -> np.ndarray:
    value = np.asarray(node3d, np.float32)
    if value.size == 0:
        return np.zeros((0, 9), np.float32)
    if value.ndim != 2 or value.shape[1] < 9:
        raise ValueError(f'node3d must be [N,>=9], got {value.shape}')
    return value[:, :9]

def isolate_tokens(rays: np.ndarray, node3d: np.ndarray, include: np.ndarray, valid3d: np.ndarray, mode: str, residual9: np.ndarray | None=None, nucleus_id: np.ndarray | None=None, node_cap: int | None=None, scale: NodePooledScale | None=None) -> dict[str, np.ndarray]:
    if scale is None:
        raise RuntimeError('G/H tokens require a FIT node-pooled scale')
    raw_rays = _as_f32(rays, MASK_DIM)
    n = len(raw_rays)
    include = _as_bool(include, n) if n else np.zeros(0, bool)
    valid3d = _as_bool(valid3d, n) if n else np.zeros(0, bool)
    raw_geom = geometry9(node3d)
    if len(raw_geom) != n:
        raise ValueError('rays/node3d row mismatch')
    if residual9 is not None:
        residual9 = _as_f32(residual9, 9)
        if len(residual9) != n:
            raise ValueError('residual/nucleus mismatch')
    if nucleus_id is None:
        nucleus_id = np.arange(n, dtype=np.int64)
    keep = truncate_by_nucleus_id(nucleus_id, node_cap)
    if keep.size:
        raw_rays = raw_rays[keep]
        raw_geom = raw_geom[keep]
        include = include[keep]
        valid3d = valid3d[keep]
        if residual9 is not None:
            residual9 = residual9[keep]
        n = int(keep.sum())
    if n == 0:
        z = np.zeros((1, TOKEN_DIM), np.float32)
        m = np.zeros(1, bool)
        return {'nucleus': z, 'nucleus_mask': m, 'cell': z.copy(), 'cell_mask': m.copy(), 'truncated': int(keep.size and (not keep.all()))}
    finite3d = np.isfinite(raw_geom).all(axis=1)
    valid = include & valid3d & finite3d
    rays = scale.transform_rays(raw_rays, include)
    geom = scale.transform_geometry(raw_geom, valid)
    raw_nuc, raw_cell = branch_tails(geom)
    if residual9 is not None:
        residual_rows = np.where(np.isfinite(residual9), residual9, 0).astype(np.float32)
        residual_rows[~valid] = 0.0
        res_nuc, res_cell = branch_tails(residual_rows)
    else:
        res_nuc = res_cell = None
    zeros36 = np.zeros((n, MASK_DIM), np.float32)
    zeros5 = np.zeros((n, THREE_D_DIM), np.float32)
    empty_cell = True
    if mode == 'g2':
        nucleus = np.concatenate((rays, zeros5), 1)
        cell = np.concatenate((zeros36, zeros5), 1)
        cell_mask = np.zeros(n, bool)
    elif mode == 'g3':
        nucleus = np.concatenate((zeros36, raw_nuc), 1)
        cell = np.concatenate((zeros36, raw_cell), 1)
        cell_mask = include.copy()
        empty_cell = False
    elif mode == 'gr':
        if res_nuc is None:
            raise ValueError('GR requires residual9')
        nucleus = np.concatenate((zeros36, res_nuc), 1)
        cell = np.concatenate((zeros36, res_cell), 1)
        cell_mask = include.copy()
        empty_cell = False
    elif mode == 'g23':
        nucleus = np.concatenate((rays, raw_nuc), 1)
        cell = np.concatenate((zeros36, raw_cell), 1)
        cell_mask = include.copy()
        empty_cell = False
    elif mode == 'g2r':
        if res_nuc is None:
            raise ValueError('G2R requires residual9')
        nucleus = np.concatenate((rays, res_nuc), 1)
        cell = np.concatenate((zeros36, res_cell), 1)
        cell_mask = include.copy()
        empty_cell = False
    else:
        raise ValueError(mode)
    return {'nucleus': nucleus.astype(np.float32), 'nucleus_mask': include.copy(), 'cell': cell.astype(np.float32), 'cell_mask': cell_mask, 'empty_cell': empty_cell, 'truncated': int(keep.size and (not keep.all()))}

def pad_sets(rows: Sequence[dict[str, np.ndarray]], device: torch.device) -> dict[str, Tensor]:
    count = max(1, len(rows))
    width = max((len(row['nucleus']) for row in rows), default=1)
    nucleus = torch.zeros(count, width, TOKEN_DIM, device=device)
    cell = torch.zeros(count, width, TOKEN_DIM, device=device)
    nucleus_mask = torch.zeros(count, width, dtype=torch.bool, device=device)
    cell_mask = torch.zeros(count, width, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        n = len(row['nucleus'])
        nucleus[index, :n] = torch.as_tensor(row['nucleus'], device=device)
        cell[index, :n] = torch.as_tensor(row['cell'], device=device)
        nucleus_mask[index, :n] = torch.as_tensor(row['nucleus_mask'], device=device)
        cell_mask[index, :n] = torch.as_tensor(row['cell_mask'], device=device)
    return {'nucleus_tokens': nucleus, 'nucleus_mask': nucleus_mask, 'cell_tokens': cell, 'cell_mask': cell_mask}

class DeepSetsTile(nn.Module):

    def __init__(self, dropout: float=DROPOUT):
        super().__init__()
        self.expert = DualGeometryExpert('deepsets', dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(128, GEOM_EMBED), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, packed: dict[str, Tensor]) -> Tensor:
        return self.proj(self.expert(**packed))

class DualStreamTile(nn.Module):

    def __init__(self, dropout: float=DROPOUT):
        super().__init__()
        self.stream_2d = DualGeometryExpert('deepsets', dropout=dropout)
        self.stream_3d = DualGeometryExpert('deepsets', dropout=dropout)
        self.proj = nn.Sequential(nn.Linear(256, GEOM_EMBED), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, packed_2d: dict[str, Tensor], packed_3d: dict[str, Tensor]) -> Tensor:
        left = self.stream_2d(**packed_2d)
        right = self.stream_3d(**packed_3d)
        return self.proj(torch.cat((left, right), dim=-1))
