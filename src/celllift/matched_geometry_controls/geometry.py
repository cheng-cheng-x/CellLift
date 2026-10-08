from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
import numpy as np

def p4_constants() -> tuple[float, np.ndarray]:
    p = 4
    volume = 2.0 * math.pi * p / (p + 1)
    covariance = np.asarray((p / (2 * (2 * p + 1)),) * 2 + ((p + 1) / (3 * (p + 3)),), np.float64)
    return (volume, covariance)

def body_statistics_numpy(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(transform, np.float64)
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError('transform must have shape [N,3,3]')
    volume_constant, covariance_constant = p4_constants()
    volume = volume_constant * np.abs(np.linalg.det(matrix))
    covariance = matrix * covariance_constant[None, None, :] @ np.swapaxes(matrix, -1, -2)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    axes = np.sqrt(np.maximum(eigenvalue[:, ::-1], np.finfo(np.float64).tiny))
    long_z2 = np.square(eigenvector[:, 2, -1])
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0) or (not np.all(np.isfinite(axes))):
        raise ValueError('invalid p=4 body')
    return (volume, axes, long_z2)

def selected_scene_descriptor(nucleus_transform: np.ndarray, cell_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nv, na, no = body_statistics_numpy(nucleus_transform)
    cv, ca, co = body_statistics_numpy(cell_transform)
    if len(nv) != len(cv):
        raise ValueError('nucleus/cell selected-scene row counts differ')

    def object_features(volume: np.ndarray, axes: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        return np.column_stack((np.log(volume), np.log(axes[:, 0] / axes[:, 1]), np.log(axes[:, 1] / axes[:, 2]), orientation))
    valid = np.isfinite(nv) & np.isfinite(cv) & (cv > nv) & (nv > 0)
    ncr = np.full(len(nv), np.nan, np.float64)
    ncr[valid] = np.log(nv[valid] / (cv[valid] - nv[valid]))
    raw = np.column_stack((object_features(nv, na, no), object_features(cv, ca, co), ncr)).astype(np.float32)
    if not np.all(np.isfinite(raw[:, :8])):
        raise ValueError('non-NCR direct geometry contains NaN/Inf')
    return (raw, valid)

def selected_scene_descriptor_torch(nucleus_body, cell_body):
    import torch
    volume_constant, covariance_constant = p4_constants()

    def stats(body):
        transform = body.transform
        cp = transform.new_tensor(covariance_constant)
        volume = volume_constant * torch.linalg.det(transform).abs()
        covariance = transform * cp @ transform.transpose(-1, -2)
        eigenvalue, eigenvector = torch.linalg.eigh(covariance)
        axes = eigenvalue.clamp_min(torch.finfo(eigenvalue.dtype).tiny).sqrt().flip(-1)
        orientation = eigenvector[..., 2, -1].square()
        features = torch.stack((volume.log(), (axes[:, 0] / axes[:, 1]).log(), (axes[:, 1] / axes[:, 2]).log(), orientation), -1)
        return (volume, features)
    nv, nf = stats(nucleus_body)
    cv, cf = stats(cell_body)
    valid = torch.isfinite(nv) & torch.isfinite(cv) & (nv > 0) & (cv > nv)
    ncr = torch.zeros_like(nv)
    ncr[valid] = (nv[valid] / (cv[valid] - nv[valid])).log()
    return (torch.cat((nf, cf, ncr[:, None]), -1), valid)
