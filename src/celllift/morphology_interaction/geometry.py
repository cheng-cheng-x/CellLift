from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
import numpy as np
RAY_DIM = 36
DISTANCE_SCALE_UM = 10.0
EPS = 1e-12

def p4_constants() -> tuple[float, np.ndarray]:
    p = 4
    volume = 2.0 * math.pi * p / (p + 1)
    covariance = np.asarray((p / (2 * (2 * p + 1)),) * 2 + ((p + 1) / (3 * (p + 3)),), np.float64)
    return (volume, covariance)

def body_statistics_torch(transform):
    import torch
    volume_constant, covariance_constant = p4_constants()
    if transform.ndim != 3 or transform.shape[1:] != (3, 3):
        raise ValueError('transform must have shape [N,3,3]')
    value = transform.to(torch.float64)
    constant = value.new_tensor(covariance_constant)
    volume = volume_constant * torch.linalg.det(value).abs()
    covariance = value * constant[None, None, :] @ value.transpose(-1, -2)
    eigenvalue, eigenvector = torch.linalg.eigh(covariance)
    axes = eigenvalue.clamp_min(torch.finfo(eigenvalue.dtype).tiny).sqrt().flip(-1)
    long_z2 = eigenvector[..., 2, -1].square()
    return (volume, axes, long_z2, covariance)

def body_features_torch(nucleus_transform, cell_transform):
    import torch
    nv, na, no, ncov = body_statistics_torch(nucleus_transform)
    cv, ca, co, ccov = body_statistics_torch(cell_transform)

    def object_features(volume, axes, orientation):
        flat = (axes[:, 0] / axes[:, 1]).clamp_min(1e-12)
        second = (axes[:, 1] / axes[:, 2]).clamp_min(1e-12)
        return torch.stack((volume.clamp_min(1e-12).log(), flat.log(), second.log(), orientation), -1)
    valid = torch.isfinite(nv) & torch.isfinite(cv) & (nv > 0) & (cv > nv)
    ncr = torch.full_like(nv, float('nan'))
    ncr[valid] = (nv[valid] / (cv[valid] - nv[valid])).log()
    direct = torch.cat((object_features(nv, na, no), object_features(cv, ca, co), ncr[:, None]), -1)
    return (direct, valid, nv, cv, ncov, ccov, na, ca)

def equivalent_radius(volume):
    import torch
    return (3.0 * volume.clamp_min(1e-12) / (4.0 * math.pi)) ** (1.0 / 3.0)

def ray_angles(count: int=RAY_DIM) -> np.ndarray:
    return np.arange(count, dtype=np.float64) * (2.0 * math.pi / count)

def ray_contour_stats(rays: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rays = np.asarray(rays, np.float64)
    if rays.ndim != 2 or rays.shape[1] != RAY_DIM:
        raise ValueError(f'rays must be [N,{RAY_DIM}]')
    angles = ray_angles(RAY_DIM)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    tips = np.stack((rays * cosine, rays * sine), axis=-1)
    rolled = np.roll(rays, -1, axis=1)
    area = 0.5 * (rays * rolled * math.sin(2.0 * math.pi / RAY_DIM)).sum(axis=1)
    area = np.maximum(area, EPS)
    centered = tips - tips.mean(axis=1, keepdims=True)
    covariance = np.matmul(centered.transpose(0, 2, 1), centered) / float(RAY_DIM)
    covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
    covariance = covariance + (EPS * np.eye(2, dtype=np.float64))[None, :, :]
    evals = np.linalg.eigvalsh(covariance)
    major = np.maximum(evals[:, 1], EPS)
    minor = np.maximum(evals[:, 0], EPS)
    axis_ratio = major / minor
    node2d = np.stack((np.log(area), np.log(axis_ratio)), axis=1)
    return (node2d.astype(np.float32), covariance.astype(np.float32), area.astype(np.float32))

def direction_spacing(delta: np.ndarray, cov_i: np.ndarray, cov_j: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, np.float64)
    cov_i = np.asarray(cov_i, np.float64)
    cov_j = np.asarray(cov_j, np.float64)
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], EPS)
    scale_i = np.sqrt(np.einsum('ei,eij,ej->e', unit, cov_i, unit).clip(min=0.0) + EPS)
    scale_j = np.sqrt(np.einsum('ei,eij,ej->e', unit, cov_j, unit).clip(min=0.0) + EPS)
    return (distance / (scale_i + scale_j)).astype(np.float32)

def knn_edges(xy: np.ndarray, cap: float, k: int) -> np.ndarray:
    count = len(xy)
    if count < 2:
        return np.zeros((0, 2), np.int64)
    blocks = []
    xy = np.asarray(xy, np.float64)
    for start in range(0, count, 2048):
        block = xy[start:start + 2048]
        distance = np.linalg.norm(block[:, None, :] - xy[None, :, :], axis=-1)
        rows = np.arange(len(block))
        distance[rows, rows + start] = np.inf
        neighbours = min(k, count - 1)
        index = np.argpartition(distance, neighbours - 1, axis=1)[:, :neighbours]
        selected = np.take_along_axis(distance, index, axis=1)
        keep = selected <= cap
        source = np.repeat(np.arange(start, start + len(block)), neighbours).reshape(len(block), neighbours)
        blocks.append(np.stack((source[keep], index[keep]), axis=1))
    pair = np.concatenate(blocks, 0) if blocks else np.zeros((0, 2), np.int64)
    if not len(pair):
        return pair
    pair = np.concatenate((pair, pair[:, ::-1]), 0)
    order = np.lexsort((pair[:, 1], pair[:, 0]))
    pair = pair[order]
    keep = np.ones(len(pair), bool)
    keep[1:] = np.any(pair[1:] != pair[:-1], axis=1)
    return pair[keep].astype(np.int64)
