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
    if len(nv) != len(cv):
        raise ValueError('nucleus/cell selected-scene row counts differ')

    def object_features(volume, axes, orientation):
        flat = (axes[:, 0] / axes[:, 1]).clamp_min(1e-12)
        second = (axes[:, 1] / axes[:, 2]).clamp_min(1e-12)
        return torch.stack((volume.clamp_min(1e-12).log(), flat.log(), second.log(), orientation), -1)
    valid = torch.isfinite(nv) & torch.isfinite(cv) & (nv > 0) & (cv > nv)
    ncr = torch.full_like(nv, float('nan'))
    ncr[valid] = (nv[valid] / (cv[valid] - nv[valid])).log()
    direct = torch.cat((object_features(nv, na, no), object_features(cv, ca, co), ncr[:, None]), -1)
    return (direct, valid, nv, cv, ncov, ccov)

def equivalent_radius(volume):
    import torch
    return (3.0 * volume.clamp_min(1e-12) / (4.0 * math.pi)) ** (1.0 / 3.0)

def covariance_upper(covariance):
    import torch
    return torch.stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2], covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]), -1)
