from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import math
from celllift.runtime import torch
from torch import Tensor
from torch.nn import functional as F

def unit_circle(count: int=36, *, device=None, dtype=torch.float32) -> Tensor:
    angle = torch.arange(count, device=device, dtype=dtype) * (2.0 * math.pi / count)
    return torch.stack((angle.cos(), angle.sin()), -1)

def symmetric_root_2x2(matrix: Tensor) -> Tensor:
    tiny = torch.finfo(matrix.dtype).tiny
    determinant = torch.linalg.det(matrix).clamp_min(tiny).sqrt()
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(2, device=matrix.device, dtype=matrix.dtype)
    return (matrix + determinant[..., None, None] * identity) / (trace + 2.0 * determinant).clamp_min(tiny).sqrt()[..., None, None]

def root_from_precision(precision: Tensor) -> Tensor:
    return symmetric_root_2x2(torch.linalg.inv(precision))

def slab_bounds(plane: Tensor, *, dtype=None) -> Tensor:
    table = torch.tensor(((-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)), device=plane.device, dtype=dtype or torch.float32)
    return table.index_select(0, plane.long())

def _expand_directions(directions: Tensor, count: int) -> Tensor:
    if directions.ndim == 2:
        return directions[None].expand(count, -1, -1)
    if directions.ndim != 3 or directions.shape[0] != count:
        raise ValueError('directions must have shape [K,2] or [N,K,2]')
    return directions

def projection_support(root_xy: Tensor, center_xy: Tensor, c: Tensor, slope_xy: Tensor, kappa: Tensor, bounds: Tensor, directions: Tensor, p: float, *, steps: int=32) -> tuple[Tensor, Tensor, Tensor]:
    count = c.shape[0]
    direction = _expand_directions(directions, count)
    linear = torch.einsum('nki,ni->nk', direction, slope_xy)
    radial = torch.linalg.vector_norm(torch.einsum('nji,nkj->nki', root_xy, direction), dim=-1)
    low, high = (bounds[:, 0], bounds[:, 1])
    closest = torch.maximum(torch.minimum(c, high), low)
    min_level = kappa * (closest - c).abs().pow(float(p))
    nonempty = min_level < 1.0
    tiny = torch.finfo(c.dtype).tiny
    with torch.no_grad():
        radius = kappa.clamp_min(tiny).pow(-1.0 / float(p))
        ratio = linear.abs() * radius[:, None] / radial.clamp_min(tiny)
        from .fast_kernels import support_bisection
        optimum_sum = support_bisection(ratio, p, steps, tiny)
        relative = radius[:, None] * optimum_sum * 0.5 * linear.sign()
        optimum = (c[:, None] + relative).clamp(low[:, None], high[:, None])
        cylinder_optimum = torch.where(linear > 0, high[:, None], torch.where(linear < 0, low[:, None], closest[:, None]))
        optimum = torch.where(kappa[:, None] == 0, cylinder_optimum, optimum)
        optimum = torch.where(nonempty[:, None], optimum, closest[:, None])
    relative = optimum - c[:, None]
    remaining = 1.0 - kappa[:, None] * relative.abs().pow(float(p))
    smooth_radius = torch.where(nonempty[:, None], remaining.clamp_min(torch.finfo(c.dtype).eps ** 2).sqrt(), torch.zeros_like(remaining))
    ridge_reference = center_xy + slope_xy * (closest - c)[:, None]
    support = torch.einsum('nki,ni->nk', direction, ridge_reference) + linear * (optimum - closest[:, None]) + radial * smooth_radius
    return (support, nonempty, min_level)

def polygon_moments_from_support(support: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    count = support.shape[-1]
    if count < 8:
        raise ValueError('at least eight equally spaced support directions required')
    normals = unit_circle(count, device=support.device, dtype=support.dtype)
    tangent = torch.stack((-normals[:, 1], normals[:, 0]), -1)
    delta = 2.0 * math.pi / count
    along = (support.roll(-1, -1) - math.cos(delta) * support) / math.sin(delta)
    vertices = support[..., None] * normals + along[..., None] * tangent
    origin = vertices.mean(-2)
    left = vertices - origin[:, None]
    right = left.roll(-1, -2)
    cross = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    area = 0.5 * cross.sum(-1)
    denominator = area.clamp_min(torch.finfo(area.dtype).tiny)
    mean_local = (cross[..., None] * (left + right)).sum(-2) / (6.0 * denominator[:, None])
    second_sum = 2.0 * left[..., :, None] * left[..., None, :] + left[..., :, None] * right[..., None, :] + right[..., :, None] * left[..., None, :] + 2.0 * right[..., :, None] * right[..., None, :]
    second = (cross[..., None, None] * second_sum).sum(-3) / (24.0 * denominator[:, None, None])
    covariance = second - mean_local[..., :, None] * mean_local[..., None, :]
    return (area, mean_local + origin, covariance)

def projection_moments(root_xy: Tensor, center_xy: Tensor, c: Tensor, slope_xy: Tensor, kappa: Tensor, bounds: Tensor, p: float, *, directions: int=256, steps: int=32) -> tuple[Tensor, Tensor, Tensor]:
    normals = unit_circle(directions, device=c.device, dtype=c.dtype)
    closest = c.clamp(bounds[:, 0], bounds[:, 1])
    ridge_offset = slope_xy * (closest - c)[:, None]
    supports, nonempty, _ = projection_support(root_xy, -ridge_offset, c, slope_xy, kappa, bounds, normals, p, steps=steps)
    safe_supports = torch.where(nonempty[:, None], supports, torch.ones_like(supports))
    area, center, covariance = polygon_moments_from_support(safe_supports)
    center = center + center_xy + ridge_offset
    closest_center = center_xy + slope_xy * (closest - c)[:, None]
    return (torch.where(nonempty, area, torch.zeros_like(area)), torch.where(nonempty[:, None], center, closest_center), torch.where(nonempty[:, None, None], covariance, torch.zeros_like(covariance)))

def projection_area(*args, **kwargs) -> Tensor:
    return projection_moments(*args, **kwargs)[0]

def projected_level(points: Tensor, root_xy: Tensor, center_xy: Tensor, c: Tensor, slope_xy: Tensor, kappa: Tensor, bounds: Tensor, p: float, *, steps: int=32) -> Tensor:
    inverse = torch.linalg.inv(root_xy)
    local = torch.einsum('nij,nkj->nki', inverse, points - center_xy[:, None])
    beta = torch.einsum('nij,nj->ni', inverse, slope_xy)
    beta_square = beta.square().sum(-1)[:, None]
    dot = torch.einsum('nki,ni->nk', local, beta)
    with torch.no_grad():
        lower = bounds[:, 0, None].expand_as(dot)
        upper = bounds[:, 1, None].expand_as(dot)
        for _ in range(steps):
            middle = (lower + upper) * 0.5
            relative = middle - c[:, None]
            derivative = -2.0 * dot + 2.0 * beta_square * relative + float(p) * kappa[:, None] * relative.abs().pow(float(p) - 1.0) * relative.sign()
            lower = torch.where(derivative < 0, middle, lower)
            upper = torch.where(derivative < 0, upper, middle)
        optimum = (lower + upper) * 0.5
    relative = optimum - c[:, None]
    residual = local - beta[:, None] * relative[..., None]
    return residual.square().sum(-1) + kappa[:, None] * relative.abs().pow(float(p))

@dataclass(frozen=True)
class ShearedGeometry:
    beta: Tensor
    c: Tensor
    kappa: Tensor
    normalizer_center: Tensor
    normalizer_root: Tensor
    root_xy: Tensor
    center_xy: Tensor
    slope_xy: Tensor
    center: Tensor
    transform: Tensor
    h: Tensor

def build_from_parameters(anchor_center_xy: Tensor, anchor_root_xy: Tensor, beta: Tensor, c: Tensor, kappa: Tensor, p: float, *, slab_height_um: float=5.0, moment_directions: int=2048, support_steps: int=32) -> ShearedGeometry:
    count = c.shape[0]
    identity = torch.eye(2, device=c.device, dtype=c.dtype)[None].expand(count, -1, -1)
    zero = torch.zeros_like(beta)
    middle_bounds = torch.stack((torch.zeros_like(c), torch.ones_like(c)), -1)
    _, mu, covariance = projection_moments(identity, zero, c, beta, kappa, middle_bounds, p, directions=moment_directions, steps=support_steps)
    normalizer = symmetric_root_2x2(4.0 * covariance)
    root = anchor_root_xy @ torch.linalg.inv(normalizer)
    center_xy = anchor_center_xy - torch.einsum('nij,nj->ni', root, mu)
    slope_xy = torch.einsum('nij,nj->ni', root, beta)
    axial_radius = kappa.pow(-1.0 / float(p))
    h = float(slab_height_um) * axial_radius
    center = torch.cat((center_xy, (float(slab_height_um) * c)[:, None]), -1)
    transform = root.new_zeros((count, 3, 3))
    transform[:, :2, :2] = root
    transform[:, :2, 2] = slope_xy * axial_radius[:, None]
    transform[:, 2, 2] = h
    return ShearedGeometry(beta, c, kappa, mu, normalizer, root, center_xy, slope_xy, center, transform, h)

def build_sheared_geometry(anchor_center_xy: Tensor, anchor_root_xy: Tensor, raw: Tensor, p: float, *, slab_height_um: float=5.0, allow_tilt: bool=True, moment_directions: int=2048, support_steps: int=32) -> ShearedGeometry:
    expected = 4 if allow_tilt else 2
    if raw.shape != (anchor_center_xy.shape[0], expected):
        raise ValueError(f'raw must have shape [N,{expected}]')
    beta = raw[:, :2] if allow_tilt else raw.new_zeros((raw.shape[0], 2))
    c_raw = raw[:, -2]
    kappa = F.softplus(raw[:, -1]) + torch.finfo(raw.dtype).tiny
    radius = kappa.pow(-1.0 / float(p))
    interior = 1.0 - 4.0 * torch.finfo(raw.dtype).eps
    c = 0.5 + (0.5 + radius) * torch.tanh(c_raw) * interior
    return build_from_parameters(anchor_center_xy, anchor_root_xy, beta, c, kappa, p, slab_height_um=slab_height_um, moment_directions=moment_directions, support_steps=support_steps)

def raw_from_physical(beta: Tensor, c: Tensor, kappa: Tensor, p: float, *, allow_tilt: bool=True) -> Tensor:
    radius = kappa.pow(-1.0 / float(p))
    interior = 1.0 - 4.0 * torch.finfo(c.dtype).eps
    normalized = (c - 0.5) / ((0.5 + radius) * interior)
    c_raw = torch.atanh(normalized)
    kappa_raw = kappa + torch.log(-torch.expm1(-kappa))
    axial = torch.stack((c_raw, kappa_raw), -1)
    return torch.cat((beta, axial), -1) if allow_tilt else axial
