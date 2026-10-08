from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import torch
from torch import Tensor

@dataclass(frozen=True)
class SupportKKTResult:
    support: Tensor
    minimum_level: Tensor
    has_interior: Tensor
    is_tangent: Tensor
    optimizer: Tensor
    multiplier: Tensor
    stationarity_error: Tensor
    constraint_residual: Tensor

def _signed_distance(value: Tensor, bounds: Tensor) -> Tensor:
    return value - value.clamp(min=bounds[:, 0, None], max=bounds[:, 1, None])

def _scalar_level_and_derivative(coordinate: Tensor, center_z: Tensor, slope_norm: Tensor, axial_scale: Tensor, bounds: Tensor, p: float) -> tuple[Tensor, Tensor]:
    signed = _signed_distance(center_z[:, None] + slope_norm[:, None] * coordinate, bounds)
    ratio = signed / axial_scale[:, None]
    axial_derivative = p * ratio.abs().pow(p - 1) * ratio.sign() / axial_scale[:, None]
    return (coordinate.square() + ratio.abs().pow(p), 2 * coordinate + axial_derivative * slope_norm[:, None])

def compatibility_projection_support_kkt(root_xy: Tensor, center_offset_xy: Tensor, center_z: Tensor, slope_z: Tensor, axial_scale: Tensor, bounds: Tensor, directions: Tensor, p: float, *, bisection_steps: int=64, maximization_steps: int=64, interior_tolerance: float | None=None) -> SupportKKTResult:
    batch = root_xy.shape[0]
    if root_xy.shape != (batch, 2, 2) or center_offset_xy.shape != (batch, 2):
        raise ValueError('invalid planar geometry shape')
    if slope_z.shape != (batch, 2) or center_z.shape != (batch,) or axial_scale.shape != (batch,):
        raise ValueError('invalid axial geometry shape')
    if bounds.shape != (batch, 2) or directions.ndim != 2 or directions.shape[-1] != 2:
        raise ValueError('bounds must be [B,2] and directions [K,2]')
    if p < 2:
        raise ValueError('this diagnostic requires p>=2')
    precision = torch.finfo(root_xy.dtype)
    tolerance = 64 * precision.eps if interior_tolerance is None else interior_tolerance
    with torch.no_grad():
        local_direction = torch.einsum('bij,kj->bki', root_xy.transpose(-1, -2), directions)
        norm = torch.linalg.vector_norm(slope_z, dim=-1)
        default_axis = torch.zeros_like(slope_z)
        default_axis[:, 0] = 1
        axis = torch.where((norm > 0)[:, None], slope_z / norm.clamp_min(precision.tiny)[:, None], default_axis)
        perpendicular = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
        alpha = (local_direction * axis[:, None]).sum(-1)
        beta_signed = (local_direction * perpendicular[:, None]).sum(-1)
        beta = beta_signed.abs()
        lower = center_z.new_full((batch, 1), -1)
        upper = center_z.new_full((batch, 1), 1)
        for _ in range(bisection_steps):
            middle = (lower + upper) * 0.5
            _, derivative = _scalar_level_and_derivative(middle, center_z, norm, axial_scale, bounds, p)
            lower = torch.where(derivative < 0, middle, lower)
            upper = torch.where(derivative < 0, upper, middle)
        minimum = (lower + upper) * 0.5
        minimum_level, _ = _scalar_level_and_derivative(minimum, center_z, norm, axial_scale, bounds, p)
        minimum_level = minimum_level[:, 0]
        has_interior = minimum_level < 1 - tolerance
        is_tangent = (minimum_level - 1).abs() <= tolerance
        left_outside = torch.full_like(minimum, -1)
        left_inside = minimum.clone()
        right_inside = minimum.clone()
        right_outside = torch.full_like(minimum, 1)
        for _ in range(bisection_steps):
            left_middle = (left_outside + left_inside) * 0.5
            left_level, _ = _scalar_level_and_derivative(left_middle, center_z, norm, axial_scale, bounds, p)
            left_outside = torch.where(left_level > 1, left_middle, left_outside)
            left_inside = torch.where(left_level > 1, left_inside, left_middle)
            right_middle = (right_inside + right_outside) * 0.5
            right_level, _ = _scalar_level_and_derivative(right_middle, center_z, norm, axial_scale, bounds, p)
            right_inside = torch.where(right_level <= 1, right_middle, right_inside)
            right_outside = torch.where(right_level <= 1, right_outside, right_middle)
        left = torch.where(has_interior[:, None], left_inside, minimum)
        right = torch.where(has_interior[:, None], right_inside, minimum)
        search_lower = left.expand_as(alpha).clone()
        search_upper = right.expand_as(alpha).clone()
        for _ in range(maximization_steps):
            middle = (search_lower + search_upper) * 0.5
            level, derivative = _scalar_level_and_derivative(middle, center_z, norm, axial_scale, bounds, p)
            width = (1 - level).clamp_min(0).sqrt()
            derivative_sign = 2 * alpha * width - beta * derivative
            search_lower = torch.where(derivative_sign > 0, middle, search_lower)
            search_upper = torch.where(derivative_sign > 0, search_upper, middle)
        coordinate = (search_lower + search_upper) * 0.5
        coordinate = torch.where(beta == 0, torch.where(alpha >= 0, right, left), coordinate)
        level, _ = _scalar_level_and_derivative(coordinate, center_z, norm, axial_scale, bounds, p)
        width = (1 - level).clamp_min(0).sqrt()
        optimizer = coordinate[..., None] * axis[:, None] + (beta_signed.sign() * width)[..., None] * perpendicular[:, None]
        signed = _signed_distance(center_z[:, None] + (optimizer * slope_z[:, None]).sum(-1), bounds)
        ratio = signed / axial_scale[:, None]
        derivative_z = p * ratio.abs().pow(p - 1) * ratio.sign() / axial_scale[:, None]
        constraint_gradient = 2 * optimizer + derivative_z[..., None] * slope_z[:, None]
        gradient_squared = constraint_gradient.square().sum(-1)
        multiplier = (local_direction * constraint_gradient).sum(-1) / gradient_squared.clamp_min(precision.tiny)
        multiplier = torch.where(has_interior[:, None], multiplier, torch.zeros_like(multiplier))
        stationarity_error = torch.linalg.vector_norm(local_direction - multiplier[..., None] * constraint_gradient, dim=-1) / torch.linalg.vector_norm(local_direction, dim=-1).clamp_min(precision.tiny)
        constraint_residual = (optimizer.square().sum(-1) + ratio.abs().pow(p) - 1).abs()
        minimum_optimizer = minimum * axis
    local_direction = torch.einsum('bij,kj->bki', root_xy.transpose(-1, -2), directions)
    objective = torch.einsum('bi,ki->bk', center_offset_xy, directions) + (local_direction * optimizer).sum(-1)
    signed = _signed_distance(center_z[:, None] + (optimizer * slope_z[:, None]).sum(-1), bounds)
    live_constraint = optimizer.square().sum(-1) + (signed.abs() / axial_scale[:, None]).pow(p) - 1
    corrected = objective - multiplier * (live_constraint - live_constraint.detach())
    support = torch.where(has_interior[:, None], corrected, torch.full_like(corrected, float('nan')))
    minimum_ridge = center_z + (minimum_optimizer * slope_z).sum(-1)
    minimum_signed = minimum_ridge - minimum_ridge.clamp(min=bounds[:, 0], max=bounds[:, 1])
    live_minimum = minimum_optimizer.square().sum(-1) + (minimum_signed.abs() / axial_scale).pow(p)
    return SupportKKTResult(support=support, minimum_level=live_minimum, has_interior=has_interior, is_tangent=is_tangent, optimizer=optimizer, multiplier=multiplier, stationarity_error=stationarity_error, constraint_residual=constraint_residual)

def projection_support_kkt(*args, **kwargs) -> tuple[Tensor, Tensor]:
    result = compatibility_projection_support_kkt(*args, **kwargs)
    return (result.support, result.minimum_level)
