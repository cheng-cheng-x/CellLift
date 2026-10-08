from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from typing import Sequence
import numpy as np
_EPS = np.finfo(np.float64).eps

@dataclass(frozen=True)
class Morphology2D:
    log_area: np.ndarray
    log_equivalent_radius: np.ndarray
    log_axis_ratio: np.ndarray
    circularity: np.ndarray
    cos_2theta: np.ndarray
    sin_2theta: np.ndarray
    area: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    perimeter: np.ndarray
    valid: np.ndarray

    def channels(self) -> np.ndarray:
        return np.column_stack((self.log_area, self.log_equivalent_radius, self.log_axis_ratio, self.circularity, self.cos_2theta, self.sin_2theta)).astype(np.float32, copy=False)

@dataclass(frozen=True)
class Morphology3D:
    log_volume: np.ndarray
    log_ab: np.ndarray
    log_bc: np.ndarray
    long_axis_z_squared: np.ndarray
    volume: np.ndarray
    valid: np.ndarray

    def channels(self) -> np.ndarray:
        return np.column_stack((self.log_volume, self.log_ab, self.log_bc, self.long_axis_z_squared)).astype(np.float32, copy=False)

def _validate_ellipsoids(centers: np.ndarray, axes: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers[None, :]
    if axes.ndim == 1:
        axes = axes[None, :]
    if rotations.ndim == 2:
        rotations = rotations[None, :, :]
    count = centers.shape[0]
    if centers.shape != (count, 3) or axes.shape != (count, 3) or rotations.shape != (count, 3, 3):
        raise ValueError('centers/axes/rotations must have shapes [N,3], [N,3], [N,3,3]')
    if not (np.isfinite(centers).all() and np.isfinite(axes).all() and np.isfinite(rotations).all()):
        raise ValueError('ellipsoid parameters must be finite')
    if np.any(axes <= 0):
        raise ValueError('ellipsoid axes must be positive')
    gram = np.einsum('nji,njk->nik', rotations, rotations)
    if not np.allclose(gram, np.eye(3)[None], rtol=1e-05, atol=1e-06):
        raise ValueError('rotation matrices must be orthonormal (principal axes are columns)')
    return (centers, axes, rotations)

def ellipsoid_covariance(axes: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    dummy = np.zeros((np.asarray(axes).reshape(-1, 3).shape[0], 3), dtype=np.float64)
    _, axes, rotations = _validate_ellipsoids(dummy, axes, rotations)
    return np.einsum('nij,nj,nkj->nik', rotations, np.square(axes), rotations)

def finite_slab_projection_polygons(centers: np.ndarray, axes: np.ndarray, rotations: np.ndarray, *, slab_bounds_um: tuple[float, float]=(-2.5, 2.5), directions: int=64, chunk_size: int=4096) -> tuple[np.ndarray, np.ndarray]:
    centers, axes, rotations = _validate_ellipsoids(centers, axes, rotations)
    if directions < 16 or directions % 2:
        raise ValueError('directions must be an even integer >= 16')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    lower, upper = map(float, slab_bounds_um)
    if not np.isfinite([lower, upper]).all() or not lower < upper:
        raise ValueError('slab_bounds_um must be finite and increasing')
    sigma = np.einsum('nij,nj,nkj->nik', rotations, np.square(axes), rotations)
    sigma_zz = sigma[:, 2, 2]
    z_radius = np.sqrt(sigma_zz)
    valid = (centers[:, 2] + z_radius >= lower) & (centers[:, 2] - z_radius <= upper)
    angles = np.arange(directions, dtype=np.float64) * (2.0 * np.pi / directions)
    unit_xy = np.column_stack((np.cos(angles), np.sin(angles)))
    linear = np.column_stack((unit_xy, np.zeros(directions, dtype=np.float64)))
    polygon = np.full((len(centers), directions, 2), np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    for start in range(0, len(valid_indices), chunk_size):
        selected = valid_indices[start:start + chunk_size]
        current_sigma = sigma[selected]
        current_centers = centers[selected]
        sigma_l = np.einsum('dj,bkj->bdk', linear, current_sigma)
        denominator = np.sqrt(np.einsum('dj,bdj->bd', linear, sigma_l)).clip(min=np.sqrt(_EPS))
        full_points = current_centers[:, None, :] + sigma_l / denominator[:, :, None]
        z_star = full_points[:, :, 2]
        inside = (z_star >= lower) & (z_star <= upper)
        b = current_sigma[:, :2, 2]
        d = current_sigma[:, 2, 2]
        conditional = current_sigma[:, :2, :2] - np.einsum('bi,bj->bij', b, b) / d[:, None, None]
        z_face = np.where(z_star < lower, lower, upper)
        dz = z_face - current_centers[:, None, 2]
        scale = np.sqrt(np.maximum(0.0, 1.0 - np.square(dz) / d[:, None]))
        cross_center = current_centers[:, None, :2] + b[:, None, :] * (dz / d[:, None])[:, :, None]
        conditional_unit = np.einsum('bij,nj->bni', conditional, unit_xy)
        support_scale = np.sqrt(np.maximum(np.einsum('ni,bni->bn', unit_xy, conditional_unit), _EPS))
        constrained = cross_center + scale[:, :, None] * conditional_unit / support_scale[:, :, None]
        polygon[selected] = np.where(inside[:, :, None], full_points[:, :, :2], constrained)
    return (polygon, valid)

def _polygon_measurements(vertices: np.ndarray) -> tuple[float, float, float, float, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or (not np.isfinite(vertices).all()):
        return (np.nan,) * 5
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    signed_twice_area = float(cross.sum())
    if abs(signed_twice_area) <= _EPS:
        return (np.nan,) * 5
    if signed_twice_area < 0:
        vertices = vertices[::-1]
        following = np.roll(vertices, -1, axis=0)
        cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
        signed_twice_area = float(cross.sum())
    area = signed_twice_area / 2.0
    perimeter = float(np.linalg.norm(following - vertices, axis=1).sum())
    cx = float(((vertices[:, 0] + following[:, 0]) * cross).sum() / (3.0 * signed_twice_area))
    cy = float(((vertices[:, 1] + following[:, 1]) * cross).sum() / (3.0 * signed_twice_area))
    ex2 = float(((vertices[:, 0] ** 2 + vertices[:, 0] * following[:, 0] + following[:, 0] ** 2) * cross).sum() / (6.0 * signed_twice_area))
    ey2 = float(((vertices[:, 1] ** 2 + vertices[:, 1] * following[:, 1] + following[:, 1] ** 2) * cross).sum() / (6.0 * signed_twice_area))
    exy = float(((2.0 * vertices[:, 0] * vertices[:, 1] + vertices[:, 0] * following[:, 1] + following[:, 0] * vertices[:, 1] + 2.0 * following[:, 0] * following[:, 1]) * cross).sum() / (12.0 * signed_twice_area))
    covariance = np.asarray([[ex2 - cx * cx, exy - cx * cy], [exy - cx * cy, ey2 - cy * cy]])
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1]
    major, minor = 4.0 * np.sqrt(values[order])
    vector = vectors[:, order[0]]
    theta = float(np.arctan2(vector[1], vector[0]))
    return (area, perimeter, float(major), float(minor), theta)

def morphology_2d_from_polygons(polygons: Sequence[np.ndarray] | np.ndarray) -> Morphology2D:
    try:
        dense = np.asarray(polygons, dtype=np.float64)
    except ValueError:
        dense = np.empty((0, 0, 0), dtype=np.float64)
    if dense.ndim == 3 and dense.shape[2] == 2 and (dense.shape[1] >= 3):
        following = np.roll(dense, -1, axis=1)
        cross = dense[:, :, 0] * following[:, :, 1] - following[:, :, 0] * dense[:, :, 1]
        signed_twice_area = cross.sum(axis=1)
        orientation_sign = np.where(signed_twice_area < 0, -1.0, 1.0)
        cross = cross * orientation_sign[:, None]
        signed_twice_area = np.abs(signed_twice_area)
        area = signed_twice_area / 2.0
        perimeter = np.linalg.norm(following - dense, axis=2).sum(axis=1)
        safe_twice_area = np.where(signed_twice_area > _EPS, signed_twice_area, 1.0)
        cx = ((dense[:, :, 0] + following[:, :, 0]) * cross).sum(axis=1) / (3.0 * safe_twice_area)
        cy = ((dense[:, :, 1] + following[:, :, 1]) * cross).sum(axis=1) / (3.0 * safe_twice_area)
        ex2 = ((dense[:, :, 0] ** 2 + dense[:, :, 0] * following[:, :, 0] + following[:, :, 0] ** 2) * cross).sum(axis=1) / (6.0 * safe_twice_area)
        ey2 = ((dense[:, :, 1] ** 2 + dense[:, :, 1] * following[:, :, 1] + following[:, :, 1] ** 2) * cross).sum(axis=1) / (6.0 * safe_twice_area)
        exy = ((2.0 * dense[:, :, 0] * dense[:, :, 1] + dense[:, :, 0] * following[:, :, 1] + following[:, :, 0] * dense[:, :, 1] + 2.0 * following[:, :, 0] * following[:, :, 1]) * cross).sum(axis=1) / (12.0 * safe_twice_area)
        covariance_xx = ex2 - cx * cx
        covariance_yy = ey2 - cy * cy
        covariance_xy = exy - cx * cy
        half_trace = (covariance_xx + covariance_yy) / 2.0
        radius = np.sqrt(np.maximum(0.0, ((covariance_xx - covariance_yy) / 2.0) ** 2 + covariance_xy ** 2))
        lambda_major = np.maximum(0.0, half_trace + radius)
        lambda_minor = np.maximum(0.0, half_trace - radius)
        major = 4.0 * np.sqrt(lambda_major)
        minor = 4.0 * np.sqrt(lambda_minor)
        theta = 0.5 * np.arctan2(2.0 * covariance_xy, covariance_xx - covariance_yy)
        measured = np.column_stack((area, perimeter, major, minor, theta))
    else:
        items = list(polygons)
        measured = np.asarray([_polygon_measurements(item) for item in items], dtype=np.float64)
        if measured.size == 0:
            measured = np.empty((0, 5), dtype=np.float64)
    area, perimeter, major, minor, theta = measured.T
    valid = np.isfinite(measured).all(axis=1) & (area > 0) & (perimeter > 0) & (major > 0) & (minor > 0)
    safe_area = np.where(valid, area, 1.0)
    safe_major = np.where(valid, major, 1.0)
    safe_minor = np.where(valid, minor, 1.0)
    circularity = np.where(valid, 4.0 * np.pi * area / np.square(perimeter), 0.0)
    return Morphology2D(log_area=np.where(valid, np.log(safe_area), 0.0), log_equivalent_radius=np.where(valid, 0.5 * (np.log(safe_area) - np.log(np.pi)), 0.0), log_axis_ratio=np.where(valid, np.log(safe_major / safe_minor), 0.0), circularity=np.where(valid, circularity, 0.0), cos_2theta=np.where(valid, np.cos(2.0 * theta), 0.0), sin_2theta=np.where(valid, np.sin(2.0 * theta), 0.0), area=area, major=major, minor=minor, perimeter=perimeter, valid=valid)

def radial_polygons(centers_xy: np.ndarray, radial_distances: np.ndarray, *, angle_offset_radians: float=0.0) -> np.ndarray:
    centers_xy = np.asarray(centers_xy, dtype=np.float64)
    radial_distances = np.asarray(radial_distances, dtype=np.float64)
    if centers_xy.ndim == 1:
        centers_xy = centers_xy[None, :]
    if radial_distances.ndim == 1:
        radial_distances = radial_distances[None, :]
    if centers_xy.shape != (len(radial_distances), 2) or radial_distances.shape[1] < 3:
        raise ValueError('centers_xy/radial_distances must have shapes [N,2] and [N,R>=3]')
    if not np.isfinite(radial_distances).all() or np.any(radial_distances <= 0):
        raise ValueError('radial distances must be finite and positive')
    angles = angle_offset_radians + np.arange(radial_distances.shape[1]) * (2.0 * np.pi / radial_distances.shape[1])
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return centers_xy[:, None, :] + radial_distances[:, :, None] * directions[None, :, :]

def normalize_xy_positions(centers_xy: np.ndarray, patch_size_xy: tuple[float, float]) -> np.ndarray:
    centers = np.asarray(centers_xy, dtype=np.float64)
    size = np.asarray(patch_size_xy, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('centers_xy must have shape [N,2]')
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('patch_size_xy must contain two finite positive values')
    normalized = centers / size[None, :]
    if not np.isfinite(normalized).all():
        raise ValueError('normalized XY positions are not finite')
    return normalized.astype(np.float32)

def projection_border_flags(polygons: np.ndarray, patch_size_xy: tuple[float, float], *, tolerance: float=1e-06) -> np.ndarray:
    polygons = np.asarray(polygons, dtype=np.float64)
    size = np.asarray(patch_size_xy, dtype=np.float64)
    if polygons.ndim != 3 or polygons.shape[2] != 2:
        raise ValueError('polygons must have shape [N,V,2]')
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('patch_size_xy must contain two finite positive values')
    finite = np.isfinite(polygons).all(axis=(1, 2))
    lower_touch = np.any(polygons <= tolerance, axis=(1, 2))
    upper_touch = np.any(polygons >= size[None, None, :] - tolerance, axis=(1, 2))
    return finite & (lower_touch | upper_touch)

def finite_slab_morphology(centers: np.ndarray, axes: np.ndarray, rotations: np.ndarray, *, slab_bounds_um: tuple[float, float]=(-2.5, 2.5), directions: int=64, chunk_size: int=4096) -> Morphology2D:
    centers, axes, rotations = _validate_ellipsoids(centers, axes, rotations)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    parts: list[Morphology2D] = []
    for start in range(0, len(centers), chunk_size):
        stop = min(len(centers), start + chunk_size)
        polygons, intersects = finite_slab_projection_polygons(centers[start:stop], axes[start:stop], rotations[start:stop], slab_bounds_um=slab_bounds_um, directions=directions, chunk_size=chunk_size)
        result = morphology_2d_from_polygons(polygons)
        if not np.array_equal(result.valid, intersects):
            result = Morphology2D(**{**result.__dict__, 'valid': result.valid & intersects})
        parts.append(result)
    if not parts:
        empty = np.empty(0, dtype=np.float64)
        return Morphology2D(empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty.astype(bool))
    return Morphology2D(**{field: np.concatenate([getattr(part, field) for part in parts]) for field in Morphology2D.__dataclass_fields__})

def morphology_3d(axes: np.ndarray, rotations: np.ndarray) -> Morphology3D:
    axes = np.asarray(axes, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    dummy = np.zeros((axes.reshape(-1, 3).shape[0], 3), dtype=np.float64)
    _, axes, rotations = _validate_ellipsoids(dummy, axes, rotations)
    order = np.argsort(axes, axis=1)[:, ::-1]
    sorted_axes = np.take_along_axis(axes, order, axis=1)
    sorted_rotations = np.take_along_axis(rotations, order[:, None, :], axis=2)
    a, b, c = sorted_axes.T
    volume = 4.0 / 3.0 * np.pi * a * b * c
    valid = np.isfinite(volume) & (volume > 0)
    return Morphology3D(log_volume=np.where(valid, np.log(volume), 0.0), log_ab=np.where(valid, np.log(a / b), 0.0), log_bc=np.where(valid, np.log(b / c), 0.0), long_axis_z_squared=np.where(valid, np.square(sorted_rotations[:, 2, 0]), 0.0), volume=volume, valid=valid)
