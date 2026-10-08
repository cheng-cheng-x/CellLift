from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import csv
import hashlib
from celllift.runtime import json
import math
import sys
from dataclasses import fields
from celllift.runtime import ResourcePath as Path
from typing import Callable
import numpy as np
from celllift.runtime import torch
from ..frozen.batch_types import EllipseObservationBatch, NucleusGraphBatch

def moment_ellipse_from_mask(mask: np.ndarray, bbox_xyxy, mpp: float=0.46) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not math.isfinite(mpp) or mpp <= 0:
        raise ValueError('expected a 2-D binary crop and positive finite mpp')
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError('cannot construct moments of an empty observed mask')
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    if tuple(mask.shape) != (int(y1 - y0), int(x1 - x0)):
        raise ValueError('mask crop shape and half-open bounding box disagree')
    cx, cy = (float(xs.mean()), float(ys.mean()))
    dx, dy = (xs.astype(np.float64) - cx, ys.astype(np.float64) - cy)
    covariance_px = np.array([[np.mean(dx * dx) + 1.0 / 12.0, np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy) + 1.0 / 12.0]])
    det_root = math.sqrt(float(np.linalg.det(covariance_px)))
    root_px = 2.0 * (covariance_px + det_root * np.eye(2)) / math.sqrt(float(np.trace(covariance_px)) + 2.0 * det_root)
    precision_px = np.linalg.inv(4.0 * covariance_px)
    center_px = np.array([x0 + cx + 0.5, y0 + cy + 0.5])
    extent = np.sqrt(4.0 * np.diag(covariance_px))
    lo = np.floor(center_px - extent - 0.5).astype(np.int64)
    hi = np.ceil(center_px + extent - 0.5).astype(np.int64) + 1
    gy, gx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
    ex, ey = (gx + 0.5 - center_px[0], gy + 0.5 - center_px[1])
    ellipse_raster = precision_px[0, 0] * ex * ex + 2.0 * precision_px[0, 1] * ex * ey + precision_px[1, 1] * ey * ey <= 1.0
    fx, fy = (xs - cx, ys - cy)
    intersection = int(np.count_nonzero(precision_px[0, 0] * fx * fx + 2.0 * precision_px[0, 1] * fx * fy + precision_px[1, 1] * fy * fy <= 1.0))
    dice = 2.0 * intersection / (int(xs.size) + int(ellipse_raster.sum()))
    return {'center_xy': center_px * mpp, 'root_xy': root_px * mpp, 'precision_xy': precision_px / (mpp * mpp), 'covariance_xy': covariance_px * (mpp * mpp), 'area_um2': math.pi * float(np.linalg.det(root_px)) * mpp * mpp, 'raw_area_um2': float(xs.size) * mpp * mpp, 'representation_dice': float(dice), 'circle_fallback': False}

def _fit_uid(uid: str, read_target: Callable, cache: dict, mpp: float) -> np.ndarray:
    if uid not in cache:
        target = read_target(uid)
        expected = f'{int(target.layer_idx)}:{int(target.object_type)}:{int(target.instance_id)}'
        if str(target.target_uid) != uid or expected != uid:
            raise ValueError(f'target UID/header mismatch: requested {uid}, got {expected}')
        fitted = moment_ellipse_from_mask(target.decode_mask(), target.bbox_xyxy, mpp)
        if not math.isclose(fitted['raw_area_um2'], float(target.area_px) * mpp * mpp, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'packed mask area/header mismatch: {uid}')
        root = fitted['root_xy']
        cache[uid] = np.array([*fitted['center_xy'], root[0, 0], root[0, 1], root[1, 1], fitted['area_um2'], fitted['raw_area_um2'], fitted['representation_dice']], dtype=np.float32)
    return cache[uid]

def _root_rows(rows: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rows[:, [2, 3, 3, 4]].copy().reshape(-1, 2, 2))

def graph_from_record(record, read_target: Callable, cache: dict, mpp: float):
    from .components import _union_find_components
    ids = np.asarray(record.nucleus_id, dtype=np.int64)
    rows = np.stack([_fit_uid(f'{int(record.layer_idx)}:0:{int(nucleus_id)}', read_target, cache, mpp) for nucleus_id in ids])
    root = _root_rows(rows)
    inverse = torch.linalg.inv(root)
    undirected = np.stack([record.undirected_edge_src, record.undirected_edge_dst]).astype(np.int64)
    graph = NucleusGraphBatch(nucleus_rays_um=torch.from_numpy(np.asarray(record.rho_um, dtype=np.float32)), nucleus_xy_um=torch.from_numpy(np.asarray(record.xy_um, dtype=np.float32)), edge_index=torch.from_numpy(np.stack([record.edge_src, record.edge_dst]).astype(np.int64)), undirected_edge_index=torch.from_numpy(undirected), graph_ptr=torch.tensor([0, len(ids)], dtype=torch.long), nucleus_id=torch.from_numpy(ids), nucleus_component_index=torch.from_numpy(_union_find_components(len(ids), undirected)), fitted_center_xy=torch.from_numpy(rows[:, :2].copy()), fitted_root_xy=root, fitted_precision_xy=inverse.transpose(-1, -2) @ inverse, fitted_area_um2=torch.from_numpy(rows[:, 5].copy()), ellipse_circle_fallback=torch.zeros(len(ids), dtype=torch.bool))
    graph.validate()
    return (graph, rows)

def observations_from_record(record, graph: NucleusGraphBatch, object_type: int, read_target: Callable, cache: dict, mpp: float):
    chosen = np.flatnonzero(np.asarray(record.object_type) == int(object_type))
    if not len(chosen):
        return EllipseObservationBatch(torch.empty((0, 2)), torch.empty((0, 2, 2)), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool))
    rows = np.stack([_fit_uid(str(record.target_uid[i]), read_target, cache, mpp) for i in chosen])
    owners = torch.as_tensor(np.asarray(record.anchor_node_idx)[chosen], dtype=torch.long)
    planes = torch.as_tensor(np.asarray(record.plane_idx)[chosen], dtype=torch.long)
    if not bool(((planes >= 0) & (planes <= 2)).all()):
        raise ValueError('observation plane must be lower=0, middle=1, upper=2')
    if owners.min() < 0 or owners.max() >= graph.nucleus_id.numel():
        raise ValueError('observation owner is outside the middle nucleus graph')
    weights = torch.as_tensor(np.asarray(record.weight)[chosen], dtype=torch.float32)
    centers = torch.from_numpy(rows[:, :2].copy())
    roots = _root_rows(rows)
    anchor_roots = graph.fitted_root_xy[owners]
    result = EllipseObservationBatch(target_center_normalized=torch.linalg.solve(anchor_roots, (centers - graph.fitted_center_xy[owners])[..., None]).squeeze(-1), target_root_normalized=torch.linalg.solve(anchor_roots, roots), anchor_node_index=owners, plane_index=planes, weight=weights, raw_area_um2=torch.from_numpy(rows[:, 6].copy()), representation_dice=torch.from_numpy(rows[:, 7].copy()), circle_fallback=torch.zeros(len(rows), dtype=torch.bool))
    result.validate(graph.nucleus_id.numel())
    return result
