from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from .geometry import DISTANCE_SCALE_UM, RAY_DIM, body_features_torch, direction_spacing, equivalent_radius, knn_edges, ray_contour_stats
KNN = 12
MAX_XY_UM = 50.0
FLAT_AXIS_RATIO = 0.05
NODE_2D = 2 + RAY_DIM
NODE_3D = 12
NODE_DIM = NODE_2D + NODE_3D
EDGE_2D = 4
EDGE_3D = 8
EDGE_DIM = EDGE_2D + EDGE_3D
NODE_2D_SLICE = slice(0, NODE_2D)
NODE_3D_SLICE = slice(NODE_2D, NODE_DIM)
EDGE_2D_SLICE = slice(0, EDGE_2D)
EDGE_3D_SLICE = slice(EDGE_2D, EDGE_DIM)
ARM_NODE_3D = {'E2': False, 'ES': True, 'ER': True}
ARM_EDGE_3D = {'E2': False, 'ES': False, 'ER': True}

def _pack_node3d(direct, offset) -> np.ndarray:
    return np.concatenate((direct, offset), axis=1).astype(np.float32)

def resolve_observed_xy(fitted: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    fitted = np.asarray(fitted, np.float64)
    fallback = np.asarray(fallback, np.float64)
    if fitted.shape != fallback.shape or fitted.ndim != 2 or fitted.shape[1] != 2:
        raise RuntimeError(f'observed XY must be [N,2], got {fitted.shape} and {fallback.shape}')
    finite = np.isfinite(fitted).all(axis=1)
    return np.where(finite[:, None], fitted, fallback)

def include_from_observed(observed_xy: np.ndarray, rays: np.ndarray) -> np.ndarray:
    observed_xy = np.asarray(observed_xy, np.float64)
    rays = np.asarray(rays, np.float64)
    return np.isfinite(observed_xy).all(axis=1) & np.isfinite(rays).all(axis=1)

def assemble_graph(payload: dict, rays: np.ndarray, nucleus_ids_modal: np.ndarray, observed_xy: np.ndarray, observed_ids: np.ndarray) -> dict:
    import torch
    scene_ids = np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64)
    modal_ids = np.asarray(nucleus_ids_modal).reshape(-1).astype(np.int64)
    observed_ids = np.asarray(observed_ids).reshape(-1).astype(np.int64)
    observed_xy = np.asarray(observed_xy, np.float64)
    rays = np.asarray(rays, np.float32)
    if rays.ndim != 2 or rays.shape[1] != RAY_DIM:
        raise RuntimeError(f'rays must be [N,{RAY_DIM}], got {tuple(rays.shape)}')
    if scene_ids.shape != modal_ids.shape or not np.array_equal(scene_ids, modal_ids):
        raise RuntimeError('selected-scene nucleus_id does not match modal rays')
    if not np.array_equal(scene_ids, observed_ids):
        raise RuntimeError('selected-scene nucleus_id does not match observed input ids')
    if len(rays) != len(scene_ids) or len(observed_xy) != len(scene_ids):
        raise RuntimeError('ray or observed XY count does not match selected-scene nuclei')
    node2d, cov2d, _area = ray_contour_stats(rays)
    nucleus_transform = torch.as_tensor(payload['nucleus_transform']).to(torch.float64)
    cell_transform = torch.as_tensor(payload['cell_transform']).to(torch.float64)
    direct, valid_ncr, nucleus_volume, cell_volume, ncov, ccov, naxes, _caxes = body_features_torch(nucleus_transform, cell_transform)
    valid_scene = torch.as_tensor(payload['valid']).to(torch.bool).numpy().astype(bool)
    include = include_from_observed(observed_xy, rays)
    valid3d = valid_scene & valid_ncr.numpy().astype(bool)
    direct_np = direct.numpy().astype(np.float64)
    flat_ratio = ((naxes[:, 0] - naxes[:, 1]) / naxes[:, 0].clamp_min(1e-12)).numpy()
    direct_np[flat_ratio < FLAT_AXIS_RATIO, 3] = 1.0 / 3.0
    nucleus_center = torch.as_tensor(payload['nucleus_center']).to(torch.float64).numpy()
    cell_center = torch.as_tensor(payload['cell_center']).to(torch.float64).numpy()
    radius = equivalent_radius(nucleus_volume).numpy()
    offset = cell_center - nucleus_center
    denom = np.maximum(radius, 1e-06)
    offset_features = np.stack((np.linalg.norm(offset[:, :2], axis=1) / denom, np.abs(offset[:, 2]) / denom, np.linalg.norm(offset, axis=1) / denom), -1)
    node3d = _pack_node3d(direct_np, offset_features)
    node3d[~valid3d] = np.nan
    included = np.flatnonzero(include)
    if len(included) >= 2:
        local = knn_edges(observed_xy[included], MAX_XY_UM, KNN)
        pair = included[local] if len(local) else np.zeros((0, 2), np.int64)
    else:
        pair = np.zeros((0, 2), np.int64)
    ncov_np = ncov.numpy()
    ccov_np = ccov.numpy()
    if len(pair):
        source, target = (pair[:, 0], pair[:, 1])
        delta_xy = observed_xy[target] - observed_xy[source]
        delta_xyz_n = nucleus_center[target] - nucleus_center[source]
        delta_xyz_c = cell_center[target] - cell_center[source]
        q2d = direction_spacing(delta_xy, cov2d[source], cov2d[target])
        q3d_n = direction_spacing(delta_xyz_n, ncov_np[source], ncov_np[target])
        q3d_c = direction_spacing(delta_xyz_c, ccov_np[source], ccov_np[target])
        d_xy = np.linalg.norm(delta_xy, axis=1).astype(np.float32) / DISTANCE_SCALE_UM
        d_log_area = node2d[target, 0] - node2d[source, 0]
        d_axis = node2d[target, 1] - node2d[source, 1]
        radius_sum = np.maximum(radius[source] + radius[target], 1e-06)
        dz_rel = (np.abs(delta_xyz_n[:, 2]) / radius_sum).astype(np.float32)
        d_log_vol_n = (direct_np[target, 0] - direct_np[source, 0]).astype(np.float32)
        d_log_vol_c = (direct_np[target, 4] - direct_np[source, 4]).astype(np.float32)
        d_axis_n = (direct_np[target, 1] - direct_np[source, 1]).astype(np.float32)
        d_axis_c = (direct_np[target, 5] - direct_np[source, 5]).astype(np.float32)
        both3d = valid3d[source] & valid3d[target]
        edge2d = np.stack((d_xy, q2d, d_log_area, d_axis), axis=1).astype(np.float32)
        edge3d = np.stack((dz_rel, q3d_n, q3d_c, q3d_n - q2d, d_log_vol_n, d_log_vol_c, d_axis_n, d_axis_c), axis=1).astype(np.float32)
        edge3d[~both3d] = 0.0
        edge_index = pair.T.astype(np.int32)
    else:
        edge2d = np.zeros((0, EDGE_2D), np.float32)
        edge3d = np.zeros((0, EDGE_3D), np.float32)
        edge_index = np.zeros((2, 0), np.int32)
    node2d_full = np.concatenate((node2d, np.asarray(rays, np.float32)), axis=1)
    return {'graph_id': str(payload.get('graph_id') or (payload.get('metadata') or {}).get('graph_id')), 'node2d': node2d_full.astype(np.float32), 'node3d': node3d.astype(np.float32), 'include': include.astype(bool), 'valid3d': valid3d.astype(bool), 'center_xy': observed_xy.astype(np.float32), 'center_z': nucleus_center[:, 2].astype(np.float32), 'edge_index': edge_index, 'edge2d': edge2d, 'edge3d': edge3d, 'metadata': dict(payload.get('metadata') or {}), 'stats': {'nodes': int(len(node2d_full)), 'include': int(include.sum()), 'valid3d': int(valid3d.sum()), 'failed3d': int((include & ~valid3d).sum()), 'edges': int(edge_index.shape[1])}}

def apply_arm(node: np.ndarray, edge: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    if arm not in ARM_NODE_3D:
        raise ValueError(arm)
    node = np.array(node, np.float32, copy=True)
    edge = np.array(edge, np.float32, copy=True)
    if not ARM_NODE_3D[arm]:
        node[..., NODE_3D_SLICE] = 0.0
    if not ARM_EDGE_3D[arm]:
        edge[..., EDGE_3D_SLICE] = 0.0
    return (node, edge)
