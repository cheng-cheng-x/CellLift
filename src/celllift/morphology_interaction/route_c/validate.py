from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
import numpy as np
from ..dataset import B_ROLE, PREDICTION_FORM, batch_result_root
from ..io_utils import atomic_json
from .fields import EXTRUDE_THICKNESS_UM, PLANES, raster_extrude, raster_pair, xy_grid_um

def _synthetic_body(center, scale):
    transform = np.eye(3, dtype=np.float32) * float(scale)
    return (np.asarray(center, np.float32)[None], transform[None])

def run_validate(cfg: Mapping[str, Any], dataset: str='sicapv2') -> dict[str, Any]:
    offsets = np.asarray((-8.0, -4.0, 0.0, 4.0, 8.0), np.float32)
    include = np.ones((1,), bool)
    valid = np.ones((1,), bool)
    grid = xy_grid_um(dataset)
    mid = grid.shape[0] // 2
    xy = grid[mid, mid]
    center = np.asarray([xy[0], xy[1], 0.0], np.float32)
    c1, t1 = _synthetic_body(center, 32.0)
    c2, t2 = _synthetic_body(center, 32.0 * 2.0 ** (1.0 / 3.0))
    field = raster_pair(c1, t1, c1, t1, dataset, offsets, include, valid)
    peak_plane = int(np.argmin([field[2 * i].min() for i in range(PLANES)]))
    peak_yx = np.unravel_index(np.argmin(field[2 * peak_plane]), field[2 * peak_plane].shape)
    aligned = abs(int(peak_yx[0]) - mid) <= 2 and abs(int(peak_yx[1]) - mid) <= 2
    z_ok = offsets[peak_plane] == 0.0 or abs(float(offsets[peak_plane])) <= 4.0
    field_big = raster_pair(c2, t2, c2, t2, dataset, offsets, include, valid)
    scale_ok = float(field_big[4].min()) < float(field[4].min()) or float((field_big[4] < 0).sum()) > float((field[4] < 0).sum())
    outside = np.ones((1,), bool)
    far, tfar = _synthetic_body(np.asarray([100000.0, 100000.0, 0.0], np.float32), 8.0)
    far_field = raster_pair(far, tfar, far, tfar, dataset, offsets, include, valid)
    oob_ok = float(far_field[4].min()) >= 0.0
    mask = np.zeros((1024, 1024), np.uint8)
    mask[500:524, 500:524] = 1
    extruded = raster_extrude(mask, dataset, offsets, EXTRUDE_THICKNESS_UM)
    thickness_ok = extruded.shape[0] == 10 and float(np.max(np.abs(extruded))) > 0
    gates = {'known_body_depth_xy': bool(z_ok and aligned), 'two_x_volume_distinct': bool(scale_ok), 'peak_aligns_xy': bool(aligned), 'outside_window_not_empty_tissue': bool(oob_ok), 'extrude_thickness_fixed': bool(thickness_ok and EXTRUDE_THICKNESS_UM == 4.0)}
    passed = all(gates.values())
    payload = {'status': 'PASS' if passed else 'FAIL', 'dataset': dataset, 'route': 'c', 'gates': gates, 'offsets_um': offsets.tolist(), 'extrude_thickness_um': EXTRUDE_THICKNESS_UM, 'prediction_form': PREDICTION_FORM, 'b_role': B_ROLE, 'allow_train': passed}
    destination = batch_result_root(cfg, 'c') / dataset / 'field_validate.json'
    atomic_json(destination, payload)
    return payload
