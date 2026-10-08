from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
import numpy as np
from ..baseline import crc_mean_then_vote
from ..dataset import ARM_EDGE_3D, ARM_NODE_3D, DINO_DIM
from ..features import EDGE_DIM, NODE_DIM, apply_arm, include_from_observed, resolve_observed_xy
from ..geometry import direction_spacing, ray_contour_stats
from ..route_c.fields import EXTRUDE_THICKNESS_UM, raster_pair
from ..route_c.validate import run_validate

def test_include_ignores_valid():
    xy = np.array([[0.0, 0.0], [1.0, 0.0]], np.float64)
    rays = np.ones((2, 36), np.float32)
    assert include_from_observed(xy, rays).all()
    fitted = np.array([[np.nan, 0.0], [1.0, 1.0]], np.float64)
    fallback = np.array([[3.0, 4.0], [1.0, 1.0]], np.float64)
    resolved = resolve_observed_xy(fitted, fallback)
    assert np.allclose(resolved[0], [3.0, 4.0])

def test_arm_masks_unused_3d():
    node = np.ones((4, NODE_DIM), np.float32)
    edge = np.ones((3, EDGE_DIM), np.float32)
    for arm, use_n, use_e in (('A2', False, False), ('AS', True, False), ('AR', True, True), ('B2', False, False), ('B3', True, True)):
        n, e = apply_arm(node, edge, arm)
        assert ARM_NODE_3D[arm] is use_n
        assert ARM_EDGE_3D[arm] is use_e
        if not use_n:
            assert np.allclose(n[:, 38:], 0)
        if not use_e:
            assert np.allclose(e[:, 4:], 0)

def test_q3d_c_uses_cell_centers():
    try:
        import torch
    except ImportError:
        return
    from ..features import assemble_graph
    ids = np.asarray([1, 2], np.int64)
    xy = np.array([[0.0, 0.0], [10.0, 0.0]], np.float64)
    rays = np.ones((2, 36), np.float32) * 4.0
    dino = np.zeros((2, DINO_DIM), np.float32)
    nucle = np.eye(3, dtype=np.float32)[None].repeat(2, 0) * 4.0
    cell = np.eye(3, dtype=np.float32)[None].repeat(2, 0) * 8.0
    payload = {'graph_id': 'g', 'nucleus_id': ids, 'valid': np.ones(2, bool), 'nucleus_transform': nucle, 'cell_transform': cell, 'nucleus_center': np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], np.float32), 'cell_center': np.array([[0.0, 0.0, 2.0], [10.0, 0.0, -2.0]], np.float32), 'metadata': {}}
    graph = assemble_graph(payload, rays, ids, xy, ids, dino)
    assert graph['edge3d'].shape[1] == 8
    assert np.isfinite(graph['edge3d'][:, 2]).any()

def test_full_prediction_does_not_add_baseline():
    source = Path(__file__).resolve().parents[1] / 'route_b' / 'models.py'
    text = source.read_text(encoding='utf-8')
    assert 'def forward' in text
    assert 'baseline + ' not in text
    assert 'increment' not in text

def test_crc_vote_order():
    tiles = np.array([[0.6, 0.4, 0.7], [0.6, 0.4, 0.7]], np.float64)
    assert 0.0 < crc_mean_then_vote(tiles) < 1.0

def test_c_validation_gates():
    payload = run_validate({'paths': {'result_root': '.'}}, 'sicapv2')
    assert payload['gates']['extrude_thickness_fixed']
    assert EXTRUDE_THICKNESS_UM == 4.0

def test_source_uniform_sampling():
    from ..data import Sample, source_uniform_batches
    samples = [Sample(f's{i}', [f's{i}'], 0, 0, 'a' if i < 5 else 'b', np.ones(4)) for i in range(20)]
    batches, meta = source_uniform_batches(samples, 4, np.random.default_rng(0))
    assert meta['sampling'] == 'source_uniform_replacement'
    assert batches

def test_ray_area_finite():
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rays = (8.0 + 2.0 * np.cos(2 * angles))[None, :].repeat(2, 0)
    node, cov, area = ray_contour_stats(rays)
    assert np.isfinite(node).all() and np.all(area > 0)
    along = direction_spacing(np.array([[6.0, 0.0]]), np.array([[[4.0, 0.0], [0.0, 0.25]]]), np.array([[[4.0, 0.0], [0.0, 0.25]]]))
    across = direction_spacing(np.array([[0.0, 6.0]]), np.array([[[4.0, 0.0], [0.0, 0.25]]]), np.array([[[4.0, 0.0], [0.0, 0.25]]]))
    assert float(np.asarray(along).reshape(-1)[0]) < float(np.asarray(across).reshape(-1)[0])
