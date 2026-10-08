from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from ..features import EDGE_DIM, NODE_DIM, apply_arm, include_from_observed, resolve_observed_xy
from ..geometry import direction_spacing, ray_contour_stats
from ..models import FusionHead, GeometryExpert

def test_ray_area_finite_and_rotation_stable():
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rays = (8.0 + 2.0 * np.cos(2 * angles))[None, :].repeat(3, 0)
    node, cov, area = ray_contour_stats(rays)
    assert np.isfinite(node).all() and np.isfinite(cov).all()
    assert np.all(area > 0)
    rolled = np.roll(rays, 9, axis=1)
    node2, _, area2 = ray_contour_stats(rolled)
    assert np.allclose(area, area2, rtol=1e-05, atol=1e-05)
    assert np.allclose(node[:, 0], node2[:, 0], rtol=1e-05, atol=1e-05)

def test_direction_spacing_prefers_long_axis():
    cov = np.array([[[4.0, 0.0], [0.0, 0.25]], [[4.0, 0.0], [0.0, 0.25]]], np.float64)
    along = direction_spacing(np.array([[6.0, 0.0]], np.float64), cov[:1], cov[1:])
    across = direction_spacing(np.array([[0.0, 6.0]], np.float64), cov[:1], cov[1:])
    assert float(along) < float(across)

def test_e2_masks_volume_and_depth():
    node = np.ones((4, NODE_DIM), np.float32)
    edge = np.ones((3, EDGE_DIM), np.float32)
    node2, edge2 = apply_arm(node, edge, 'E2')
    assert np.allclose(node2[:, 38:], 0)
    assert np.allclose(edge2[:, 4:], 0)
    node_es, edge_es = apply_arm(node, edge, 'ES')
    assert np.allclose(node_es[:, 38:], 1)
    assert np.allclose(edge_es[:, 4:], 0)
    node_er, edge_er = apply_arm(node, edge, 'ER')
    assert np.allclose(node_er, 1) and np.allclose(edge_er, 1)

def test_failed_3d_keeps_node_count():
    from ..data import SceneGraph
    payload = {'node2d': np.ones((5, 38), np.float32), 'node3d': np.full((5, 12), np.nan, np.float32), 'include': np.ones(5, bool), 'valid3d': np.zeros(5, bool), 'center_xy': np.zeros((5, 2), np.float32), 'center_z': np.zeros(5, np.float32), 'edge_index': np.zeros((2, 0), np.int64), 'edge2d': np.zeros((0, 4), np.float32), 'edge3d': np.zeros((0, 8), np.float32)}
    graph = SceneGraph('g', payload)
    assert len(graph.node) == 5
    assert graph.include.all()
    assert not graph.valid3d.any()

def test_three_arms_same_parameter_count():
    import torch
    torch.manual_seed(0)
    counts = []
    for _arm in ('E2', 'ES', 'ER'):
        counts.append(GeometryExpert(classes=4, bag=False).parameter_count())
    assert len(set(counts)) == 1
    assert counts[0] > 0

def test_expert_has_no_baseline_tensors():
    model = GeometryExpert(classes=4, bag=False)
    names = {name for name, _ in model.named_parameters()}
    assert not any(('baseline' in name or 'rgb' in name for name in names))

def test_fusion_zero_recovers_baseline_and_stays_nonnegative():
    import torch
    torch.manual_seed(0)
    classes = 4
    baseline = torch.log(torch.softmax(torch.randn(6, classes), -1).clamp(min=1e-06))
    evidence = torch.randn(6, classes)
    head = FusionHead(classes)
    logits = head(baseline, evidence)
    recovered = torch.softmax(logits, -1)
    original = torch.softmax(baseline, -1)
    assert torch.allclose(recovered, original, atol=1e-06)
    head.a.data.fill_(-0.3)
    head.project_()
    assert bool((head.a >= 0).all())

def test_source_uniform_sampling_avoids_exhaustion_tail():
    from ..data import Sample, group_batches, source_uniform_batches
    rng = np.random.default_rng(0)
    samples = []
    for index in range(20):
        samples.append(Sample(f'small-{index}', [f's{index}'], 1, 0, 'small', np.ones(4)))
    for index in range(80):
        samples.append(Sample(f'large-{index}', [f'l{index}'], 0, 0, 'large', np.ones(4)))
    exhausted = [sample for chunk in group_batches(samples, 8, np.random.default_rng(0)) for sample in chunk]
    uniform, meta = source_uniform_batches(samples, 8, rng)
    ordered = [sample for chunk in uniform for sample in chunk]
    tail = ordered[int(0.8 * len(ordered)):]
    assert meta['sampling'] == 'source_uniform_replacement'
    assert meta['steps_per_epoch'] == 13
    assert meta['max_run_same_group'] <= 16
    assert len({sample.group_id for sample in tail}) > 1
    assert all((sample.group_id == 'large' for sample in exhausted[int(0.8 * len(exhausted)):]))

def test_include_ignores_reconstruction_valid_flag():
    from ..features import assemble_graph
    xy = np.array([[0.0, 0.0], [1.0, 0.0]], np.float64)
    rays = np.ones((2, 36), np.float32)
    include = include_from_observed(xy, rays)
    assert include.all()
    fitted = np.array([[np.nan, 0.0], [1.0, 1.0]], np.float64)
    fallback = np.array([[3.0, 4.0], [1.0, 1.0]], np.float64)
    resolved = resolve_observed_xy(fitted, fallback)
    assert np.allclose(resolved[0], [3.0, 4.0])
    assert np.allclose(resolved[1], [1.0, 1.0])
    ids = np.asarray([1, 2], np.int64)
    n_tf = np.repeat(np.eye(3, dtype=np.float64)[None] * 2.0, 2, 0)
    c_tf = np.repeat(np.eye(3, dtype=np.float64)[None] * 4.0, 2, 0)
    payload = {'graph_id': 'g', 'nucleus_id': ids, 'valid': np.array([False, False]), 'nucleus_transform': n_tf, 'cell_transform': c_tf, 'nucleus_center': np.zeros((2, 3), np.float64), 'cell_center': np.zeros((2, 3), np.float64)}
    graph = assemble_graph(payload, rays, ids, xy, ids)
    assert graph['include'].all()
    assert not graph['valid3d'].any()

def test_q3d_c_follows_cell_center_not_nucleus_center():
    from ..features import assemble_graph
    rays = np.full((2, 36), 8.0, np.float32)
    ids = np.asarray([1, 2], np.int64)
    nucleus = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]], np.float64)
    cell_a = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]], np.float64)
    cell_b = np.array([[0.0, 0.0, 0.0], [24.0, 0.0, 0.0]], np.float64)
    obs = nucleus[:, :2]
    n_tf = np.repeat(np.eye(3, dtype=np.float64)[None] * 2.0, 2, 0)
    c_tf = np.repeat(np.eye(3, dtype=np.float64)[None] * 4.0, 2, 0)

    def payload(cell_center, nucleus_center=nucleus):
        return {'graph_id': 'g', 'nucleus_id': ids, 'valid': np.array([True, True]), 'nucleus_transform': n_tf, 'cell_transform': c_tf, 'nucleus_center': nucleus_center, 'cell_center': cell_center}
    first = assemble_graph(payload(cell_a), rays, ids, obs, ids)
    second = assemble_graph(payload(cell_b), rays, ids, obs, ids)
    moved_nucleus = nucleus.copy()
    moved_nucleus[1, 0] = 30.0
    third = assemble_graph(payload(cell_a, moved_nucleus), rays, ids, obs, ids)
    assert first['include'].all()
    assert first['valid3d'].all()
    assert first['center_xy'].shape == (2, 2)
    assert np.allclose(first['center_xy'], obs)
    assert first['edge3d'].shape[0]
    assert not np.allclose(first['edge3d'][:, 2], second['edge3d'][:, 2])
    assert np.allclose(first['edge3d'][:, 2], third['edge3d'][:, 2])

def test_e2_forward_ignores_scrambled_volume():
    import torch
    from ..data import Batch
    from ..features import apply_arm
    torch.manual_seed(0)
    nodes, edges = (12, 20)
    node = torch.randn(nodes, NODE_DIM)
    edge = torch.randn(edges, EDGE_DIM)
    node_a, edge_a = apply_arm(node.numpy(), edge.numpy(), 'E2')
    node_b = node.numpy().copy()
    node_b[:, 38:] = np.random.default_rng(1).normal(size=(nodes, 12))
    node_b, edge_b = apply_arm(node_b, edge.numpy(), 'E2')
    assert np.allclose(node_a, node_b)
    source = np.arange(edges) % nodes
    target = (np.arange(edges) + 1) % nodes
    index = torch.as_tensor(np.stack((source, target)), dtype=torch.long)
    include = torch.ones(nodes, dtype=torch.bool)

    def batch(node_array, edge_array):
        return Batch(node=torch.as_tensor(node_array), edge=torch.as_tensor(edge_array), edge_index=index, include=include, valid3d=torch.ones(nodes, dtype=torch.bool), node_tile=torch.zeros(nodes, dtype=torch.long), tile_bag=torch.zeros(1, dtype=torch.long), labels=torch.zeros(1, dtype=torch.long), baseline=torch.ones(1, 4), n_tiles=1, n_bags=1)
    model = GeometryExpert(classes=4, bag=False)
    model.eval()
    with torch.no_grad():
        left = model(batch(node_a, edge_a))['logits']
        right = model(batch(node_b, edge_b))['logits']
    assert torch.allclose(left, right, atol=1e-06)
