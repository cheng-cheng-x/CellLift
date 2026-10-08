from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from ..data import DISTANCE_SCALE_UM, NODE_DIM, Sample, apply_z_permutation, bag_batches, build_batch
from ..models import SceneCorrectionNet, segment_softmax
from ..training import Trainer

class _SyntheticGraph:

    def __init__(self, count: int, edges, base, z, seed: int=0, graph_id: str='synthetic'):
        generator = np.random.default_rng(seed)
        self.graph_id = graph_id
        self.node13 = generator.normal(0.0, 1.0, (count, NODE_DIM)).astype(np.float32)
        self.include = np.ones(count, bool)
        self.valid_ncr = np.ones(count, bool)
        self.center_xy = generator.normal(0.0, 50.0, (count, 2)).astype(np.float32)
        self.center_z = z.astype(np.float32)
        self.edge_index = edges
        self.edge_base = base
        source, target = (edges[0], edges[1])
        self.delta_z = (self.center_z[source] - self.center_z[target]).astype(np.float32)
        self.metadata = {}

class _SyntheticCache:

    def __init__(self, graphs):
        self.graphs = graphs

    def get(self, graph_id):
        return self.graphs[graph_id]

def _ring(count: int):
    source = np.arange(count)
    target = (np.arange(count) + 1) % count
    edges = np.stack((np.concatenate((source, target)), np.concatenate((target, source))))
    base = np.zeros((edges.shape[1], 3), np.float32)
    base[:, 0] = 1.0
    base[:, 1] = 0.5
    base[:, 2] = 0.3
    return (edges.astype(np.int64), base)

def _samples(count: int=3, nodes: int=48, bag: bool=False):
    edges, base = _ring(nodes)
    graphs, samples = ({}, [])
    for index in range(count):
        z = np.linspace(-3.0, 4.0, nodes) + index
        graphs[f'g{index}'] = _SyntheticGraph(nodes, edges, base, z, seed=index, graph_id=f'g{index}')
    if bag:
        samples = [Sample('bag', [f'g{index}' for index in range(count)], 1, 0, 'group', np.asarray([0.3, 0.7], np.float32))]
    else:
        samples = [Sample(f'g{index}', [f'g{index}'], index % 2, 0, f'group{index}', np.asarray([0.25, 0.75], np.float32)) for index in range(count)]
    return (_SyntheticCache(graphs), samples)

def _model(classes: int=2, arm: str='G', bag: bool=False, **kwargs):
    import torch
    torch.manual_seed(0)
    kwargs.setdefault('tile_rgb', False)
    return SceneCorrectionNet(classes=classes, arm=arm, bag=bag, width=16, layers=2, dropout=0.0, **kwargs)

def _final_probability(output, classes: int):
    import torch
    logits = output['logits']
    return torch.sigmoid(logits) if classes == 1 else torch.softmax(logits, -1)

def _binary_samples(nodes: int=32):
    edges, base = _ring(nodes)
    graphs = {'g0': _SyntheticGraph(nodes, edges, base, np.linspace(-2.0, 2.0, nodes), seed=5, graph_id='g0')}
    sample = Sample('patient', ['g0'], 1, 0, 'patient', np.asarray([0.8], np.float32))
    return (_SyntheticCache(graphs), [sample])

def test_zero_initialisation_is_exact_baseline():
    import torch
    cache, samples = _samples()
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, bag=False)
    output = model(batch)
    assert torch.allclose(_final_probability(output, 2), batch.baseline, atol=1e-06)
    assert not torch.allclose(output['logits'], batch.baseline, atol=0.001)
    assert float(output['delta'].abs().max()) < 1e-07
    binary_cache, binary_samples = _binary_samples()
    binary_batch = build_batch(binary_cache, binary_samples, device='cpu', arm='G', ncr_fill=0.0)
    binary = _model(classes=1, bag=False, baseline_dim=1)
    binary_output = binary(binary_batch)
    recovered = _final_probability(binary_output, 1)
    assert torch.allclose(recovered, binary_batch.baseline[:, 0], atol=1e-06)
    assert abs(float(recovered) - 0.8) < 1e-06
    squashed = float(torch.sigmoid(torch.tensor(0.8)))
    assert abs(float(recovered) - squashed) > 0.05
    assert not torch.allclose(binary_output['logits'], binary_batch.baseline[:, 0], atol=0.001)
    assert float(binary_output['delta'].abs().max()) < 1e-07
    recal = _model(classes=2, arm='Recal', bag=False)
    recal_output = recal(batch)
    assert torch.allclose(_final_probability(recal_output, 2), batch.baseline, atol=1e-06)

def test_no_valid_nodes_still_returns_baseline():
    import torch
    cache, samples = _samples(nodes=8)
    for graph in cache.graphs.values():
        graph.include = np.zeros(len(graph.node13), bool)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, bag=True)
    output = model(batch)
    assert torch.allclose(_final_probability(output, 2), batch.baseline, atol=1e-06)

def test_node_permutation_invariance():
    import torch
    cache, samples = _samples()
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, bag=True)
    with torch.no_grad():
        reference = model(batch)['logits']
    rng = np.random.default_rng(7)
    permuted = {}
    for graph_id, graph in cache.graphs.items():
        count = len(graph.node13)
        order = rng.permutation(count)
        inverse = np.argsort(order)
        moved = _SyntheticGraph(count, inverse[graph.edge_index], graph.edge_base, graph.center_z[order], seed=1)
        moved.node13 = graph.node13[order]
        moved.center_xy = graph.center_xy[order]
        permuted[graph_id] = moved
    batch2 = build_batch(_SyntheticCache(permuted), samples, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        result = model(batch2)['logits']
    assert torch.allclose(reference, result, atol=1e-05)

def test_translation_and_z_mirror_invariance():
    import torch
    cache, samples = _samples()
    model = _model(classes=2, bag=True)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        reference = model(batch)['logits']
    moved = {}
    for graph_id, graph in cache.graphs.items():
        clone = _SyntheticGraph(len(graph.node13), graph.edge_index, graph.edge_base, -graph.center_z, seed=2)
        clone.node13 = graph.node13.copy()
        clone.center_xy = (graph.center_xy + np.asarray([1234.0, -987.0], np.float32)).astype(np.float32)
        moved[graph_id] = clone
    batch2 = build_batch(_SyntheticCache(moved), samples, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        result = model(batch2)['logits']
    assert torch.allclose(reference, result, atol=1e-05)

def test_segment_softmax_matches_naive():
    import torch
    scores = torch.randn(37, dtype=torch.float64)
    segment = torch.as_tensor(sorted([0] * 10 + [1] * 5 + [2] * 22), dtype=torch.long)
    fast = segment_softmax(scores, segment, 3)
    for index in range(3):
        selected = segment == index
        shifted = scores[selected] - scores[selected].max()
        expected = shifted.exp() / shifted.exp().sum()
        assert torch.allclose(fast[selected], expected, atol=1e-12), index

def test_edge_chunking_matches_single_pass():
    import torch
    cache, samples = _samples(count=1, nodes=200, bag=True)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, bag=True, edge_chunk=64)
    with torch.no_grad():
        chunked = model(batch)['logits']
    model.edge_chunk = 1000000
    with torch.no_grad():
        whole = model(batch)['logits']
    assert torch.allclose(chunked, whole, atol=1e-06)

def test_xy_arm_zeroes_depth_channels():
    import torch
    cache, samples = _samples()
    batch = build_batch(cache, samples, device='cpu', arm='XY', ncr_fill=0.0)
    assert float(batch.edge[:, 1].abs().max()) == 0.0
    batch_g = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    assert float(batch_g.edge[:, 1].abs().max()) > 0.0

def test_bag_batch_index_bounds():
    import torch
    edges, base = _ring(16)
    graphs, samples = ({}, [])
    for index, tiles in enumerate((1, 5, 3)):
        for tile in range(tiles):
            graph_id = f'b{index}t{tile}'
            graphs[graph_id] = _SyntheticGraph(16, edges, base, np.linspace(-1.0, 1.0, 16) + tile, seed=10 + tile, graph_id=graph_id)
        samples.append(Sample(f'bag{index}', [f'b{index}t{tile}' for tile in range(tiles)], index % 2, 0, f'g{index}', np.asarray([0.4, 0.6], np.float32)))
    cache = _SyntheticCache(graphs)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    assert batch.tile_total == 9
    assert int(batch.node_tile.max()) < batch.tile_total
    assert int(batch.node_graph.max()) < len(samples)
    model = _model(classes=2, bag=True)
    output = model(batch)
    assert output['logits'].shape == (3, 2)
    assert torch.allclose(_final_probability(output, 2), batch.baseline, atol=1e-06)

def test_tile_index_out_of_range_is_rejected():
    cache, samples = _samples(count=1, nodes=16)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    batch.tile_total = 0
    model = _model(classes=2, bag=False)
    try:
        model(batch)
    except (ValueError, IndexError) as error:
        assert 'outside the' in str(error) or 'index out of range' in str(error)
    else:
        raise AssertionError('expected an index-bounds error')

def test_tile_rgb_reaches_the_model():
    import torch
    edges, base = _ring(16)
    graphs, samples = ({}, [])
    for tile in range(3):
        graph_id = f't{tile}'
        graphs[graph_id] = _SyntheticGraph(16, edges, base, np.linspace(-1, 1, 16) + tile, seed=tile, graph_id=graph_id)
    decisions = np.asarray([[0.9, 1.0], [0.2, 0.0], [0.6, 1.0]], np.float32)
    samples.append(Sample('patient', ['t0', 't1', 't2'], 1, 0, 'p', np.asarray([0.7], np.float32), decisions))
    cache = _SyntheticCache(graphs)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    assert tuple(batch.tile_rgb.shape) == (3, 2)
    assert torch.allclose(batch.tile_rgb, torch.as_tensor(decisions))
    model = _model(classes=1, bag=True, tile_rgb=True, baseline_dim=1)
    output = model(batch)
    assert output['logits'].shape == (1,)
    assert float(output['delta'].abs().max()) < 1e-07
    assert torch.allclose(_final_probability(output, 1), batch.baseline[:, 0], atol=1e-06)

def test_baseline_arm_has_no_network():
    try:
        _model(arm='B')
    except ValueError as error:
        assert 'frozen baseline' in str(error)
    else:
        raise AssertionError('expected arm B to be rejected')

def test_recal_has_no_geometry_encoder():
    recal = _model(classes=2, arm='Recal')
    assert recal.node_encoder is None
    assert len(recal.messages) == 0
    morphology = _model(classes=2, arm='M')
    assert morphology.node_encoder is not None
    assert len(morphology.messages) == 0
    geometry = _model(classes=2, arm='G')
    assert len(geometry.messages) == 2

def test_recal_ignores_geometry_and_reads_baseline():
    import torch
    cache, samples = _samples()
    model = _model(classes=2, arm='Recal', bag=True)
    torch.nn.init.normal_(model.head[-1].weight, std=0.3)
    torch.nn.init.normal_(model.head[-1].bias, std=0.3)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        reference = model(batch)['logits']
    scrambled = {}
    for graph_id, graph in cache.graphs.items():
        clone = _SyntheticGraph(len(graph.node13), graph.edge_index, graph.edge_base, graph.center_z + 50.0, seed=99)
        clone.node13 = np.zeros_like(graph.node13)
        clone.center_xy = graph.center_xy + 1000.0
        scrambled[graph_id] = clone
    batch2 = build_batch(_SyntheticCache(scrambled), samples, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        ignored = model(batch2)['logits']
    assert torch.allclose(reference, ignored, atol=1e-06)
    shifted = [Sample(sample.bag_id, sample.graph_ids, sample.label_id, sample.fold, sample.group_id, np.asarray([0.9, 0.1], np.float32)) for sample in samples]
    batch3 = build_batch(cache, shifted, device='cpu', arm='G', ncr_fill=0.0)
    with torch.no_grad():
        changed = model(batch3)['logits']
    assert not torch.allclose(reference, changed, atol=0.001)

def test_apply_z_permutation_keeps_values_not_indices():
    z = np.asarray([-2.0, 0.5, 3.0], np.float32)
    perm = np.asarray([2, 0, 1], np.int32)
    assert np.allclose(apply_z_permutation(z, perm), np.asarray([3.0, -2.0, 0.5], np.float32))

def test_z_permutation_uses_coordinates_not_indices():
    import torch
    nodes = 4
    edges, base = _ring(nodes)
    z = np.asarray([-2.0, 0.5, 3.0, 1.25], np.float32)
    perm = np.asarray([2, 0, 1, 3], np.int32)
    graphs = {'g0': _SyntheticGraph(nodes, edges, base, z, seed=0, graph_id='g0')}
    samples = [Sample('g0', ['g0'], 0, 0, 'g', np.asarray([0.25, 0.75], np.float32))]
    cache = _SyntheticCache(graphs)
    plain = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    permuted = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0, z_offset={'g0': perm})
    assert torch.allclose(plain.node, permuted.node)
    expected_z = apply_z_permutation(z, perm)
    source, target = edges
    expected_delta = (expected_z[source] - expected_z[target]).astype(np.float32)
    assert torch.allclose(permuted.edge[:, 1], torch.as_tensor(expected_delta / DISTANCE_SCALE_UM), atol=1e-05)
    index_delta = (perm[source] - perm[target]).astype(np.float32)
    assert not torch.allclose(permuted.edge[:, 1], torch.as_tensor(index_delta / DISTANCE_SCALE_UM), atol=0.001)
    assert np.allclose(np.sort(expected_z), np.sort(z))

def test_predict_matches_input_sample_order():
    edges, base = _ring(8)
    graphs, samples = ({}, [])
    order = (('c', 2), ('a', 0), ('b', 1))
    for name, label in order:
        graphs[name] = _SyntheticGraph(8, edges, base, np.linspace(-1.0, 1.0, 8), seed=ord(name), graph_id=name)
        probability = np.full(4, 0.05, np.float32)
        probability[label] = 0.85
        samples.append(Sample(name, [name], label, 0, f'g{name}', probability))
    packed = bag_batches(samples, max_bags=8, max_tiles=256)
    packed_ids = [sample.bag_id for part in packed for sample in part]
    assert packed_ids == ['a', 'b', 'c']
    trainer = Trainer(dataset='sicapv2', cache=_SyntheticCache(graphs), arm='G', seed=0, device='cpu', width=16, layers=1, dropout=0.0, bags_per_batch=8, tiles_per_batch=256)
    baseline, final, labels = trainer.predict(samples)
    assert list(labels) == [sample.label_id for sample in samples]
    for row, sample in zip(baseline, samples):
        assert np.allclose(row, sample.baseline, atol=1e-06)
    assert np.allclose(final, baseline, atol=1e-06)
    baseline_r, final_r, labels_r, delta = trainer.predict(samples, residual=True)
    assert np.allclose(baseline_r, baseline)
    assert np.allclose(final_r, final)
    assert np.allclose(labels_r, labels)
    assert delta.shape == baseline.shape
    assert float(np.max(np.abs(delta))) < 1e-06

def test_beta_zero_recovers_baseline_and_one_matches_delta():
    from ..beta_scale import BETA_GRID, choose_beta, probabilities_from_residual
    baseline = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], np.float32)
    delta = np.asarray([[0.2, -0.1, 0.05, -0.15], [0.0, 0.5, -0.3, -0.2]], np.float32)
    delta = delta - delta.mean(-1, keepdims=True)
    zero = probabilities_from_residual(baseline, delta, 0.0)
    assert np.allclose(zero, baseline, atol=1e-06)
    logits = np.log(np.clip(baseline, 1e-07, 1.0)) + delta
    logits = logits - logits.max(-1, keepdims=True)
    expected = np.exp(logits)
    expected = expected / expected.sum(-1, keepdims=True)
    one = probabilities_from_residual(baseline, delta, 1.0)
    assert np.allclose(one, expected, atol=1e-06)
    grid = probabilities_from_residual(baseline, delta, np.asarray(BETA_GRID))
    assert grid.shape == (5, 2, 4)
    assert np.allclose(grid[0], zero)
    assert np.allclose(grid[2], one)
    binary = np.asarray([[0.8], [0.2]], np.float32)
    binary_delta = np.asarray([[0.5], [-0.25]], np.float32)
    binary_zero = probabilities_from_residual(binary, binary_delta, 0.0)
    assert np.allclose(binary_zero, binary, atol=1e-06)
    logit = np.log(0.8) - np.log(0.2)
    expected_one = 1.0 / (1.0 + np.exp(-(logit + 0.5)))
    assert abs(float(probabilities_from_residual(binary, binary_delta, 1.0)[0, 0]) - expected_one) < 1e-06
    assert choose_beta({0.0: 0.5, 1.0: 0.5, 2.0: 0.5}) == 1.0
    assert choose_beta({0.5: 0.8, 1.5: 0.8}) == 0.5
    assert choose_beta({0.0: 0.9, 1.0: 0.8}) == 0.0

def test_heldout_rows_require_matching_labels():
    from ..train import _heldout_rows
    samples = [Sample('c', ['g2'], 2, 0, 'gc', np.asarray([0.1, 0.9], np.float32)), Sample('a', ['g0'], 0, 0, 'ga', np.asarray([0.8, 0.2], np.float32))]
    baseline = np.stack([sample.baseline for sample in samples])
    try:
        _heldout_rows(samples, 'G', 42, baseline_probability=baseline, final_probability=baseline, labels=np.asarray([0, 2]))
    except RuntimeError as error:
        assert 'identity mismatch' in str(error)
    else:
        raise AssertionError('expected identity mismatch')
    rows = _heldout_rows(samples, 'B', 42)
    assert [row['sample_id'] for row in rows] == ['c', 'a']
    assert [row['label_id'] for row in rows] == [2, 0]
    assert np.allclose(rows[0]['final_probability'], samples[0].baseline)

def test_segment_std_matches_naive():
    import torch
    from ..models import segment_std
    values = torch.randn(40, 5, dtype=torch.float64)
    index = torch.as_tensor([0] * 12 + [1] * 8 + [2] * 20, dtype=torch.long)
    fast = segment_std(values, index, 3)
    for segment in range(3):
        selected = values[index == segment]
        expected = torch.sqrt((selected.square().mean(0) - selected.mean(0).square()).clamp_min(0.0))
        assert torch.allclose(fast[segment], expected, atol=1e-12), segment
    empty_index = torch.zeros(0, dtype=torch.long)
    empty = segment_std(values[:0], empty_index, 2)
    assert torch.allclose(empty, torch.zeros(2, 5, dtype=torch.float64))
    large = torch.full((16, 3), 1e+20, dtype=torch.float32)
    large[::2] += 1.0
    large_std = segment_std(large, torch.zeros(16, dtype=torch.long), 1)
    assert torch.isfinite(large_std).all()

def test_ghet_zero_init_is_baseline_and_uses_shared_attention():
    import torch
    from ..models import segment_sum
    cache, samples = _samples(count=3, nodes=24, bag=True)
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, arm='G-Het', bag=True)
    assert model.heterogeneity
    assert int(model.head[0].in_features) == 16 + 16 + 2 + 1
    assert torch.equal(model.head[0].weight[:, 16:32], torch.zeros_like(model.head[0].weight[:, 16:32]))
    assert len(model.messages) == 2
    output = model(batch)
    assert torch.allclose(_final_probability(output, 2), batch.baseline, atol=1e-06)
    assert float(output['delta'].abs().max()) < 1e-07
    encoded = model.node_encoder(batch.node / model.node_scale)
    for layer in model.messages:
        encoded = layer(encoded, batch.edge_index, batch.edge, model.edge_chunk)
    tiles = model.tile_representation(batch, encoded)
    tile_std = model.tile_heterogeneity(batch, encoded)
    weight, size = model.bag_weights(tiles, batch)
    pooled, pooled_std = model.bag_representation(tiles, batch, extra=tile_std)
    assert torch.allclose(pooled, segment_sum(tiles * weight.unsqueeze(-1), batch.tile_to_bag, size), atol=1e-06)
    assert torch.allclose(pooled_std, segment_sum(tile_std * weight.unsqueeze(-1), batch.tile_to_bag, size), atol=1e-06)
    assert tile_std.abs().max() > 0
    torch.nn.init.normal_(model.head[-1].weight, std=0.4)
    torch.nn.init.normal_(model.head[-1].bias, std=0.4)
    with torch.no_grad():
        live = model(batch)['logits']
    assert not torch.allclose(live, output['logits'], atol=0.001)

def test_ghet_std_half_changes_logits_when_spread_changes():
    import torch
    edges, base = _ring(16)
    constant = np.zeros((16, NODE_DIM), np.float32)
    constant[:, 0] = 1.0
    spread = constant.copy()
    spread[::2, 0] = 0.0
    spread[1::2, 0] = 2.0
    z = np.linspace(-1.0, 1.0, 16)
    tight = _SyntheticGraph(16, edges, base, z, seed=0, graph_id='tight')
    tight.node13 = constant
    wide = _SyntheticGraph(16, edges, base, z, seed=0, graph_id='wide')
    wide.node13 = spread
    assert np.allclose(tight.node13.mean(0), wide.node13.mean(0), atol=1e-06)
    samples = [Sample('bag', ['g0'], 1, 0, 'g', np.asarray([0.3, 0.7], np.float32))]
    model = _model(classes=2, arm='G-Het', bag=True)
    torch.nn.init.zeros_(model.head[0].weight)
    torch.nn.init.zeros_(model.head[0].bias)
    std_slice = slice(16, 32)
    torch.nn.init.normal_(model.head[0].weight[:, std_slice], std=0.5)
    torch.nn.init.normal_(model.head[-1].weight, std=0.5)
    torch.nn.init.normal_(model.head[-1].bias, std=0.1)
    with torch.no_grad():
        tight_out = model(build_batch(_SyntheticCache({'g0': tight}), samples, device='cpu', arm='G', ncr_fill=0.0))
        wide_out = model(build_batch(_SyntheticCache({'g0': wide}), samples, device='cpu', arm='G', ncr_fill=0.0))
    assert not torch.allclose(tight_out['logits'], wide_out['logits'], atol=0.0001)

def test_local_adapter_is_1312_and_starts_at_g():
    import torch
    from ..local_readout import source_arm
    from ..models import LocalReadoutAdapter
    assert source_arm('G-Local') == 'G'
    assert source_arm('XY-Local') == 'XY'
    try:
        source_arm('G')
    except ValueError as error:
        assert 'G-Local' in str(error)
    else:
        raise AssertionError('expected source_arm to reject G')
    adapter = LocalReadoutAdapter(width=64, hidden=16, classes=4)
    assert adapter.parameter_count() == 1312
    hidden = torch.randn(12, 64)
    valid = torch.ones(12, dtype=torch.bool)
    index = torch.zeros(12, dtype=torch.long)
    logits = torch.randn(1, 4)
    with torch.no_grad():
        output = adapter(hidden, valid, index, 1, logits)
    assert torch.allclose(output['logits'], logits, atol=1e-05)
    assert torch.allclose(output['delta'], torch.zeros_like(output['delta']), atol=1e-05)

def test_local_adapter_empty_and_single_node_are_zero():
    import torch
    from ..models import LocalReadoutAdapter
    adapter = LocalReadoutAdapter(width=64, hidden=16, classes=4)
    logits = torch.randn(1, 4)
    empty = adapter(torch.zeros(0, 64), torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.long), 1, logits)
    assert torch.allclose(empty['logits'], logits, atol=1e-05)
    assert torch.allclose(empty['delta'], torch.zeros(1, 64), atol=1e-05)
    single = adapter(torch.randn(1, 64), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.long), 1, logits)
    assert torch.allclose(single['delta'], torch.zeros(1, 64), atol=1e-05)
    assert torch.allclose(single['logits'], logits, atol=1e-05)

def test_local_adapter_query_gradient_is_finite_and_nonzero():
    import torch
    from ..models import LocalReadoutAdapter
    torch.manual_seed(0)
    adapter = LocalReadoutAdapter(width=64, hidden=16, classes=4)
    hidden = torch.randn(20, 64)
    valid = torch.ones(20, dtype=torch.bool)
    index = torch.zeros(20, dtype=torch.long)
    logits = torch.zeros(1, 4)
    labels = torch.tensor([1])
    output = adapter(hidden, valid, index, 1, logits)
    loss = torch.nn.functional.cross_entropy(output['logits'], labels) + 0.001 * output['epsilon'].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert adapter.query.grad is not None
    assert float(adapter.query.grad.abs().sum()) > 0
    assert torch.isfinite(adapter.query.grad).all()
    assert torch.isfinite(adapter.project.weight.grad).all()

def test_encode_nodes_does_not_change_g_logits():
    import torch
    cache, samples = _samples()
    batch = build_batch(cache, samples, device='cpu', arm='G', ncr_fill=0.0)
    model = _model(classes=2, bag=False)
    encoded = model.encode_nodes(batch)
    assert encoded.shape[0] == batch.node.shape[0]
    assert encoded.shape[1] == 16
    output = model(batch)
    assert torch.allclose(_final_probability(output, 2), batch.baseline, atol=1e-06)
    assert hasattr(batch, 'node_include')
    assert int(batch.node_include.sum()) == int(batch.node.shape[0])
if __name__ == '__main__':
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith('test_') and callable(function):
            try:
                function()
                print(f'PASS {name}')
            except Exception as error:
                failures += 1
                print(f'FAIL {name}: {error!r}')
    raise SystemExit(1 if failures else 0)
