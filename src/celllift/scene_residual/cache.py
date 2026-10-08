from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
import numpy as np
from .geometry import body_features_torch, body_statistics_torch, covariance_upper, equivalent_radius
from .io_utils import atomic_json, atomic_npz, safe_component
KNN = 12
MAX_XY_UM = 50.0
DISTANCE_SCALE_UM = 10.0
FLAT_AXIS_RATIO = 0.05

def cache_root(cfg: Mapping[str, Any], dataset: str) -> Path:
    batch = str(cfg.get('batch', 'set_encoding'))
    return Path(cfg['paths']['data_root']) / dataset / f'scene_graphs_{batch}'

def _scene_files(cfg: Mapping[str, Any], dataset: str, shard: int) -> list[Path]:
    root = Path(cfg['paths']['set_encoding_data_root']) / dataset / '02_projection_scene_selected_scene' / f'shard_{shard:03d}'
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(root.glob('*.pt'))

def _knn_edges(xy: np.ndarray, cap: float, k: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(xy)
    empty_pairs = np.zeros((0, 2), np.int64)
    if count < 2:
        return (np.zeros((2, 0), np.int64), empty_pairs)
    pairs = []
    for start in range(0, count, 2048):
        block = xy[start:start + 2048]
        distance = np.linalg.norm(block[:, None, :] - xy[None, :, :], axis=-1)
        rows = np.arange(len(block))
        distance[rows, rows + start] = np.inf
        neighbours = min(k, count - 1)
        index = np.argpartition(distance, neighbours - 1, axis=1)[:, :neighbours]
        selected = np.take_along_axis(distance, index, axis=1)
        keep = selected <= cap
        source = np.repeat(np.arange(start, start + len(block)), neighbours).reshape(len(block), neighbours)
        pairs.append(np.stack((source[keep], index[keep]), axis=1))
    pair = np.concatenate(pairs, 0) if pairs else empty_pairs
    if not len(pair):
        return (np.zeros((2, 0), np.int64), empty_pairs)
    pair = np.concatenate((pair, pair[:, ::-1]), 0)
    order = np.lexsort((pair[:, 1], pair[:, 0]))
    pair = pair[order]
    keep = np.ones(len(pair), bool)
    keep[1:] = np.any(pair[1:] != pair[:-1], axis=1)
    pair = pair[keep].astype(np.int64)
    return (pair.T, pair)

def build_scene_graph(payload: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    import torch
    nucleus_transform = torch.as_tensor(payload['nucleus_transform']).to(torch.float64)
    cell_transform = torch.as_tensor(payload['cell_transform']).to(torch.float64)
    direct, valid_ncr, nucleus_volume, cell_volume, nucleus_cov, _ = body_features_torch(nucleus_transform, cell_transform)
    _, nucleus_axes, _, _ = body_statistics_torch(nucleus_transform)
    valid_scene = torch.as_tensor(payload['valid']).to(torch.bool)
    include = (valid_scene & valid_ncr).numpy()
    direct = direct.numpy().astype(np.float64)
    nucleus_center = torch.as_tensor(payload['nucleus_center']).to(torch.float64).numpy()
    cell_center = torch.as_tensor(payload['cell_center']).to(torch.float64).numpy()
    nucleus_radius = equivalent_radius(nucleus_volume).numpy()
    nucleus_cov = nucleus_cov.numpy()
    flat_ratio = ((nucleus_axes[:, 0] - nucleus_axes[:, 1]) / nucleus_axes[:, 0].clamp_min(1e-12)).numpy()
    near_spherical = flat_ratio < FLAT_AXIS_RATIO
    direct[near_spherical, 3] = 1.0 / 3.0
    offset = cell_center - nucleus_center
    radius = np.maximum(nucleus_radius, 1e-06)
    offset_features = np.stack((np.linalg.norm(offset[:, :2], axis=1) / radius, np.abs(offset[:, 2]) / radius, np.linalg.norm(offset, axis=1) / radius), -1)
    node = np.concatenate((direct, offset_features, valid_ncr.numpy()[:, None].astype(np.float64)), -1)
    edges, pair = _knn_edges(nucleus_center[:, :2], MAX_XY_UM, KNN)
    if pair.shape[0]:
        source, target = (pair[:, 0], pair[:, 1])
        delta_xy = np.linalg.norm(nucleus_center[source, :2] - nucleus_center[target, :2], axis=1)
        radius_sum = np.maximum(nucleus_radius[source] + nucleus_radius[target], 1e-06)
        covariance_source = nucleus_cov[source]
        covariance_target = nucleus_cov[target]
        numerator = np.einsum('nij,nij->n', covariance_source, covariance_target)
        denominator = np.maximum(np.linalg.norm(covariance_source.reshape(len(source), -1), axis=1) * np.linalg.norm(covariance_target.reshape(len(target), -1), axis=1), 1e-12)
        edge_base = np.stack((delta_xy / DISTANCE_SCALE_UM, 1.0 / radius_sum, numerator / denominator), -1).astype(np.float32)
        delta_z = (nucleus_center[source, 2] - nucleus_center[target, 2]).astype(np.float32)
        degree = np.bincount(target, minlength=len(nucleus_center))
    else:
        edge_base = np.zeros((0, 3), np.float32)
        delta_z = np.zeros((0,), np.float32)
        degree = np.zeros(len(nucleus_center), np.int64)
    return {'graph_id': str(payload['graph_id']), 'dataset': dataset, 'node13': node.astype(np.float32), 'center_xy': nucleus_center[:, :2].astype(np.float32), 'center_z': nucleus_center[:, 2].astype(np.float32), 'edge_index': edges.astype(np.int32), 'edge_base': edge_base, 'delta_z': delta_z, 'include': include, 'valid_ncr': valid_ncr.numpy(), 'metadata': dict(payload.get('metadata') or {}), 'stats': {'nodes': int(len(node)), 'valid_ncr': int(valid_ncr.sum().item()), 'include': int(include.sum()), 'near_spherical': int(near_spherical.sum()), 'edges': int(pair.shape[0]), 'isolated': int(np.count_nonzero(degree == 0)), 'median_abs_dz': float(np.median(np.abs(delta_z))) if len(delta_z) else 0.0, 'mean_edge_xy_um': float(np.mean(edge_base[:, 0]) * DISTANCE_SCALE_UM) if len(edge_base) else 0.0}}

def cache_shard(cfg: Mapping[str, Any], dataset: str, shard: int) -> dict[str, Any]:
    import torch
    destination = cache_root(cfg, dataset) / f'shard_{shard:03d}'
    files = _scene_files(cfg, dataset, shard)
    entries = []
    totals = {'nodes': 0, 'include': 0, 'edges': 0, 'isolated': 0, 'near_spherical': 0}
    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        graph = build_scene_graph(payload, dataset)
        target = destination / f"{safe_component(graph['graph_id'])}.npz"
        atomic_npz(target, node13=graph['node13'], center_xy=graph['center_xy'], center_z=graph['center_z'], edge_index=graph['edge_index'], edge_base=graph['edge_base'], delta_z=graph['delta_z'], include=graph['include'], valid_ncr=graph['valid_ncr'])
        stats = graph['stats']
        for key in totals:
            totals[key] += stats[key]
        entries.append({'graph_id': graph['graph_id'], 'path': str(target), 'metadata': graph['metadata'], **stats})
    manifest = {'status': 'PASS', 'dataset': dataset, 'shard': int(shard), **totals, 'mean_degree': totals['edges'] / totals['nodes'] if totals['nodes'] else 0.0, 'entries': entries}
    atomic_json(destination / 'manifest.json', manifest)
    return {key: value for key, value in manifest.items() if key != 'entries'}

def merge_shards(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    root = Path(cfg['paths']['data_root']) / dataset
    graphs: list[dict] = []
    totals = {'graphs': 0, 'nodes': 0, 'include': 0, 'edges': 0, 'isolated': 0, 'near_spherical': 0}
    for shard in sorted(cache_root(cfg, dataset).glob('shard_*')):
        manifest_path = shard / 'manifest.json'
        if not manifest_path.is_file():
            raise RuntimeError(f'incomplete shard: {manifest_path}')
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS':
            raise RuntimeError(f'shard not PASS: {manifest_path}')
        graphs.extend(payload['entries'])
        for key in totals:
            totals[key] += int(payload.get(key, 0))
    graphs.sort(key=lambda row: row['graph_id'])
    totals['graphs'] = len(graphs)
    index = {'status': 'PASS', 'dataset': dataset, 'batch': str(cfg.get('batch', 'set_encoding')), **totals, 'mean_degree': totals['edges'] / totals['nodes'] if totals['nodes'] else 0.0, 'graphs': graphs}
    atomic_json(root / 'scene_index.json', index)
    return {'status': 'PASS', 'dataset': dataset, 'graphs': totals['graphs'], 'nodes': totals['nodes'], 'edges': totals['edges']}

def check_cache(cfg: Mapping[str, Any], dataset: str, sample: int=3) -> dict[str, Any]:
    import numpy as np
    root = Path(cfg['paths']['data_root']) / dataset
    index = json.loads((root / 'scene_index.json').read_text(encoding='utf-8'))
    if str(index.get('batch')) != str(cfg.get('batch', 'set_encoding')):
        raise RuntimeError(f"scene_index was built for batch {index.get('batch')!r} but config asks for {cfg.get('batch')!r}")
    expected = set()
    for shard in sorted((Path(cfg['paths']['set_encoding_data_root']) / dataset / '02_projection_scene_selected_scene').glob('shard_*')):
        expected.update((path.stem for path in shard.glob('*.pt')))
    rows = index['graphs']
    if len(rows) != len({row['graph_id'] for row in rows}):
        raise RuntimeError('duplicate graph_id in scene index')
    problems = []
    sample = max(1, int(sample))
    step = max(1, len(rows) // sample)
    degree_total, node_total = (0, 0)
    for row in rows:
        arrays = np.load(row['path'])
        count = len(arrays['node13'])
        if count != row['nodes']:
            problems.append({'graph_id': row['graph_id'], 'reason': 'node count mismatch'})
            continue
        if arrays['node13'].shape[1] != 13:
            problems.append({'graph_id': row['graph_id'], 'reason': 'node width mismatch'})
            continue
        edges = arrays['edge_index']
        if edges.shape != (2, int(row['edges'])):
            problems.append({'graph_id': row['graph_id'], 'reason': 'edge count mismatch'})
            continue
        if edges.size and (edges.min() < 0 or edges.max() >= count):
            problems.append({'graph_id': row['graph_id'], 'reason': 'edge index out of range'})
        node_total += count
        degree_total += edges.shape[1]
    payload = {'status': 'FAIL' if problems else 'PASS', 'dataset': dataset, 'scene_files': len(expected), 'cache_graphs': len(rows), 'nodes': int(node_total), 'edges': int(degree_total), 'mean_degree': degree_total / node_total if node_total else 0.0, 'problems': problems[:20], 'problem_count': len(problems)}
    return payload
