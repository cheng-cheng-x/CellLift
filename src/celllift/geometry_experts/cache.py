from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
import numpy as np
from .dataset import batch_name, cache_index_path, cache_root
from .features import assemble_graph, resolve_observed_xy
from .io_utils import atomic_json, atomic_npz, load_npz, safe_component

def _scene_files(cfg: Mapping[str, Any], dataset: str, shard: int) -> list[Path]:
    root = Path(cfg['paths']['set_encoding_data_root']) / dataset / '02_projection_scene_selected_scene' / f'shard_{shard:03d}'
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(root.glob('*.pt'))

def modal_lookup(cfg: Mapping[str, Any], dataset: str) -> dict[str, str]:
    import pyarrow.parquet as pq
    path = Path(cfg['paths']['set_encoding_data_root']) / dataset / '03_modal_features' / 'index.parquet'
    if not path.is_file():
        raise FileNotFoundError(path)
    mapping = {}
    for row in pq.read_table(path, partitioning=None).to_pylist():
        raw = Path(str(row['path']))
        mapping[str(row['graph_id'])] = str(raw if raw.is_absolute() else path.parent / raw)
    return mapping

def input_lookup(cfg: Mapping[str, Any], dataset: str) -> dict[str, str]:
    root = Path(cfg['paths']['set_encoding_data_root']) / dataset / '01_projection_scene_inputs'
    manifest = root / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    mapping = {}
    for row in payload['items']:
        raw = Path(str(row['path']))
        mapping[str(row['graph_id'])] = str(raw if raw.is_absolute() else root / raw)
    return mapping

def _ensure_projection_scene(cfg: Mapping[str, Any]) -> None:
    from celllift.matched_geometry_controls.upstream import load_projection_scene
    load_projection_scene(cfg['paths']['projection_scene_source_root'])

def _observed_from_input(path: str, nucleus_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    graph = payload['graph']
    ids = np.asarray(graph.nucleus_id).reshape(-1).astype(np.int64)
    fitted = np.asarray(graph.fitted_center_xy, np.float64)
    fallback = np.asarray(graph.nucleus_xy_um, np.float64)
    del payload
    if not np.array_equal(ids, nucleus_ids):
        raise RuntimeError(f'input nucleus_id does not match selected-scene for {path}')
    return (resolve_observed_xy(fitted, fallback), ids)

def cache_shard(cfg: Mapping[str, Any], dataset: str, shard: int) -> dict[str, Any]:
    import torch
    _ensure_projection_scene(cfg)
    destination = cache_root(cfg, dataset) / f'shard_{shard:03d}'
    lookup = modal_lookup(cfg, dataset)
    inputs = input_lookup(cfg, dataset)
    files = _scene_files(cfg, dataset, shard)
    entries = []
    totals = {'nodes': 0, 'include': 0, 'valid3d': 0, 'failed3d': 0, 'edges': 0}
    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        graph_id = str(payload.get('graph_id') or (payload.get('metadata') or {}).get('graph_id') or path.stem)
        payload['graph_id'] = graph_id
        modal_path = lookup.get(graph_id)
        input_path = inputs.get(graph_id)
        if modal_path is None:
            raise RuntimeError(f'modal rays missing for {graph_id}')
        if input_path is None:
            raise RuntimeError(f'observed input missing for {graph_id}')
        modal = load_npz(modal_path)
        observed_xy, observed_ids = _observed_from_input(input_path, np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64))
        graph = assemble_graph(payload, modal['rays36'], modal['nucleus_id'], observed_xy, observed_ids)
        target = destination / f"{safe_component(graph['graph_id'])}.npz"
        atomic_npz(target, node2d=graph['node2d'], node3d=graph['node3d'], include=graph['include'], valid3d=graph['valid3d'], center_xy=graph['center_xy'], center_z=graph['center_z'], edge_index=graph['edge_index'], edge2d=graph['edge2d'], edge3d=graph['edge3d'])
        stats = graph['stats']
        for key in totals:
            totals[key] += stats[key]
        entries.append({'graph_id': graph['graph_id'], 'path': str(target), 'metadata': graph['metadata'], **stats})
    manifest = {'status': 'PASS', 'dataset': dataset, 'batch': batch_name(cfg), 'shard': int(shard), 'graphs': len(entries), **totals, 'entries': entries}
    atomic_json(destination / 'manifest.json', manifest)
    return {key: value for key, value in manifest.items() if key != 'entries'}

def merge_shards(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    root = cache_root(cfg, dataset)
    graphs: list[dict] = []
    totals = {'graphs': 0, 'nodes': 0, 'include': 0, 'valid3d': 0, 'failed3d': 0, 'edges': 0}
    for shard in sorted(root.glob('shard_*')):
        manifest_path = shard / 'manifest.json'
        if not manifest_path.is_file():
            raise RuntimeError(f'incomplete shard: {manifest_path}')
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS':
            raise RuntimeError(f'shard not PASS: {manifest_path}')
        graphs.extend(payload['entries'])
        for key in totals:
            if key == 'graphs':
                continue
            totals[key] += int(payload.get(key, 0))
    graphs.sort(key=lambda row: row['graph_id'])
    totals['graphs'] = len(graphs)
    failed_rate = totals['failed3d'] / totals['include'] if totals['include'] else 0.0
    index = {'status': 'PASS', 'dataset': dataset, 'batch': batch_name(cfg), 'failed3d_rate': failed_rate, **totals, 'graphs': graphs}
    atomic_json(cache_index_path(cfg, dataset), index)
    return {'status': 'PASS', 'dataset': dataset, 'batch': batch_name(cfg), 'graphs': totals['graphs'], 'failed3d_rate': failed_rate}

def check_cache(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    from .features import EDGE_2D, EDGE_3D, NODE_2D, NODE_3D
    index = json.loads(cache_index_path(cfg, dataset).read_text(encoding='utf-8'))
    if str(index.get('batch')) != batch_name(cfg):
        raise RuntimeError(f"scene_index batch {index.get('batch')!r} != {batch_name(cfg)!r}")
    expected = 0
    for shard in sorted((Path(cfg['paths']['set_encoding_data_root']) / dataset / '02_projection_scene_selected_scene').glob('shard_*')):
        expected += len(list(shard.glob('*.pt')))
    problems = []
    ids = [row['graph_id'] for row in index['graphs']]
    if len(ids) != len(set(ids)):
        problems.append({'reason': 'duplicate graph_id'})
    if len(index['graphs']) != expected:
        problems.append({'reason': f"coverage {len(index['graphs'])} != selected-scene files {expected}"})
    for row in index['graphs']:
        arrays = np.load(row['path'])
        if arrays['node2d'].shape != (row['nodes'], NODE_2D):
            problems.append({'graph_id': row['graph_id'], 'reason': 'node2d shape'})
        if arrays['node3d'].shape != (row['nodes'], NODE_3D):
            problems.append({'graph_id': row['graph_id'], 'reason': 'node3d shape'})
        if arrays['edge2d'].shape != (row['edges'], EDGE_2D) or arrays['edge3d'].shape != (row['edges'], EDGE_3D):
            problems.append({'graph_id': row['graph_id'], 'reason': 'edge shape'})
        if arrays['edge_index'].shape != (2, row['edges']):
            problems.append({'graph_id': row['graph_id'], 'reason': 'edge_index shape'})
        if int(arrays['include'].sum()) != int(row['include']):
            problems.append({'graph_id': row['graph_id'], 'reason': 'include count'})
        if str(Path(row['path']).as_posix()).find('/expert_set_encoding/') >= 0:
            problems.append({'graph_id': row['graph_id'], 'reason': 'path still points at expert_set_encoding'})
    return {'status': 'FAIL' if problems else 'PASS', 'dataset': dataset, 'batch': batch_name(cfg), 'graphs': len(index['graphs']), 'scene_files': expected, 'failed3d_rate': index.get('failed3d_rate'), 'problem_count': len(problems), 'problems': problems[:20], 'official_test_touched': False}
