from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
import numpy as np
from .dataset import DINO_DIM, cache_batch, cache_index_path, cache_root
from .features import assemble_graph, extract_dino, resolve_observed_xy
from .foundation.spatial import encode_spatial_batch, load_encoder, resolve_spatial, save_spatial, spatial_path
from .io_utils import atomic_json, atomic_npz, load_npz, safe_component
SPATIAL_ENCODE_BATCH = 8

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

def _observed_and_meta(path: str, nucleus_ids: np.ndarray):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    graph = payload['graph']
    meta = dict(payload.get('metadata') or {})
    ids = np.asarray(graph.nucleus_id).reshape(-1).astype(np.int64)
    fitted = np.asarray(graph.fitted_center_xy, np.float64)
    fallback = np.asarray(graph.nucleus_xy_um, np.float64)
    dino = extract_dino(graph)
    del payload
    if not np.array_equal(ids, nucleus_ids):
        raise RuntimeError(f'input nucleus_id does not match selected-scene for {path}')
    return (resolve_observed_xy(fitted, fallback), ids, dino, meta)

def _stats_from_arrays(arrays: Mapping[str, Any]) -> dict[str, int]:
    include = np.asarray(arrays['include'], bool)
    valid3d = np.asarray(arrays['valid3d'], bool)
    return {'nodes': int(include.shape[0]), 'include': int(include.sum()), 'valid3d': int((include & valid3d).sum()), 'failed3d': int((include & ~valid3d).sum()), 'edges': int(np.asarray(arrays['edge_index']).shape[1])}

def _flush_spatial(encoder, pending, device, totals):
    if not pending or encoder is None:
        return []
    images = []
    for item in pending:
        image = item[0].result() if hasattr(item[0], 'result') else item[0]
        images.append(image)
    fields = encode_spatial_batch(encoder, images, device)
    for (_, dest), field in zip(pending, fields):
        save_spatial(dest, field)
    totals['spatial'] += len(pending)
    return []

def cache_shard(cfg: Mapping[str, Any], dataset: str, shard: int, device: str='cpu') -> dict[str, Any]:
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from celllift.matched_geometry_controls.dino import load_rgb
    destination = cache_root(cfg, dataset) / f'shard_{shard:03d}'
    destination.mkdir(parents=True, exist_ok=True)
    lookup = modal_lookup(cfg, dataset)
    inputs = input_lookup(cfg, dataset)
    files = _scene_files(cfg, dataset, shard)
    encoder = None
    if str(device).startswith('cuda'):
        try:
            encoder = load_encoder(cfg, device)
        except Exception:
            encoder = None
    entries = []
    totals = {'nodes': 0, 'include': 0, 'valid3d': 0, 'failed3d': 0, 'edges': 0, 'spatial': 0}
    pending_rgb = []
    rgb_pool = ThreadPoolExecutor(max_workers=2)

    def queue_rgb(rgb_path, spatial_file):
        nonlocal pending_rgb
        pending_rgb.append((rgb_pool.submit(load_rgb, rgb_path, dataset), spatial_file))
        if len(pending_rgb) >= SPATIAL_ENCODE_BATCH:
            pending_rgb = _flush_spatial(encoder, pending_rgb, device, totals)
    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        graph_id = str(payload.get('graph_id') or (payload.get('metadata') or {}).get('graph_id') or path.stem)
        payload['graph_id'] = graph_id
        target = destination / f'{safe_component(graph_id)}.npz'
        spatial_file = spatial_path(target)
        existing_spatial = resolve_spatial(spatial_file) or resolve_spatial(target)
        modal_path = lookup.get(graph_id)
        input_path = inputs.get(graph_id)
        if modal_path is None or input_path is None:
            raise RuntimeError(f'missing modal/input for {graph_id}')
        if target.is_file():
            arrays = load_npz(target)
            stats = _stats_from_arrays(arrays)
            meta = dict(payload.get('metadata') or {})
            rgb_path = meta.get('rgb_path')
            mask_path = meta.get('nucleus_mask_path')
            if not existing_spatial and (not rgb_path):
                _, _, _, input_meta = _observed_and_meta(input_path, np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64))
                meta = {**input_meta, **meta}
                rgb_path = meta.get('rgb_path')
                mask_path = meta.get('nucleus_mask_path')
            spatial_ok = existing_spatial is not None
            if spatial_ok:
                totals['spatial'] += 1
            elif encoder is not None and rgb_path and Path(rgb_path).is_file():
                queue_rgb(rgb_path, spatial_file)
                spatial_ok = True
            for key in ('nodes', 'include', 'valid3d', 'failed3d', 'edges'):
                totals[key] += stats[key]
            entries.append({'graph_id': graph_id, 'path': str(target), 'spatial_path': str(existing_spatial or spatial_file) if spatial_ok else None, 'rgb_path': rgb_path, 'mask_path': mask_path, 'metadata': meta, **stats})
            continue
        modal = load_npz(modal_path)
        nucleus_ids = np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64)
        observed_xy, observed_ids, dino, meta = _observed_and_meta(input_path, nucleus_ids)
        if payload.get('metadata'):
            meta = {**meta, **dict(payload['metadata'])}
        payload['metadata'] = meta
        graph = assemble_graph(payload, modal['rays36'], modal['nucleus_id'], observed_xy, observed_ids, dino)
        atomic_npz(target, dino=graph['dino'], node2d=graph['node2d'], node3d=graph['node3d'], include=graph['include'], valid3d=graph['valid3d'], center_xy=graph['center_xy'], center_z=graph['center_z'], nucleus_center=graph['nucleus_center'], cell_center=graph['cell_center'], nucleus_transform=graph['nucleus_transform'], cell_transform=graph['cell_transform'], edge_index=graph['edge_index'], edge2d=graph['edge2d'], edge3d=graph['edge3d'])
        rgb_path = meta.get('rgb_path')
        mask_path = meta.get('nucleus_mask_path')
        spatial_ok = existing_spatial is not None
        if spatial_ok:
            totals['spatial'] += 1
        elif encoder is not None and rgb_path and Path(rgb_path).is_file():
            queue_rgb(rgb_path, spatial_file)
            spatial_ok = True
        stats = graph['stats']
        for key in ('nodes', 'include', 'valid3d', 'failed3d', 'edges'):
            totals[key] += stats[key]
        entries.append({'graph_id': graph['graph_id'], 'path': str(target), 'spatial_path': str(existing_spatial or spatial_file) if spatial_ok else None, 'rgb_path': rgb_path, 'mask_path': mask_path, 'metadata': graph['metadata'], **stats})
    pending_rgb = _flush_spatial(encoder, pending_rgb, device, totals)
    rgb_pool.shutdown(wait=False)
    manifest = {'status': 'PASS', 'dataset': dataset, 'batch': cache_batch(cfg), 'shard': int(shard), 'graphs': len(entries), 'dino_dim': int(DINO_DIM), **totals, 'entries': entries}
    atomic_json(destination / 'manifest.json', manifest)
    return {key: value for key, value in manifest.items() if key != 'entries'}

def merge_shards(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    root = cache_root(cfg, dataset)
    graphs: list[dict] = []
    totals = {'graphs': 0, 'nodes': 0, 'include': 0, 'valid3d': 0, 'failed3d': 0, 'edges': 0, 'spatial': 0}
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
    index = {'status': 'PASS', 'dataset': dataset, 'batch': cache_batch(cfg), 'dino_dim': int(DINO_DIM), 'failed3d_rate': failed_rate, **totals, 'graphs': graphs}
    atomic_json(cache_index_path(cfg, dataset), index)
    return {'status': 'PASS', 'dataset': dataset, 'batch': cache_batch(cfg), 'graphs': totals['graphs'], 'failed3d_rate': failed_rate, 'spatial': totals['spatial'], 'dino_dim': int(DINO_DIM)}

def check_cache(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    from .features import EDGE_2D, EDGE_3D, NODE_2D, NODE_3D
    index = json.loads(cache_index_path(cfg, dataset).read_text(encoding='utf-8'))
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
        posix = str(Path(row['path']).as_posix())
        if '/parallel_set_encoding/' not in posix:
            problems.append({'graph_id': row['graph_id'], 'reason': 'path does not contain parallel_set_encoding'})
        arrays = np.load(row['path'])
        if arrays['dino'].shape != (row['nodes'], DINO_DIM):
            problems.append({'graph_id': row['graph_id'], 'reason': 'dino shape'})
        if arrays['node2d'].shape != (row['nodes'], NODE_2D) or arrays['node3d'].shape != (row['nodes'], NODE_3D):
            problems.append({'graph_id': row['graph_id'], 'reason': 'node shape'})
        if arrays['nucleus_transform'].shape != (row['nodes'], 3, 3):
            problems.append({'graph_id': row['graph_id'], 'reason': 'transform shape'})
        if arrays['edge2d'].shape != (row['edges'], EDGE_2D) or arrays['edge3d'].shape != (row['edges'], EDGE_3D):
            problems.append({'graph_id': row['graph_id'], 'reason': 'edge shape'})
        if int(arrays['include'].sum()) != int(row['include']):
            problems.append({'graph_id': row['graph_id'], 'reason': 'include count'})
    return {'status': 'FAIL' if problems else 'PASS', 'dataset': dataset, 'batch': cache_batch(cfg), 'graphs': len(index['graphs']), 'scene_files': expected, 'failed3d_rate': index.get('failed3d_rate'), 'spatial': index.get('spatial'), 'problem_count': len(problems), 'problems': problems[:20], 'official_test_touched': False}
