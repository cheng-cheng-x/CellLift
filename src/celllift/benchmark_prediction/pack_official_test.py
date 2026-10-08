from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.benchmark_prediction.paths import load_config, split_root
from celllift.morphology_interaction.cache import _observed_and_meta, _stats_from_arrays
from celllift.morphology_interaction.dataset import cache_index_path, cache_root
from celllift.morphology_interaction.features import assemble_graph
from celllift.morphology_interaction.io_utils import atomic_json, atomic_npz, load_npz, safe_component

def _test_root(cfg: dict[str, Any], dataset: str) -> Path:
    return Path(cfg['paths']['set_encoding_data_root']) / dataset / '09_official_test'

def _labels(dataset: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    path = split_root() / dataset / 'test_rows.json'
    for row in json.loads(path.read_text(encoding='utf-8')):
        lid = row.get('label_id')
        if lid in (None, ''):
            continue
        lid = int(lid)
        labels[str(row['sample_id'])] = lid
        if row.get('graph_id'):
            labels[str(row['graph_id'])] = lid
        if row.get('patient_id'):
            labels[str(row['patient_id'])] = lid
        if row.get('roi_id'):
            labels[str(row['roi_id'])] = lid
    return labels

def _lookups(cfg: dict[str, Any], dataset: str) -> tuple[dict[str, str], dict[str, str]]:
    import pyarrow.parquet as pq
    root = _test_root(cfg, dataset)
    modal = {}
    index = root / '03_modal_features' / 'index.parquet'
    for row in pq.read_table(index, partitioning=None).to_pylist():
        raw = Path(str(row['path']))
        modal[str(row['graph_id'])] = str(raw if raw.is_absolute() else index.parent / raw)
    inputs = {}
    payload = json.loads((root / '01_projection_scene_inputs' / 'manifest.json').read_text(encoding='utf-8'))
    for row in payload['items']:
        raw = Path(str(row['path']))
        inputs[str(row['graph_id'])] = str(raw if raw.is_absolute() else root / '01_projection_scene_inputs' / raw)
    return (modal, inputs)

def pack_shard(dataset: str, shard: int, device: str='cpu', with_spatial: bool=False) -> dict[str, Any]:
    from celllift.morphology_interaction.foundation.spatial import encode_spatial_batch, load_encoder, resolve_spatial, save_spatial, spatial_path
    cfg = load_config(dataset)
    from celllift.matched_geometry_controls.upstream import load_projection_scene
    load_projection_scene(cfg['paths']['projection_scene_source_root'])
    labels = _labels(dataset)
    modal, inputs = _lookups(cfg, dataset)
    scene_dir = _test_root(cfg, dataset) / '02_projection_scene_selected_scene' / f'shard_{int(shard):03d}'
    files = sorted(scene_dir.glob('*.pt'))
    if not files:
        return {'status': 'SKIP', 'dataset': dataset, 'shard': shard, 'graphs': 0}
    destination = cache_root(cfg, dataset) / 'official_test' / f'shard_{int(shard):03d}'
    destination.mkdir(parents=True, exist_ok=True)
    encoder = None
    if with_spatial and str(device).startswith('cuda'):
        try:
            encoder = load_encoder(cfg, device)
        except Exception:
            encoder = None
    entries = []
    totals = {'nodes': 0, 'include': 0, 'valid3d': 0, 'failed3d': 0, 'edges': 0, 'spatial': 0}
    import torch
    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        graph_id = str(payload.get('graph_id') or (payload.get('metadata') or {}).get('graph_id') or path.stem)
        payload['graph_id'] = graph_id
        target = destination / f'{safe_component(graph_id)}.npz'
        spatial_file = spatial_path(target)
        existing_spatial = resolve_spatial(spatial_file) or resolve_spatial(target)
        modal_path = modal.get(graph_id)
        input_path = inputs.get(graph_id)
        if modal_path is None or input_path is None:
            raise RuntimeError(f'missing modal/input for {graph_id}')
        if target.is_file():
            arrays = load_npz(target)
            stats = _stats_from_arrays(arrays)
            meta = dict(payload.get('metadata') or {})
        else:
            nucleus_ids = np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64)
            observed_xy, observed_ids, dino, meta = _observed_and_meta(input_path, nucleus_ids)
            if payload.get('metadata'):
                meta = {**meta, **dict(payload['metadata'])}
            payload['metadata'] = meta
            modal_npz = load_npz(modal_path)
            graph = assemble_graph(payload, modal_npz['rays36'], modal_npz['nucleus_id'], observed_xy, observed_ids, dino)
            atomic_npz(target, dino=graph['dino'], node2d=graph['node2d'], node3d=graph['node3d'], include=graph['include'], valid3d=graph['valid3d'], center_xy=graph['center_xy'], center_z=graph['center_z'], nucleus_center=graph['nucleus_center'], cell_center=graph['cell_center'], nucleus_transform=graph['nucleus_transform'], cell_transform=graph['cell_transform'], edge_index=graph['edge_index'], edge2d=graph['edge2d'], edge3d=graph['edge3d'])
            stats = graph['stats']
            meta = graph['metadata']
        for key in ('patient_id', 'roi_id', 'wsi_id'):
            if meta.get(key) in (None, '') and (payload.get('metadata') or {}).get(key):
                meta[key] = (payload.get('metadata') or {}).get(key)
        lid = labels.get(graph_id) or labels.get(str(meta.get('roi_id') or '')) or labels.get(str(meta.get('patient_id') or ''))
        if lid is not None:
            meta['label_id'] = int(lid)
        meta['official_split'] = 'test'
        meta['split'] = 'test'
        rgb_path = meta.get('rgb_path')
        spatial_ok = existing_spatial is not None
        if not spatial_ok and encoder is not None and rgb_path and Path(str(rgb_path)).is_file():
            from celllift.matched_geometry_controls.dino import load_rgb
            image = load_rgb(rgb_path, dataset)
            field = encode_spatial_batch(encoder, [image], device)[0]
            save_spatial(spatial_file, field)
            spatial_ok = True
            existing_spatial = spatial_file
        if spatial_ok:
            totals['spatial'] += 1
        for key in ('nodes', 'include', 'valid3d', 'failed3d', 'edges'):
            totals[key] += stats[key]
        entries.append({'graph_id': graph_id, 'path': str(target), 'spatial_path': str(existing_spatial or spatial_file) if spatial_ok else None, 'rgb_path': rgb_path, 'mask_path': meta.get('nucleus_mask_path'), 'metadata': meta, **stats})
    manifest = {'status': 'PASS', 'dataset': dataset, 'shard': int(shard), 'graphs': len(entries), 'official_test': True, **totals, 'entries': entries}
    atomic_json(destination / 'manifest.json', manifest)
    return {k: v for k, v in manifest.items() if k != 'entries'}

def merge_test(dataset: str) -> dict[str, Any]:
    cfg = load_config(dataset)
    root = cache_root(cfg, dataset) / 'official_test'
    extra = []
    for shard in sorted(root.glob('shard_*')):
        payload = json.loads((shard / 'manifest.json').read_text(encoding='utf-8'))
        extra.extend(payload['entries'])
    index_path = cache_index_path(cfg, dataset)
    payload = json.loads(index_path.read_text(encoding='utf-8'))
    existing = {str(row['graph_id']) for row in payload['graphs']}
    added = 0
    for row in extra:
        if str(row['graph_id']) in existing:
            continue
        payload['graphs'].append(row)
        existing.add(str(row['graph_id']))
        added += 1
    payload['graphs'].sort(key=lambda row: row['graph_id'])
    payload['graphs_official_test'] = sum((1 for row in payload['graphs'] if (row.get('metadata') or {}).get('official_split') == 'test' or (row.get('metadata') or {}).get('split') == 'test'))
    totals_graphs = len(payload['graphs'])
    payload['n_graphs'] = totals_graphs
    atomic_json(index_path, payload)
    return {'dataset': dataset, 'added': added, 'graphs_official_test': payload['graphs_official_test'], 'n_graphs': totals_graphs, 'index': str(index_path)}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pack-shard', 'merge'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--with-spatial', action='store_true')
    args = parser.parse_args()
    if args.stage == 'pack-shard':
        print(json.dumps(pack_shard(args.dataset, args.shard, args.device, args.with_spatial)))
        return 0
    print(json.dumps(merge_test(args.dataset)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
