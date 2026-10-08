from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
import numpy as np
from .adapter import PublicRow, cache_many, normalize_manifest_row
from .feature_store import FoldModalStore, MeanPoolStore, build_modal_file, build_shuffle_for_fold, read_index, write_meanpool_cache
from .inference import infer_items, load_model
from .inventory import require_all_oof
from .io_utils import atomic_json, atomic_npz, atomic_parquet, safe_component, sha256_file
from .protocol import ARMS, DATASET_CONTRACTS, PROTOCOL_ID, freeze_protocol, prepare_tree
from .residual import crossfit_residual
from .training import train_expert
from .upstream import load_projection_scene, locked_hashes

def load_config(path: str | Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('config/protocol mismatch')
    return cfg

def _read(path: str | Path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns, partitioning=None).to_pylist()

def _row_id(row: Mapping[str, Any]) -> str:
    for key in ('graph_id', 'patch_id', 'tile_id'):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError('manifest row has no stable ID')

def all_dataset_rows(cfg: Mapping[str, Any], include_labels: bool=True) -> list[PublicRow]:
    import pyarrow.parquet as pq
    merged: dict[str, dict[str, Any]] = {}
    label_columns = {'label_id', 'label_7_id', 'label_7', 'label_3', 'target', 'label_name', 'label_3_name'}
    identity_columns = {'official_split', 'validation_fold', 'final_validation_fold'}
    for source in cfg['paths']['manifest_sources']:
        schema = pq.read_schema(source)
        columns = None if include_labels else [name for name in schema.names if name not in label_columns]
        for row in _read(source, columns=columns):
            target = merged.setdefault(_row_id(row), {})
            for key, value in row.items():
                if key in identity_columns and target.get(key) is not None:
                    continue
                target[key] = value
    eligible_path = cfg['paths']['eligible_graph_manifest']
    schema = pq.read_schema(eligible_path)
    id_column = next((name for name in ('graph_id', 'patch_id', 'tile_id') if name in schema.names))
    eligible = {_row_id(row) for row in _read(eligible_path, columns=[id_column])}
    rows = [normalize_manifest_row(row, cfg['dataset']) for graph_id, row in merged.items() if graph_id in eligible]
    return sorted(rows, key=lambda row: row.graph_id)

def dataset_rows(cfg: Mapping[str, Any], include_test: bool=False, include_labels: bool | None=None) -> list[PublicRow]:
    if include_labels is None:
        include_labels = not include_test
    rows = all_dataset_rows(cfg, include_labels=include_labels)
    if include_test:
        allowed = {'test'}
    else:
        allowed = {'train', 'val', 'validation'} if cfg['dataset'] == 'bracs' else {'train'}
    return [row for row in rows if row.split in allowed]

def run_lock(cfg: Mapping[str, Any]) -> dict:
    data_root = prepare_tree(cfg['paths']['data_root'])
    result_root = prepare_tree(cfg['paths']['result_root'])
    hashes = locked_hashes(cfg['paths']['projection_scene_source_root'], cfg['paths']['projection_scene_checkpoint'], cfg['paths']['projection_scene_input_manifest'], cfg['paths']['dino_source_root'], cfg['paths']['dino_weight_path'])
    hashes.update({key: str(value) for key, value in cfg.get('locked_provenance', {}).items()})
    manifests = {str(path): sha256_file(path) for path in cfg['paths']['manifest_sources']}
    rows = all_dataset_rows(cfg, include_labels=False)
    split_counts = dict(sorted(Counter((row.split for row in rows)).items()))
    expected_total = DATASET_CONTRACTS[cfg['dataset']]['expected_graphs']
    if len(rows) != expected_total:
        raise RuntimeError(f'eligible graph count mismatch: {len(rows)} != {expected_total}')
    payload = {'status': 'LOCKED', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'hashes': hashes, 'input_manifests': manifests, 'eligible_graphs': len(rows), 'split_counts': split_counts, 'expected': DATASET_CONTRACTS[cfg['dataset']]}
    atomic_json(data_root / '00_manifest' / 'lock.json', payload)
    atomic_json(result_root / '00_manifest' / 'lock.json', payload)
    return payload

def _reader(cfg, modules):
    graph = cfg['paths']['graph_cache']
    return modules.cache_io.ShardedLMDBReader(graph['root'], graph.get('prefix', 'graph_cache'), int(graph.get('shards', 64)))

def run_cache_inputs(cfg: Mapping[str, Any], shard: int, shards: int, device: str, include_test: bool=False) -> dict:
    import torch
    cpu_threads = int(cfg.get('runtime', {}).get('cpu_threads', 8))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    modules = load_projection_scene(cfg['paths']['projection_scene_source_root'])
    reader = _reader(cfg, modules)
    encoder = modules.he_features.DINOv2S14Encoder(Path(cfg['paths']['dino_source_root']), Path(cfg['paths']['dino_weight_path'])).to(device).eval()
    rows = dataset_rows(cfg, include_test=include_test, include_labels=not include_test)
    if include_test and any((row.label_id is not None for row in rows)):
        raise RuntimeError('TEST cache adapter unexpectedly received labels')
    output = []
    root = Path(cfg['paths']['data_root']) / ('09_official_test/01_projection_scene_inputs' if include_test else '01_projection_scene_inputs')
    selected = rows[shard::shards]
    batch_size = int(cfg.get('runtime', {}).get('dino_batch_size', 4))
    for start in range(0, len(selected), batch_size):
        chunk = selected[start:start + batch_size]
        destinations = [root / f'{safe_component(row.graph_id)}.pt' for row in chunk]
        missing = [(row, path) for row, path in zip(chunk, destinations) if not path.is_file()]
        for row, path in zip(chunk, destinations):
            if path.is_file():
                output.append({'graph_id': row.graph_id, 'path': str(path), 'reused': True})
        if missing:
            output.extend(cache_many(modules, encoder, reader, [value[0] for value in missing], [value[1] for value in missing], cfg['dataset'], float(cfg['resolution']['mpp']), device))
        if (start // batch_size + 1) % 16 == 0:
            atomic_json(root / f'progress_shard_{shard:03d}.json', {'status': 'RUNNING', 'done': min(start + batch_size, len(selected)), 'total': len(selected)})
    reader.close()
    atomic_json(root / f'progress_shard_{shard:03d}.json', {'status': 'PASS', 'done': len(selected), 'total': len(selected)})
    atomic_json(root / f'manifest_shard_{shard:03d}.json', {'status': 'PASS', 'shard': shard, 'shards': shards, 'items': output})
    return {'status': 'PASS', 'items': len(output), 'official_test': include_test}

def finalize_shards(root: str | Path, pattern: str, destination: str | Path) -> dict:
    manifests = sorted(Path(root).glob(pattern))
    items = []
    declared_shards = set()
    expected_shards = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS':
            raise RuntimeError(f'incomplete shard manifest: {path}')
        declared_shards.add(int(payload['shard']))
        expected_shards.update(range(int(payload['shards'])))
        items.extend(payload['items'])
    if declared_shards != expected_shards:
        raise RuntimeError(f'shard manifest coverage mismatch: {declared_shards} != {expected_shards}')
    if len({row['graph_id'] for row in items}) != len(items):
        raise RuntimeError('duplicate graph in shard manifests')
    payload = {'status': 'PASS', 'graphs': len(items), 'items': sorted(items, key=lambda row: row['graph_id'])}
    atomic_json(destination, payload)
    return payload

def run_infer(cfg: Mapping[str, Any], shard: int, shards: int, device: str, include_test: bool=False) -> dict:
    import torch
    modules = load_projection_scene(cfg['paths']['projection_scene_source_root'])
    model, _ = load_model(modules, cfg['paths']['projection_scene_checkpoint'], cfg['paths']['projection_scene_input_manifest'], device)
    input_root = Path(cfg['paths']['data_root']) / ('09_official_test/01_projection_scene_inputs' if include_test else '01_projection_scene_inputs')
    input_manifest = json.loads((input_root / 'manifest.json').read_text(encoding='utf-8'))
    selected = input_manifest['items'][shard::shards]

    def items():
        for row in selected:
            payload = torch.load(row['path'], map_location='cpu', weights_only=False)
            yield (payload['graph'], payload['metadata'])
    output_root = Path(cfg['paths']['data_root']) / ('09_official_test/02_projection_scene_selected_scene' if include_test else '02_projection_scene_selected_scene') / f'shard_{shard:03d}'
    values = infer_items(modules, model, items(), output_root, device, total=len(selected), progress_path=output_root.parent / f'progress_shard_{shard:03d}.json')
    return {'status': 'PASS', 'items': len(values), 'official_test': include_test}

def run_modal(cfg: Mapping[str, Any], shard: int, shards: int, include_test: bool=False) -> dict:
    data_root = Path(cfg['paths']['data_root'])
    load_projection_scene(cfg['paths']['projection_scene_source_root'])
    input_root = data_root / ('09_official_test/01_projection_scene_inputs' if include_test else '01_projection_scene_inputs')
    scene_root = data_root / ('09_official_test/02_projection_scene_selected_scene' if include_test else '02_projection_scene_selected_scene')
    modal_root = data_root / ('09_official_test/03_modal_features' if include_test else '03_modal_features')
    inputs = json.loads((input_root / 'manifest.json').read_text(encoding='utf-8'))['items']
    scene_items = []
    for manifest in scene_root.glob('shard_*/manifest.json'):
        scene_items.extend(json.loads(manifest.read_text(encoding='utf-8'))['items'])
    scenes = {row['graph_id']: row['path'] for row in scene_items}
    output = []
    for row in inputs[shard::shards]:
        if row['graph_id'] not in scenes:
            raise RuntimeError(f"selected scene missing for {row['graph_id']}")
        destination = modal_root / f"{safe_component(row['graph_id'])}.npz"
        output.append(build_modal_file(row['path'], scenes[row['graph_id']], destination))
    index_path = modal_root / f'index_shard_{shard:03d}.parquet'
    atomic_parquet(index_path, output)
    payload = {'status': 'PASS', 'items': len(output), 'official_test': include_test, 'index': str(index_path)}
    atomic_json(modal_root / f'index_shard_{shard:03d}.json', payload)
    return payload

def finalize_modal(cfg: Mapping[str, Any], include_test: bool=False) -> dict:
    root = Path(cfg['paths']['data_root']) / ('09_official_test/03_modal_features' if include_test else '03_modal_features')
    rows = []
    for path in sorted(root.glob('index_shard_*.parquet')):
        rows.extend(_read(path))
    expected = dataset_rows(cfg, include_test=include_test)
    expected_ids = {row.graph_id for row in expected}
    observed_ids = {str(row['graph_id']) for row in rows}
    if len(rows) != len(observed_ids) or observed_ids != expected_ids:
        raise RuntimeError(f'modal coverage mismatch: rows={len(rows)} unique={len(observed_ids)} expected={len(expected_ids)}')
    _enforce_modal_identity(rows, expected)
    atomic_parquet(root / 'index.parquet', sorted(rows, key=lambda row: row['graph_id']))
    payload = {'status': 'PASS', 'graphs': len(rows), 'anchors': int(sum((row['count'] for row in rows))), 'invalid_ncr': int(sum((row['invalid_ncr'] for row in rows))), 'official_test': include_test}
    atomic_json(root / 'manifest.json', payload)
    return payload

def _enforce_modal_identity(rows: list[dict[str, Any]], expected: list[PublicRow]) -> None:
    expected_by_id = {row.graph_id: row for row in expected}
    for row in rows:
        expected_row = expected_by_id[str(row['graph_id'])]
        row['split'] = expected_row.split
        row['fold'] = expected_row.fold
        row['final_fold'] = expected_row.final_fold

def run_shuffle(cfg: Mapping[str, Any], fold: int) -> dict:
    root = Path(cfg['paths']['data_root'])
    rows = read_index(root / '03_modal_features/index.parquet')
    mapping = build_shuffle_for_fold(rows, fold, cfg['dataset'])
    destination = root / f'04_shuffle_maps/fold_{fold:02d}/mapping.parquet'
    atomic_parquet(destination, [{'graph_id': key, 'donor_graph_id': value} for key, value in sorted(mapping.items())])
    payload = {'status': 'PASS', 'fold': fold, 'graphs': len(mapping), 'sha256': sha256_file(destination)}
    atomic_json(destination.with_suffix('.json'), payload)
    return payload

def _load_mapping(path):
    return {row['graph_id']: row['donor_graph_id'] for row in _read(path)}

def run_residual(cfg: Mapping[str, Any], fold: int, device: str) -> dict:
    root = Path(cfg['paths']['data_root'])
    rows = read_index(root / '03_modal_features/index.parquet')
    arrays = []
    for row in rows:
        with np.load(row.path) as value:
            arrays.append({key: value[key].copy() for key in value.files})
    counts = np.asarray([len(value['rays36']) for value in arrays])
    graph_code = np.repeat(np.arange(len(rows)), counts)
    rays = np.concatenate([value['rays36'] for value in arrays])
    direct = np.concatenate([value['direct9'] for value in arrays])
    valid = np.concatenate([value['valid_ncr'] for value in arrays])
    groups = np.repeat([row.wsi_id if cfg['dataset'] == 'bracs' else row.patient_id for row in rows], counts)
    train_graph = np.asarray([row.split == 'train' and row.fold != fold for row in rows])
    held_graph = np.asarray([row.split == 'train' and row.fold == fold for row in rows])
    train = np.repeat(train_graph, counts)
    held = np.repeat(held_graph, counts)
    external_graph = np.asarray([cfg['dataset'] == 'bracs' and row.split in {'val', 'validation'} for row in rows])
    external = np.repeat(external_graph, counts)
    destination_root = root / f'03_modal_features/residual/fold_{fold:02d}'
    global_path = destination_root / 'all_residuals.npz'
    manifest = crossfit_residual(rays36=rays, direct9=direct, valid_ncr=valid, graph_code=graph_code, groups=groups, outer_train=train, outer_heldout=held, external=external, conditional_geometry_root=cfg['paths']['mask_probe_conditional_geometry_root'], destination=global_path, device=device)
    with np.load(global_path) as value:
        residual = value['residual9']
        written = value['written']
    cursor = 0
    anchor_offset = 0
    for row, count in zip(rows, counts):
        selected = written[anchor_offset:anchor_offset + count]
        if selected.all():
            atomic_npz(destination_root / f'{safe_component(row.graph_id)}.npz', residual9=residual[cursor:cursor + count])
            cursor += count
        elif selected.any():
            raise RuntimeError('partial graph residual coverage')
        anchor_offset += count
    if cursor != len(residual):
        raise RuntimeError('residual graph split coverage mismatch')
    atomic_json(destination_root / 'all_residuals.json', manifest)
    return manifest

def _fold_store(cfg: Mapping[str, Any], fold: int):
    root = Path(cfg['paths']['data_root'])
    rows = read_index(root / '03_modal_features/index.parquet')
    mapping = _load_mapping(root / f'04_shuffle_maps/fold_{fold:02d}/mapping.parquet')
    store = FoldModalStore(rows, fold, dataset=cfg['dataset'], residual_root=root / f'03_modal_features/residual/fold_{fold:02d}', donor_map=mapping)
    return (root, rows, mapping, store)

def run_meanpool(cfg: Mapping[str, Any], fold: int, arm: str) -> dict:
    root, rows, _, store = _fold_store(cfg, fold)
    return write_meanpool_cache(store, rows, arm, root / f'05_meanpool_cache/fold_{fold:02d}/{arm}')

def run_meanpool_all(cfg: Mapping[str, Any], fold: int) -> dict:
    root, rows, _, store = _fold_store(cfg, fold)
    arms = {}
    for arm in ('D', 'DS', 'R', 'RS'):
        arms[arm] = write_meanpool_cache(store, rows, arm, root / f'05_meanpool_cache/fold_{fold:02d}/{arm}')
    payload = {'status': 'PASS', 'fold': fold, 'arms': arms}
    atomic_json(root / f'05_meanpool_cache/fold_{fold:02d}/manifest.json', payload)
    return payload

def run_train(cfg: Mapping[str, Any], fold: int, arm: str, encoder: str, seed: int, device: str) -> dict:
    root = Path(cfg['paths']['data_root'])
    result = Path(cfg['paths']['result_root'])
    rows = read_index(root / '03_modal_features/index.parquet')
    mapping = _load_mapping(root / f'04_shuffle_maps/fold_{fold:02d}/mapping.parquet')
    if encoder == 'meanpool':
        store = MeanPoolStore(root / f'05_meanpool_cache/fold_{fold:02d}/{arm}/manifest.json')
    else:
        store = FoldModalStore(rows, fold, dataset=cfg['dataset'], residual_root=root / f'03_modal_features/residual/fold_{fold:02d}', donor_map=mapping)
    return train_expert(dataset=cfg['dataset'], rows=rows, store=store, fold=fold, arm=arm, encoder=encoder, seed=seed, destination=result / f'06_experts/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}', device=device, **cfg.get('training', {}))

def run_freeze(config_paths: Mapping[str, str | Path], result_root: str | Path) -> dict:
    configs = {name: load_config(path) for name, path in config_paths.items()}
    inventory = require_all_oof(configs)
    hashes = {}
    manifests = {}
    for name, cfg in configs.items():
        lock = json.loads((Path(cfg['paths']['data_root']) / '00_manifest/lock.json').read_text(encoding='utf-8'))
        hashes.update({f'{name}:{key}': value for key, value in lock['hashes'].items()})
        manifests.update({f'{name}:{key}': value for key, value in lock['input_manifests'].items()})
        manifests[f'{name}:config_sha256'] = sha256_file(config_paths[name])
    payload = freeze_protocol(result_root, hashes=hashes, manifests=manifests, config_paths={key: str(value) for key, value in config_paths.items()}, inventory=inventory)
    return payload
