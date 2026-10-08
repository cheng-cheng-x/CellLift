from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
_MODEL_INPUT = Path(__file__).resolve().parents[1]
_PUBLIC = _MODEL_INPUT.parent
for path in (_MODEL_INPUT, _PUBLIC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from .constants import ARVANITI_DATA, DINO_SOURCE, DINO_WEIGHTS, LIZARD_DATA, TARGET_MPP, ProjectionScene_CHECKPOINT, ProjectionScene_MANIFEST, ProjectionScene_SOURCE
from .graphs import GraphReader
from .io_utils import atomic_json, read_parquet

def _limit_threads() -> None:
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('TORCH_NUM_THREADS', '1')
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def _rows(dataset: str, *, engineering: bool, revision: str='set_encoding') -> list[dict[str, Any]]:
    root = Path(ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA)
    name = 'engineering_windows.parquet' if dataset == 'arvaniti' else 'engineering_tiles.parquet'
    if not engineering:
        if revision == 'conditional_geometry':
            name = 'inference_window_manifest_conditional_geometry.parquet' if dataset == 'arvaniti' else 'tile_manifest_conditional_geometry.parquet'
        else:
            name = 'window_manifest.parquet' if dataset == 'arvaniti' else 'tile_manifest.parquet'
    rows = read_parquet(root / '00_manifest' / name)
    for row in rows:
        row['rgb_path'] = row['dino_rgb_path']
        row['split'] = row.get('official_split', 'train')
        row['wsi_id'] = row.get('wsi_id') or row.get('core_id') or row.get('roi_id')
        row['roi_id'] = row.get('roi_id') or row['graph_id']
        row['patient_id'] = row.get('patient_id') or row['graph_id']
        if row.get('label_id') is None:
            row['label_id'] = -1
    return rows

def cache_projection_scene(dataset: str, *, engineering: bool=False, device: str='cuda', limit: int | None=None, shard_id: int=0, num_shards: int=1, revision: str='set_encoding') -> dict[str, Any]:
    _limit_threads()
    from celllift.matched_geometry_controls.adapter import cache_many, cache_one, normalize_manifest_row
    from celllift.matched_geometry_controls.upstream import load_projection_scene
    root = Path(ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA)
    graph_root = root / ('03_graph_cache_engineering' if engineering else '03_graph_cache_conditional_geometry' if revision == 'conditional_geometry' else '03_graph_cache')
    out_root = root / ('04_projection_scene_inputs_engineering' if engineering else '04_projection_scene_inputs_conditional_geometry' if revision == 'conditional_geometry' else '04_projection_scene_inputs')
    out_root.mkdir(parents=True, exist_ok=True)
    modules = load_projection_scene(ProjectionScene_SOURCE)
    encoder = modules.he_features.DINOv2S14Encoder(Path(DINO_SOURCE), Path(DINO_WEIGHTS)).to(device).eval()
    reader = GraphReader(graph_root)
    rows = _rows(dataset, engineering=engineering, revision=revision)
    rows = [row for row in rows if int(row.get('n_nuclei') or 0) > 0]
    rows = rows[int(shard_id)::int(num_shards)]
    if limit is not None:
        rows = rows[:limit]
    written = []
    failed = []
    batch_size = 4
    progress_path = out_root / f'cache_progress_{int(shard_id):03d}_of_{int(num_shards):03d}.json'

    def _flush_progress() -> None:
        payload = {'written': len(written), 'failed': len(failed), 'assigned': len(rows), 'root': str(out_root), 'shard_id': shard_id, 'num_shards': num_shards}
        atomic_json(progress_path, payload)
        print(json.dumps(payload), flush=True)

    def _cache_batch(pending_rows, pending_paths) -> None:
        try:
            written.extend(cache_many(modules, encoder, reader, pending_rows, pending_paths, dataset, TARGET_MPP, device))
            return
        except Exception as batch_exc:
            print(f'cache_many failed ({type(batch_exc).__name__}: {batch_exc}); falling back to cache_one', flush=True)
        for item, destination in zip(pending_rows, pending_paths):
            if destination.is_file():
                written.append({'graph_id': item.graph_id, 'path': str(destination), 'reused': True})
                continue
            try:
                written.append(cache_one(modules, encoder, reader, item, destination, dataset, TARGET_MPP, device))
            except Exception as exc:
                failed.append({'graph_id': item.graph_id, 'error': f'{type(exc).__name__}: {exc}'})
    try:
        pending_rows = []
        pending_paths = []
        for row in rows:
            destination = out_root / f"{row['graph_id']}.pt"
            if destination.is_file() and destination.stat().st_size > 0:
                written.append({'graph_id': row['graph_id'], 'path': str(destination), 'reused': True})
                continue
            pending_rows.append(normalize_manifest_row(row, dataset))
            pending_paths.append(destination)
            if len(pending_rows) >= batch_size:
                _cache_batch(pending_rows, pending_paths)
                pending_rows, pending_paths = ([], [])
                if len(written) % 16 == 0:
                    _flush_progress()
        if pending_rows:
            _cache_batch(pending_rows, pending_paths)
    finally:
        reader.close()
    payload = {'written': len(written), 'failed': failed, 'root': str(out_root), 'shard_id': shard_id, 'num_shards': num_shards}
    atomic_json(out_root / f'cache_status_{int(shard_id):03d}_of_{int(num_shards):03d}.json', payload)
    _flush_progress()
    return payload

def infer_projection_scene(dataset: str, *, engineering: bool=False, device: str='cuda', limit: int | None=None, shard_id: int=0, num_shards: int=1, revision: str='set_encoding') -> dict[str, Any]:
    _limit_threads()
    import torch
    from celllift.matched_geometry_controls.inference import infer_items, load_model
    from celllift.matched_geometry_controls.upstream import load_projection_scene, locked_hashes
    root = Path(ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA)
    in_root = root / ('04_projection_scene_inputs_engineering' if engineering else '04_projection_scene_inputs_conditional_geometry' if revision == 'conditional_geometry' else '04_projection_scene_inputs')
    scene_root = root / ('05_qc' if engineering else '04_projection_scene_inputs_conditional_geometry' if revision == 'conditional_geometry' else '04_projection_scene_inputs') / ('selected_scene_engineering' if engineering else 'selected_scene')
    scene_root.mkdir(parents=True, exist_ok=True)
    modules = load_projection_scene(ProjectionScene_SOURCE)
    hashes = locked_hashes(ProjectionScene_SOURCE, ProjectionScene_CHECKPOINT, ProjectionScene_MANIFEST, DINO_SOURCE, DINO_WEIGHTS)
    expected = '718d6445489266ec0708443262cae8b25aa2e458c963d11eaa9bb922aca06ee3'
    if hashes.get('projection_scene_checkpoint_sha256') != expected:
        raise RuntimeError(f"projection_scene checkpoint SHA mismatch: {hashes.get('projection_scene_checkpoint_sha256')}")
    model, info = load_model(modules, ProjectionScene_CHECKPOINT, ProjectionScene_MANIFEST, device)
    from celllift.matched_geometry_controls.io_utils import safe_component
    existing = [path for path in scene_root.glob('*.pt') if path.stat().st_size > 0]

    def _has_scene(stem: str) -> bool:
        return (scene_root / f'{stem}.pt').is_file() or (scene_root / f'{safe_component(stem)}.pt').is_file()
    paths = sorted((path for path in in_root.glob('*.pt') if not _has_scene(path.stem)))
    paths = paths[int(shard_id)::int(num_shards)]
    if limit is not None:
        paths = paths[:limit]
    records = []
    batch = []
    for index, path in enumerate(paths, start=1):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        batch.append((payload['graph'], payload.get('metadata') or {'graph_id': path.stem}))
        if len(batch) >= 4:
            records.extend(infer_items(modules, model, batch, scene_root, device))
            batch = []
            if len(records) % 32 == 0:
                print(json.dumps({'infer_written': len(records), 'assigned': len(paths), 'shard_id': shard_id}), flush=True)
    if batch:
        records.extend(infer_items(modules, model, batch, scene_root, device))
    valid_total = int(sum((row.get('valid_scene', 0) for row in records)))
    status = {'status': 'PASS' if not paths and existing or (records and valid_total > 0) or (not paths) else 'FAIL', 'n': len(records), 'skipped_existing': len(existing), 'assigned': len(paths), 'valid_scene_total': valid_total, 'checkpoint_sha256': hashes['projection_scene_checkpoint_sha256'], 'scene_root': str(scene_root)}
    atomic_json(scene_root / 'infer_status.json', status)
    atomic_json(root / '05_qc' / f"projection_scene_forward_{('engineering' if engineering else 'full')}.json", status)
    return status
