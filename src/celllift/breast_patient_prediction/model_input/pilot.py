from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from ..io_utils import atomic_json
from . import jobs, layout
from .jobs import select_pilot_tiles

def _job_for_slide(slide_id: str, tiles: list[dict[str, Any]], base: Path, inventory: dict[str, Any], batch_size: int) -> dict[str, Any]:
    enriched = [jobs.enrich_tile(row, inventory, base) for row in tiles]
    return {'slide_id': slide_id, 'model_input_root': str(base), 'tiles': enriched, 'batch_size': batch_size, 'cpu_threads': 8}

def run_pilot(data_root: Path | None=None, *, device: str='cuda', batch_size: int=8) -> dict[str, Any]:
    from celllift.matched_geometry_controls.inference import load_model
    from ..protocol import ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST
    from .config import write_config
    from .encode import infer_slide, load_encoder
    from .graphs import build_slide_graphs, load_slide_graphs
    from .invariants import check_batch_vs_single, check_empty, check_mask_graph, check_rgb, check_scene_alignment
    from .overlays import write_overlay
    from .rgb import export_rgb
    from .segment import load_slide_instances, load_slide_masks
    from .. import paths
    import os
    import subprocess
    write_config(data_root, verify_checkpoint=True)
    write_config(data_root, verify_checkpoint=True)
    base = layout.ensure(data_root)
    manifest = jobs.load_tile_manifest(data_root)
    labels = jobs.load_patient_labels(data_root)
    inventory = jobs.load_slide_inventory(data_root)
    selected = select_pilot_tiles(manifest, labels)
    tile_ids = {str(row['tile_id']) for row in selected}
    atomic_json(base / 'qc' / 'pilot_tiles.json', {'tiles': selected, 'n': len(selected)})
    timings: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    rgb_summary = export_rgb(data_root, tile_ids=tile_ids, workers=min(8, max(1, len(tile_ids))), write_cfg=False)
    rgb_seconds = time.perf_counter() - t0
    grouped = jobs.slide_groups(selected)
    mask_seconds = 0.0
    graph_seconds = 0.0
    infer_seconds = 0.0
    modules = encoder = model = None
    nonempty_items = []
    for slide_id, tiles in grouped.items():
        job = _job_for_slide(slide_id, tiles, base, inventory, batch_size)
        t1 = time.perf_counter()
        job_path = base / 'logs' / 'pilot_jobs' / f'{layout._slide_stem(slide_id)}.json'
        job_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(job_path, job)
        code_root = Path(__file__).resolve().parents[1].parent
        subprocess.check_call([str(paths.CELLPOSE_PYTHON), '-m', 'breast_patient_prediction.model_input.run', 'segment-slide', '--job-json', str(job_path)], cwd=str(code_root), env={**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0')})
        mask_seconds += time.perf_counter() - t1
        t1 = time.perf_counter()
        build_slide_graphs(job)
        graph_seconds += time.perf_counter() - t1
        if modules is None:
            modules, encoder = load_encoder(device)
            model, _ = load_model(modules, ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST, device)
        t1 = time.perf_counter()
        inferred = infer_slide(job, device=device, modules=modules, encoder=encoder, model=model)
        infer_seconds += time.perf_counter() - t1
        masks = load_slide_masks(layout.mask_npz_path(base, slide_id))
        instances = load_slide_instances(layout.instances_path(base, slide_id))['tiles']
        records = load_slide_graphs(layout.graph_path(base, slide_id))
        dino_by_tile = {row['tile_id']: row for row in inferred['tile_rows']}
        for row in job['tiles']:
            tile_id = str(row['tile_id'])
            rgb = layout.rgb_path(base, slide_id, tile_id)
            check_rgb(rgb)
            record = records[tile_id]
            check_mask_graph(masks[tile_id], instances[tile_id], record)
            overlay = write_overlay(rgb, masks[tile_id], base / 'qc' / 'pilot_overlays' / f'{tile_id}.png')
            info = dino_by_tile[tile_id]
            scene = inferred['scenes'].get(str(record.graph_id))
            if len(record.nucleus_id) == 0:
                check_empty(rgb, record, scene, 0, np.asarray(info['dino'], np.float32))
            else:
                check_scene_alignment(record, scene)
            timings.append({'tile_id': tile_id, 'slide_id': slide_id, 'patient_id': row['patient_id'], 'split': row.get('split'), 'n_nuclei': int(len(record.nucleus_id)), 'png_bytes': rgb.stat().st_size, 'overlay': str(overlay), 'empty': len(record.nucleus_id) == 0})
        nonempty_items.extend(inferred['items'])
    if len(nonempty_items) >= 2:
        check_batch_vs_single(modules, model, [(g, m) for g, m, *_ in nonempty_items], device, base / 'qc' / 'pilot_regression')
    payload = {'status': 'PASS', 'n_tiles': len(selected), 'n_slides': len(grouped), 'rgb': rgb_summary, 'seconds': {'rgb': rgb_seconds, 'mask': mask_seconds, 'graph': graph_seconds, 'infer': infer_seconds, 'per_tile_rgb': rgb_seconds / max(1, len(selected)), 'per_tile_mask': mask_seconds / max(1, len(selected)), 'per_tile_infer': infer_seconds / max(1, len(selected))}, 'extrapolated_hours_123406': {'rgb_16_workers': rgb_seconds / max(1, len(selected)) * 123406 / 16 / 3600, 'mask_one_gpu': mask_seconds / max(1, len(selected)) * 123406 / 3600, 'infer_one_gpu': infer_seconds / max(1, len(selected)) * 123406 / 3600}, 'tiles': timings}
    atomic_json(base / 'qc' / 'pilot_summary.json', payload)
    return payload
