from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from celllift.runtime import ResourcePath as Path
from typing import Any
from PIL import Image
from ..io_utils import atomic_json, atomic_parquet
from ..protocol import IMAGE_SIZE
from . import jobs, layout
from .config import write_config

def rgba_to_rgb_white(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    rgba = image.convert('RGBA')
    background = Image.new('RGB', rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[-1])
    return background

def resize_lanczos(image: Image.Image, size: int=IMAGE_SIZE) -> Image.Image:
    rgb = rgba_to_rgb_white(image)
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
    return rgb.convert('RGB')

def png_is_valid(path: Path, size: int=IMAGE_SIZE) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == 'RGB' and image.size == (size, size)
    except Exception:
        return False

def _atomic_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    image.save(temporary, format='PNG', compress_level=1)
    with Image.open(temporary) as check:
        check.load()
        if check.mode != 'RGB' or check.size != image.size:
            raise RuntimeError(f'PNG validation failed: {temporary}')
    os.replace(temporary, destination)

def export_tile(slide, row: dict[str, Any], destination: Path, size: int=IMAGE_SIZE) -> dict[str, Any]:
    if png_is_valid(destination, size):
        return {'tile_id': row['tile_id'], 'status': 'reused', 'rgb_path': str(destination), 'bytes': destination.stat().st_size}
    region = slide.read_region((int(row['level0_x']), int(row['level0_y'])), 0, (int(row['level0_size']), int(row['level0_size'])))
    rgb = resize_lanczos(region, size)
    _atomic_png(rgb, destination)
    return {'tile_id': row['tile_id'], 'status': 'wrote', 'rgb_path': str(destination), 'bytes': destination.stat().st_size}

def export_slide(payload: dict[str, Any]) -> dict[str, Any]:
    import openslide
    rows = payload['tiles']
    source = payload['source_path']
    base = Path(payload['model_input_root'])
    slide_id = payload['slide_id']
    started = payload.get('started')
    records = []
    error = ''
    status = 'done'
    try:
        handle = openslide.OpenSlide(source)
        try:
            for row in rows:
                dest = layout.rgb_path(base, slide_id, str(row['tile_id']))
                records.append(export_tile(handle, row, dest))
        finally:
            handle.close()
    except Exception as exc:
        status = 'fail'
        error = f'{type(exc).__name__}: {exc}'
        for row in rows:
            dest = layout.rgb_path(base, slide_id, str(row['tile_id']))
            if not any((item['tile_id'] == row['tile_id'] for item in records)):
                records.append({'tile_id': row['tile_id'], 'status': 'fail', 'rgb_path': str(dest), 'error': error})
    summary = {'slide_id': slide_id, 'stage': 'rgb', 'status': status, 'error': error, 'n_tiles': len(rows), 'n_ok': sum((1 for item in records if item['status'] in {'wrote', 'reused'})), 'source_path': source, 'mpp_x': payload.get('mpp_x'), 'mpp_y': payload.get('mpp_y'), 'mpp_xy_mismatch': bool(payload.get('mpp_xy_mismatch')), 'started': started}
    atomic_json(layout.slide_status_path(base, slide_id), summary)
    return {**summary, 'tiles': records}

def _slide_payloads(tile_ids: set[str] | None, data_root: Path | None) -> list[dict[str, Any]]:
    base = layout.ensure(data_root)
    manifest = jobs.load_tile_manifest(data_root)
    inventory = jobs.load_slide_inventory(data_root)
    rows = [jobs.enrich_tile(row, inventory, base) for row in manifest.to_dict('records')]
    if tile_ids is not None:
        rows = [row for row in rows if str(row['tile_id']) in tile_ids]
    grouped = jobs.slide_groups(rows)
    payloads = []
    for slide_id, tiles in grouped.items():
        mpp_x = float(tiles[0]['mpp_x'])
        mpp_y = float(tiles[0]['mpp_y'])
        payloads.append({'slide_id': slide_id, 'source_path': tiles[0]['wsi_source_path'], 'model_input_root': str(base), 'tiles': tiles, 'mpp_x': mpp_x, 'mpp_y': mpp_y, 'mpp_xy_mismatch': abs(mpp_x - mpp_y) > 1e-06})
    return payloads

def export_rgb(data_root: Path | None=None, *, tile_ids: set[str] | None=None, workers: int=16, write_cfg: bool=True) -> dict[str, Any]:
    if write_cfg:
        write_config(data_root, verify_checkpoint=Path(_resource_path('artifact_0019')).exists())
    payloads = _slide_payloads(tile_ids, data_root)
    results = []
    if workers <= 1 or len(payloads) <= 1:
        results = [export_slide(item) for item in payloads]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(export_slide, item) for item in payloads]
            for future in as_completed(futures):
                results.append(future.result())
    failed = [row for row in results if row['status'] != 'done']
    mismatches = [row for row in results if row.get('mpp_xy_mismatch')]
    base = layout.root(data_root)
    index_rows = []
    for result in results:
        for tile in result.get('tiles', []):
            index_rows.append({'slide_id': result['slide_id'], 'tile_id': tile['tile_id'], 'rgb_path': tile.get('rgb_path'), 'status': tile.get('status'), 'wsi_source_path': result['source_path']})
    if index_rows:
        tag = 'pilot' if tile_ids is not None else 'full'
        atomic_parquet(base / 'logs' / f'rgb_export_{tag}.parquet', index_rows)
    summary = {'status': 'PASS' if not failed else 'PARTIAL', 'slides': len(results), 'failed_slides': len(failed), 'tiles_ok': sum((int(row.get('n_ok') or 0) for row in results)), 'mpp_xy_mismatch_slides': len(mismatches)}
    atomic_json(base / 'logs' / 'rgb_export_summary.json', summary)
    return summary
