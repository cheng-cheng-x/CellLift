from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import gzip
from celllift.runtime import json
import os
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.model_inputs.common.segment import _eval, _load_models, canonical_relabel, instance_payload, stain_inputs
from ..io_utils import atomic_json, atomic_npz
from ..protocol import CELLPOSE_DIAMETER_PX, IMAGE_SIZE
from . import layout

def _write_instances(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as handle:
        json.dump(payload, handle, separators=(',', ':'), sort_keys=True)
        handle.flush()
    os.replace(temporary, path)

def load_slide_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(item) for item in data['tile_ids'].tolist()]
        masks = data['masks']
    return {tile_id: masks[index] for index, tile_id in enumerate(ids)}

def load_slide_instances(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def load_cellpose_model():
    return _load_models(nucleus_only=True, require_cuda=True, vectorized_masks=True)[0]

def segment_slide_job(job: dict[str, Any], model=None) -> dict[str, Any]:
    from PIL import Image
    base = Path(job['model_input_root'])
    slide_id = job['slide_id']
    tiles = job['tiles']
    dest = layout.mask_npz_path(base, slide_id)
    inst_path = layout.instances_path(base, slide_id)
    if dest.is_file() and inst_path.is_file():
        existing = load_slide_instances(inst_path)
        if len(existing.get('tiles', {})) == len(tiles):
            return {'slide_id': slide_id, 'stage': 'mask', 'status': 'reused', 'n_tiles': len(tiles)}
    batch_size = int(job.get('batch_size') or 8)
    try:
        import torch
        if torch.cuda.is_available() and 'A800' in torch.cuda.get_device_name(0):
            batch_size = max(batch_size, 16)
    except Exception:
        pass
    if model is None:
        model = load_cellpose_model()
    tile_ids: list[str] = []
    masks: list[np.ndarray] = []
    instances: dict[str, Any] = {}
    timings: list[dict[str, Any]] = []
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start:start + batch_size]
        images = []
        for row in chunk:
            with Image.open(row['rgb_path']) as image:
                rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
            if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise RuntimeError(f"RGB is not 1024 RGB: {row['rgb_path']} {rgb.shape}")
            nucleus_input, _ = stain_inputs(rgb)
            images.append(nucleus_input)
        t0 = time.perf_counter()
        raw_masks = _eval(model, images, diameter=CELLPOSE_DIAMETER_PX, channels=[0, 0], batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        for row, raw in zip(chunk, raw_masks):
            mask = canonical_relabel(raw)
            if mask.dtype != np.int32:
                mask = np.asarray(mask, dtype=np.int32)
            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"mask shape {mask.shape} for {row['tile_id']}")
            payload = instance_payload(mask)
            tile_ids.append(str(row['tile_id']))
            masks.append(mask)
            instances[str(row['tile_id'])] = payload
            timings.append({'tile_id': row['tile_id'], 'seconds': elapsed / max(1, len(chunk)), 'nucleus_count': len(payload['instances'])})
    stacked = np.stack(masks, axis=0).astype(np.int32, copy=False)
    atomic_npz(dest, compressed=True, tile_ids=np.asarray(tile_ids, dtype='U128'), masks=stacked)
    _write_instances(inst_path, {'slide_id': slide_id, 'tiles': instances})
    summary = {'slide_id': slide_id, 'stage': 'mask', 'status': 'done', 'n_tiles': len(tiles), 'empty_nuclei': sum((1 for item in timings if item['nucleus_count'] == 0)), 'mask_bytes': dest.stat().st_size, 'timings': timings}
    atomic_json(layout.slide_status_path(base, slide_id).with_name(layout.slide_status_path(base, slide_id).stem + '.mask.json'), summary)
    return summary
