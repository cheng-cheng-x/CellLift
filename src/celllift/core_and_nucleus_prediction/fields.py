from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.runtime import torch
from .constants import ARVANITI_DATA
SPATIAL_HW = 74

def _resize(array: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    image = Image.fromarray(array)
    return np.asarray(image.resize((size, size), Image.BILINEAR if array.ndim == 2 else Image.NEAREST))

def field_c2(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((1024, 1024), np.uint8)
    mask = np.asarray(mask)
    canvas[:min(1024, mask.shape[0]), :min(1024, mask.shape[1])] = mask[:1024, :1024] > 0
    binary = canvas.astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        dist = np.clip(distance_transform_edt(1.0 - binary) / 32.0, 0, 2.0)
    except Exception:
        dist = 1.0 - binary
    small = _resize((binary * 255).astype(np.uint8), SPATIAL_HW).astype(np.float32) / 255.0
    dist_s = _resize((dist * 127).astype(np.uint8), SPATIAL_HW).astype(np.float32) / 255.0
    channels = [small, dist_s]
    for scale in (0.5, 0.25, 0.125):
        size = max(8, int(SPATIAL_HW * scale))
        resized = _resize(_resize((small * 255).astype(np.uint8), size), SPATIAL_HW).astype(np.float32) / 255.0
        channels.append(resized)
    while len(channels) < 10:
        channels.append(channels[-1])
    return np.stack(channels[:10], 0).astype(np.float32)

def field_c3(scene: dict[str, np.ndarray] | None) -> np.ndarray:
    if scene is None or 'nucleus_transform' not in scene:
        return np.zeros((10, SPATIAL_HW, SPATIAL_HW), np.float32)
    try:
        from celllift.morphology_interaction.route_c.fields import raster_pair
        include = np.ones(len(scene['nucleus_id']), bool)
        valid = np.asarray(scene.get('valid', include), bool).reshape(-1)
        if len(valid) != len(include):
            valid = include
        return raster_pair(np.asarray(scene['nucleus_center'], np.float32), np.asarray(scene['nucleus_transform'], np.float32), np.asarray(scene['cell_center'], np.float32), np.asarray(scene['cell_transform'], np.float32), 'sicapv2', np.asarray((-8.0, -4.0, 0.0, 4.0, 8.0), np.float32), include, valid)
    except Exception:
        return np.zeros((10, SPATIAL_HW, SPATIAL_HW), np.float32)

def dino_spatial(graph_id: str) -> np.ndarray:
    path = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'dino_spatial' / f'{graph_id}.npy'
    if path.is_file():
        return np.load(path).astype(np.float32)
    return np.zeros((384, SPATIAL_HW, SPATIAL_HW), np.float32)

def cache_dino_spatial(device: str='cuda', limit: int | None=None) -> dict[str, Any]:
    from PIL import Image
    from celllift.model_inputs.arvaniti_lizard.constants import DINO_SOURCE, DINO_WEIGHTS, ProjectionScene_SOURCE
    from celllift.model_inputs.arvaniti_lizard.io_utils import read_parquet
    from celllift.model_inputs.arvaniti_lizard.rgb import pad_white
    from celllift.matched_geometry_controls.upstream import load_projection_scene
    from celllift.morphology_interaction.foundation.spatial import encode_spatial_batch
    dest = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'dino_spatial'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ('window_manifest.parquet', 'inference_window_manifest_conditional_geometry.parquet'):
        path = ARVANITI_DATA / '00_manifest' / name
        if path.is_file():
            rows.extend(read_parquet(path))
    seen = set()
    unique = []
    for row in rows:
        if row['graph_id'] in seen:
            continue
        seen.add(row['graph_id'])
        unique.append(row)
    if limit is not None:
        unique = unique[:limit]
    modules = load_projection_scene(ProjectionScene_SOURCE)
    encoder = modules.he_features.DINOv2S14Encoder(Path(DINO_SOURCE), Path(DINO_WEIGHTS)).to(device).eval()
    written = 0
    skipped = 0
    batch_rows, batch_images = ([], [])
    for row in unique:
        target = dest / f"{row['graph_id']}.npy"
        if target.is_file() and target.stat().st_size > 0:
            skipped += 1
            continue
        rgb_path = row.get('dino_rgb_path') or row.get('rgb_path')
        if not rgb_path or not Path(str(rgb_path)).is_file():
            continue
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB'))
        if rgb.shape[0] != 1024 or rgb.shape[1] != 1024:
            rgb = pad_white(rgb)
        batch_rows.append(row['graph_id'])
        batch_images.append(rgb)
        if len(batch_images) >= 4:
            maps = encode_spatial_batch(encoder, batch_images, device)
            for graph_id, field in zip(batch_rows, maps):
                np.save(dest / f'{graph_id}.npy', field)
                written += 1
            batch_rows, batch_images = ([], [])
    if batch_images:
        maps = encode_spatial_batch(encoder, batch_images, device)
        for graph_id, field in zip(batch_rows, maps):
            np.save(dest / f'{graph_id}.npy', field)
            written += 1
    return {'written': written, 'skipped': skipped, 'root': str(dest)}

def cache_c3_fields(limit: int | None=None, shard: int=0, n_shards: int=1) -> dict[str, Any]:
    import os
    import zlib
    from celllift.model_inputs.arvaniti_lizard.io_utils import read_parquet
    from .data import _load_scene, scene_path
    dest = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'field_c3'
    dest.mkdir(parents=True, exist_ok=True)
    scene_root = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
    rows = []
    for name in ('window_manifest.parquet', 'inference_window_manifest_conditional_geometry.parquet'):
        path = ARVANITI_DATA / '00_manifest' / name
        if path.is_file():
            rows.extend(read_parquet(path))
    seen = set()
    written = skipped = 0
    for row in rows:
        graph_id = row['graph_id']
        if graph_id in seen:
            continue
        seen.add(graph_id)
        if n_shards > 1 and zlib.crc32(str(graph_id).encode('utf-8')) % n_shards != shard:
            continue
        target = dest / f'{graph_id}.npy'
        if target.is_file() and target.stat().st_size > 0:
            skipped += 1
            continue
        tmp = dest / f'.{graph_id}.{os.getpid()}.tmp.npy'
        np.save(tmp, field_c3(_load_scene(scene_path(scene_root, graph_id))))
        os.replace(tmp, target)
        written += 1
        if limit is not None and written >= limit:
            break
        if written % 200 == 0:
            print({'c3_fields_written': written, 'skipped': skipped, 'shard': shard, 'n_shards': n_shards}, flush=True)
    return {'written': written, 'skipped': skipped, 'root': str(dest), 'shard': shard, 'n_shards': n_shards}
_FIELD_MEMO: dict[str, np.ndarray] = {}
_DINO_SPATIAL_MEMO: dict[str, np.ndarray] = {}

def _memo_npy(memo: dict[str, np.ndarray], path: Path) -> np.ndarray | None:
    key = str(path)
    hit = memo.get(key)
    if hit is not None:
        return hit
    if not path.is_file():
        return None
    arr = np.load(path).astype(np.float32)
    memo[key] = arr
    return arr

def arvaniti_field_batch(batch: dict[str, Any], arm: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from .data import _load_scene, scene_path
    fields, dinos = ([], [])
    scene_root = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
    if not scene_root.is_dir():
        scene_root = ARVANITI_DATA / '04_projection_scene_inputs' / 'selected_scene'
    for i, graph_id in enumerate(batch['graph_id']):
        mask_path = None
        if 'mask_path' in batch and isinstance(batch['mask_path'], list):
            mask_path = batch['mask_path'][i]
        field_dir = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / ('field_c3' if arm == 'C3' else 'field_c2')
        field_dir.mkdir(parents=True, exist_ok=True)
        cached = field_dir / f'{graph_id}.npy'
        loaded = _memo_npy(_FIELD_MEMO, cached)
        if loaded is not None:
            fields.append(loaded)
        else:
            scene = _load_scene(scene_path(scene_root, graph_id))
            if mask_path and Path(str(mask_path)).is_file():
                mask = np.load(mask_path)
            else:
                mask = np.zeros((1024, 1024), np.uint8)
            field = field_c3(scene) if arm == 'C3' else field_c2(mask)
            np.save(cached, field)
            _FIELD_MEMO[str(cached)] = field.astype(np.float32)
            fields.append(_FIELD_MEMO[str(cached)])
        dino_path = ARVANITI_DATA / '04_projection_scene_inputs_conditional_geometry' / 'dino_spatial' / f'{graph_id}.npy'
        dino = _memo_npy(_DINO_SPATIAL_MEMO, dino_path)
        dinos.append(dino if dino is not None else dino_spatial(graph_id))
    return (torch.from_numpy(np.stack(fields)).to(device), torch.from_numpy(np.stack(dinos)).to(device))
