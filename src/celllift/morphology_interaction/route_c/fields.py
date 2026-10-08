from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from celllift.runtime import ResourcePath as Path
from ..dataset import DATASET_SPEC, SPATIAL_HW
from ..foundation.spatial import mpp_of
PLANES = 5
DEFAULT_OFFSETS_UM = (-8.0, -4.0, 0.0, 4.0, 8.0)
EXTRUDE_THICKNESS_UM = 4.0
SDF_CLIP = 2.0
P_NORM = 4.0

def plane_offsets(center_z: np.ndarray | None=None) -> np.ndarray:
    if center_z is None or not len(center_z):
        return np.asarray(DEFAULT_OFFSETS_UM, np.float32)
    finite = np.asarray(center_z, np.float32)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.asarray(DEFAULT_OFFSETS_UM, np.float32)
    return np.quantile(finite, np.linspace(0.1, 0.9, PLANES)).astype(np.float32)

def xy_grid_um(dataset: str, height: int=SPATIAL_HW) -> np.ndarray:
    mpp = mpp_of(dataset)
    canvas = 1024.0
    pad = float(DATASET_SPEC[dataset]['crc_pad'])
    xs = (np.arange(height, dtype=np.float32) + 0.5) * (canvas / height) - pad
    ys = xs.copy()
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    return np.stack((grid_x, grid_y), -1) * np.float32(mpp)

def _invert_transforms(transforms: np.ndarray) -> np.ndarray:
    return np.linalg.inv(transforms.astype(np.float64) + 1e-08 * np.eye(3)[None])

def p4_sdf(points: np.ndarray, centers: np.ndarray, transforms: np.ndarray, inv: np.ndarray | None=None) -> np.ndarray:
    if not len(centers):
        return np.full((0,) + points.shape[:2], SDF_CLIP, np.float32)
    if inv is None:
        inv = _invert_transforms(transforms)
    delta = points[None] - centers[:, None, None]
    local = np.einsum('nij,nhwj->nhwi', inv, delta)
    radius = np.power(np.clip(np.abs(local), 0, 1000000.0) ** P_NORM, 1.0).sum(-1)
    radius = np.power(np.maximum(radius, 0.0), 1.0 / P_NORM)
    return np.clip(radius - 1.0, -SDF_CLIP, SDF_CLIP).astype(np.float32)

def raster_pair(centers_n, transforms_n, centers_c, transforms_c, dataset: str, offsets: np.ndarray, include: np.ndarray, valid3d: np.ndarray) -> np.ndarray:
    grid = xy_grid_um(dataset)
    keep_n = include & valid3d
    keep_c = include & valid3d
    inv_n = _invert_transforms(transforms_n[keep_n]) if keep_n.any() else None
    inv_c = _invert_transforms(transforms_c[keep_c]) if keep_c.any() else None
    channels = []
    for z in offsets:
        points = np.concatenate((grid, np.full(grid.shape[:2] + (1,), float(z), np.float32)), -1)
        if keep_n.any():
            sdf_n = p4_sdf(points, centers_n[keep_n], transforms_n[keep_n], inv=inv_n).min(0)
        else:
            sdf_n = np.full(grid.shape[:2], SDF_CLIP, np.float32)
        if keep_c.any():
            sdf_c = p4_sdf(points, centers_c[keep_c], transforms_c[keep_c], inv=inv_c).min(0)
        else:
            sdf_c = np.full(grid.shape[:2], SDF_CLIP, np.float32)
        channels.extend((sdf_n, sdf_c))
    return np.stack(channels, 0).astype(np.float32)

def _resize(array: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image
        mode = Image.NEAREST if array.dtype == np.uint8 and array.ndim == 2 else Image.BILINEAR
        image = Image.fromarray(array)
        return np.asarray(image.resize((size, size), mode))
    except Exception:
        step_y = max(1, array.shape[0] // size)
        step_x = max(1, array.shape[1] // size)
        return array[::step_y, ::step_x][:size, :size]

def raster_2d_mask(mask: np.ndarray, dataset: str) -> np.ndarray:
    del dataset
    canvas = 1024
    if mask.shape[0] != canvas or mask.shape[1] != canvas:
        mask = _resize(mask.astype(np.uint8), canvas)
    binary = (mask > 0).astype(np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(1.0 - binary).astype(np.float32)
        dist = np.clip(dist / 32.0, 0, SDF_CLIP)
    except Exception:
        dist = 1.0 - binary
    small = _resize((binary * 255).astype(np.uint8), SPATIAL_HW).astype(np.float32) / 255.0
    dist_s = _resize((dist * 255 / max(SDF_CLIP, 1e-06)).astype(np.uint8), SPATIAL_HW).astype(np.float32) / 255.0
    if small.ndim != 2:
        small = np.asarray(small, np.float32)
        dist_s = np.asarray(dist_s, np.float32)
    if small.shape != (SPATIAL_HW, SPATIAL_HW):
        padded = np.zeros((SPATIAL_HW, SPATIAL_HW), np.float32)
        padded[:small.shape[0], :small.shape[1]] = small[:SPATIAL_HW, :SPATIAL_HW]
        small = padded
        padded = np.zeros((SPATIAL_HW, SPATIAL_HW), np.float32)
        padded[:dist_s.shape[0], :dist_s.shape[1]] = dist_s[:SPATIAL_HW, :SPATIAL_HW]
        dist_s = padded
    channels = [small, dist_s]
    for scale in (0.5, 0.25, 0.125):
        size = max(8, int(SPATIAL_HW * scale))
        resized = _resize((small * 255).astype(np.uint8), size)
        resized = _resize(np.asarray(resized, np.uint8), SPATIAL_HW).astype(np.float32) / 255.0
        if resized.shape != (SPATIAL_HW, SPATIAL_HW):
            pad = np.zeros((SPATIAL_HW, SPATIAL_HW), np.float32)
            pad[:resized.shape[0], :resized.shape[1]] = resized[:SPATIAL_HW, :SPATIAL_HW]
            resized = pad
        channels.append(resized)
    while len(channels) < 10:
        channels.append(channels[-1])
    return np.stack(channels[:10], 0).astype(np.float32)

def raster_extrude(mask: np.ndarray, dataset: str, offsets: np.ndarray, thickness: float=EXTRUDE_THICKNESS_UM) -> np.ndarray:
    del dataset
    base = raster_2d_mask(mask, 'sicapv2')
    channels = []
    for z in offsets:
        weight = float(np.exp(-0.5 * (float(z) / max(thickness, 0.001)) ** 2))
        channels.append(base[0] * weight)
        channels.append(base[1] * weight)
    stacked = np.stack(channels, 0)
    if stacked.shape[0] < 10:
        pad = np.repeat(stacked[-1:], 10 - stacked.shape[0], 0)
        stacked = np.concatenate((stacked, pad), 0)
    return stacked[:10].astype(np.float32)

def field_sidecar(npz_path: str | Path, arm: str) -> Path:
    path = Path(npz_path)
    return path.with_name(f'{path.stem}.field_{arm}.npy')

def load_or_raster_field(graph, entry: dict, dataset: str, arm: str) -> np.ndarray:
    cached = getattr(graph, '_fields', None)
    if cached is None:
        cached = {}
        graph._fields = cached
    hit = cached.get(arm)
    if hit is not None:
        return hit
    sidecar = field_sidecar(entry.get('path') or '', arm) if entry.get('path') else None
    if sidecar is not None and sidecar.is_file():
        field = np.asarray(np.load(sidecar), np.float32)
        cached[arm] = field
        return field
    offsets = np.asarray(DEFAULT_OFFSETS_UM, np.float32)
    if arm == 'C3':
        field = raster_pair(graph.nucleus_center, graph.nucleus_transform, graph.cell_center, graph.cell_transform, dataset, offsets, graph.include, graph.valid3d)
    else:
        mask_path = (entry or {}).get('mask_path')
        if mask_path and Path(mask_path).is_file():
            mask = np.load(mask_path)
        else:
            mask = np.zeros((1024, 1024), np.uint8)
            for xy in graph.center_xy[graph.include]:
                y = int(np.clip(xy[1] / 0.46, 0, 1023))
                x = int(np.clip(xy[0] / 0.46, 0, 1023))
                mask[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = 1
        field = raster_extrude(mask, dataset, offsets) if arm == 'CX' else raster_2d_mask(mask, dataset)
    cached[arm] = field
    if sidecar is not None:
        from ..io_utils import atomic_npy
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        atomic_npy(sidecar, field)
    return field

def raster_arm_from_arrays(arrays: dict, entry: dict, dataset: str, arm: str) -> np.ndarray:
    offsets = np.asarray(DEFAULT_OFFSETS_UM, np.float32)
    include = np.asarray(arrays['include'], bool)
    valid3d = np.asarray(arrays['valid3d'], bool)
    if arm == 'C3':
        return raster_pair(np.asarray(arrays['nucleus_center'], np.float32), np.asarray(arrays['nucleus_transform'], np.float32), np.asarray(arrays['cell_center'], np.float32), np.asarray(arrays['cell_transform'], np.float32), dataset, offsets, include, valid3d)
    mask_path = (entry or {}).get('mask_path')
    if mask_path and Path(mask_path).is_file():
        mask = np.load(mask_path)
    else:
        mask = np.zeros((1024, 1024), np.uint8)
        for xy in np.asarray(arrays['center_xy'], np.float32)[include]:
            y = int(np.clip(xy[1] / 0.46, 0, 1023))
            x = int(np.clip(xy[0] / 0.46, 0, 1023))
            mask[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = 1
    return raster_extrude(mask, dataset, offsets) if arm == 'CX' else raster_2d_mask(mask, dataset)

def _prefetch_one(item: tuple) -> str:
    path, dataset, arm, entry = item
    sidecar = field_sidecar(path, arm)
    if sidecar.is_file():
        return 'skip'
    from ..io_utils import atomic_npy, load_npz
    arrays = load_npz(path)
    field = raster_arm_from_arrays(arrays, entry, dataset, arm)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    atomic_npy(sidecar, field)
    return 'write'

def prefetch_missing_fields(cfg, dataset: str, arm: str, workers: int=4) -> dict:
    from celllift.runtime import json
    from concurrent.futures import ProcessPoolExecutor
    from ..dataset import cache_index_path
    payload = json.loads(Path(cache_index_path(cfg, dataset)).read_text(encoding='utf-8'))
    jobs = []
    skipped = 0
    for entry in payload['graphs']:
        path = entry.get('path') or ''
        if not path:
            continue
        if field_sidecar(path, arm).is_file():
            skipped += 1
            continue
        jobs.append((path, dataset, arm, entry))
    written = 0
    if jobs:
        workers = max(1, int(workers))
        if workers == 1:
            written = sum((1 for item in jobs if _prefetch_one(item) == 'write'))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                written = sum((1 for status in pool.map(_prefetch_one, jobs, chunksize=8) if status == 'write'))
    return {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'written': written, 'skipped': skipped, 'missing': len(jobs)}
