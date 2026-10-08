from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from ..dataset import DATASET_SPEC, DINO_DIM, DINO_ENCODED, DINO_PAD, SPATIAL_HW

def mpp_of(dataset: str) -> float:
    return float(DATASET_SPEC[dataset]['mpp'])

def xy_um_to_pixel(xy_um: np.ndarray, dataset: str) -> np.ndarray:
    xy = np.asarray(xy_um, np.float32)
    pixel = xy / np.float32(mpp_of(dataset))
    pad = float(DATASET_SPEC[dataset]['crc_pad'])
    if pad:
        pixel = pixel + np.float32(pad)
    return pixel

def xy_um_to_grid(xy_um: np.ndarray, dataset: str) -> np.ndarray:
    pixel = xy_um_to_pixel(xy_um, dataset)
    return (2.0 * (pixel + np.float32(DINO_PAD)) / np.float32(DINO_ENCODED) - 1.0).astype(np.float32)

def xy_um_to_grid_torch(xy_um, dataset: str):
    import torch
    pixel = xy_um.to(dtype=torch.float32) / float(mpp_of(dataset))
    pad = float(DATASET_SPEC[dataset]['crc_pad'])
    if pad:
        pixel = pixel + pad
    return 2.0 * (pixel + float(DINO_PAD)) / float(DINO_ENCODED) - 1.0

def sample_maps_at(maps, xy_um, tile_index, dataset: str):
    import torch
    if maps.shape[0] == 0 or xy_um.shape[0] == 0:
        return maps.new_zeros((xy_um.shape[0], maps.shape[1]))
    grid = xy_um_to_grid_torch(xy_um, dataset)
    height, width = (int(maps.shape[-2]), int(maps.shape[-1]))
    gx = ((grid[:, 0] + 1.0) * width / 2.0 - 0.5).clamp(0, width - 1)
    gy = ((grid[:, 1] + 1.0) * height / 2.0 - 0.5).clamp(0, height - 1)
    x0 = gx.floor().long()
    y0 = gy.floor().long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = (gx - x0.to(gx.dtype)).unsqueeze(1)
    wy = (gy - y0.to(gy.dtype)).unsqueeze(1)
    tile = tile_index.long()
    ia = maps[tile, :, y0, x0]
    ib = maps[tile, :, y0, x1]
    ic = maps[tile, :, y1, x0]
    id_ = maps[tile, :, y1, x1]
    return ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy

def spatial_path(npz_path: str | Path) -> Path:
    path = Path(npz_path)
    return path.with_name(path.stem + '.spatial.npy')

def resolve_spatial(path: str | Path) -> Path | None:
    primary = Path(path)
    candidates = [primary, Path(str(primary) + '.npy'), primary.with_name(primary.stem + '.spatial.f16.npy'), primary.with_name(primary.stem + '.spatial.f16')]
    if primary.suffix != '.npy':
        candidates.insert(0, spatial_path(primary if primary.suffix == '.npz' else primary))
    seen = []
    for item in candidates:
        if item in seen:
            continue
        seen.append(item)
        if item.is_file():
            return item
    return None

def save_spatial(path: str | Path, field: np.ndarray) -> None:
    destination = spatial_path(path) if Path(path).suffix == '.npz' else Path(path)
    if destination.suffix != '.npy':
        destination = destination.with_name(destination.name + '.npy')
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(field, np.float16)
    if array.shape != (DINO_DIM, SPATIAL_HW, SPATIAL_HW):
        raise RuntimeError(f'spatial field must be [{DINO_DIM},{SPATIAL_HW},{SPATIAL_HW}], got {array.shape}')
    np.save(destination, array)

def load_spatial(path: str | Path) -> np.ndarray:
    resolved = resolve_spatial(path)
    if resolved is None:
        raise FileNotFoundError(path)
    array = np.load(resolved)
    value = np.asarray(array, np.float32)
    if value.shape != (DINO_DIM, SPATIAL_HW, SPATIAL_HW):
        raise RuntimeError(f'spatial field must be [{DINO_DIM},{SPATIAL_HW},{SPATIAL_HW}], got {value.shape}')
    return value

def encode_spatial_batch(encoder, images_hwc, device: str='cuda') -> list[np.ndarray]:
    import torch
    images = torch.from_numpy(np.stack([np.ascontiguousarray(image.transpose(2, 0, 1)) for image in images_hwc])).to(device)
    with torch.inference_mode():
        field = encoder(images)
    if tuple(field.shape[1:]) != (DINO_DIM, SPATIAL_HW, SPATIAL_HW):
        raise RuntimeError(f'unexpected spatial DINO shape {tuple(field.shape)}')
    host = field.detach().to(torch.float16).cpu().numpy()
    return [np.ascontiguousarray(host[index]) for index in range(host.shape[0])]

def load_encoder(cfg: Mapping[str, Any], device: str='cuda'):
    from celllift.matched_geometry_controls.upstream import load_projection_scene
    modules = load_projection_scene(cfg['paths']['projection_scene_source_root'])
    source = Path(cfg['paths']['dino_source_root'])
    weights = Path(cfg['paths']['dino_weight_path'])
    encoder = modules.he_features.DINOv2S14Encoder(source, weights).to(device).eval()
    return encoder
