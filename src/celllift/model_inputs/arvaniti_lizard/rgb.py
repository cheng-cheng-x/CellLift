from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
import numpy as np
from PIL import Image
from .constants import DINO_CANVAS, TARGET_MPP
from .io_utils import sha256_file

def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)

def save_png(path: Path, rgb: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    image.save(temporary, format='PNG', compress_level=1)
    os.replace(temporary, path)
    return sha256_file(path)

def resample_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return np.asarray(rgb, dtype=np.uint8)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB')
    return np.asarray(image.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)

def resample_ids(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[0] == height and mask.shape[1] == width:
        return np.asarray(mask)
    image = Image.fromarray(np.asarray(mask).astype(np.int32), mode='I')
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.int32)

def target_hw(source_h: int, source_w: int, source_mpp: float, target_mpp: float=TARGET_MPP) -> tuple[int, int, float, float]:
    scale = float(source_mpp) / float(target_mpp)
    height = int(round(source_h * scale))
    width = int(round(source_w * scale))
    return (height, width, scale, source_w / width if width else 0.0)

def pad_white(rgb: np.ndarray, size: int=DINO_CANVAS) -> np.ndarray:
    height, width = rgb.shape[:2]
    if height > size or width > size:
        raise RuntimeError(f'image {height}x{width} exceeds DINO canvas {size}; refuse stretch')
    if height == size and width == size:
        return np.asarray(rgb, dtype=np.uint8)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas[:height, :width] = rgb
    return canvas

def pad_mask(mask: np.ndarray, size: int=DINO_CANVAS) -> np.ndarray:
    height, width = mask.shape[:2]
    if height > size or width > size:
        raise RuntimeError(f'mask {height}x{width} exceeds DINO canvas {size}')
    if height == size and width == size:
        return np.asarray(mask)
    canvas = np.zeros((size, size), dtype=np.int32)
    canvas[:height, :width] = mask
    return canvas
