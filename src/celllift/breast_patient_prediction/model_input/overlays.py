from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from PIL import Image

def nucleus_boundary(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask)
    padded = np.pad(labels, 1)
    differ = (padded[1:-1, 1:-1] != padded[:-2, 1:-1]) | (padded[1:-1, 1:-1] != padded[2:, 1:-1]) | (padded[1:-1, 1:-1] != padded[1:-1, :-2]) | (padded[1:-1, 1:-1] != padded[1:-1, 2:])
    return (labels > 0) & differ

def write_overlay(rgb_path: Path, mask: np.ndarray, dest: Path) -> Path:
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    overlay = rgb.copy()
    if mask.max(initial=0) > 0:
        overlay[nucleus_boundary(mask)] = [255, 0, 0]
    canvas = Image.new('RGB', (rgb.shape[1] * 2, rgb.shape[0]))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    canvas.paste(Image.fromarray(overlay), (rgb.shape[1], 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, format='PNG', compress_level=1)
    return dest
