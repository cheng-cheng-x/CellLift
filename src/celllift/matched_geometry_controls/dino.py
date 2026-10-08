from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Sequence
import numpy as np
CRC_SOURCE_SIZE = 557
CANVAS_SIZE = 1024
CRC_PAD_LEFT_TOP = 233
CRC_PAD_RIGHT_BOTTOM = 234

def crc_white_canvas(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (CRC_SOURCE_SIZE, CRC_SOURCE_SIZE, 3):
        raise ValueError(f'CRC image must be {CRC_SOURCE_SIZE}x{CRC_SOURCE_SIZE} RGB')
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    canvas[CRC_PAD_LEFT_TOP:CRC_PAD_LEFT_TOP + CRC_SOURCE_SIZE, CRC_PAD_LEFT_TOP:CRC_PAD_LEFT_TOP + CRC_SOURCE_SIZE] = value.astype(np.uint8, copy=False)
    return canvas

def dino_xy_px(dataset: str, xy_px: np.ndarray) -> np.ndarray:
    value = np.asarray(xy_px, np.float32)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError('xy_px must have shape [N,2]')
    return value + np.float32(CRC_PAD_LEFT_TOP) if dataset == 'tcga_crc_msi' else value.copy()

def load_rgb(path: str | Path, dataset: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'), np.uint8)
    if dataset == 'tcga_crc_msi':
        return crc_white_canvas(rgb)
    if rgb.shape != (CANVAS_SIZE, CANVAS_SIZE, 3):
        raise ValueError(f'{dataset} RGB must be 1024x1024')
    return rgb

def encode_graphs(encoder, images_hwc: Sequence[np.ndarray], xy_px: Sequence[np.ndarray], device: str='cuda'):
    import torch
    if len(images_hwc) != len(xy_px) or not images_hwc:
        raise ValueError('images/coordinate batches must be nonempty and aligned')
    images = torch.from_numpy(np.stack([np.ascontiguousarray(image.transpose(2, 0, 1)) for image in images_hwc])).to(device)
    counts = [len(value) for value in xy_px]
    maximum = max(counts)
    grid = torch.full((len(counts), maximum, 1, 2), -1.0, dtype=torch.float32, device=device)
    for index, value in enumerate(xy_px):
        xy = torch.as_tensor(value, dtype=torch.float32, device=device)
        grid[index, :len(xy), 0] = 2.0 * (xy + float(encoder.source_padding)) / float(encoder.encoded_size) - 1.0
    with torch.inference_mode():
        field = encoder(images)
        sampled = torch.nn.functional.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return [sampled[index, :, :count, 0].T for index, count in enumerate(counts)]

def encode_nuclei(encoder, image_hwc: np.ndarray, xy_px: np.ndarray, device: str='cuda'):
    return encode_graphs(encoder, [image_hwc], [xy_px], device)[0]
