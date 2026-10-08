from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import numpy as np
from PIL import Image
from .aggregation import assign_group, gleason_sum_to_qwk_label, gleason_summary_wsum, primary_secondary_from_counts
from .constants import ARVANITI_CORE_PX, ARVANITI_DATA, ARVANITI_NATIVE, ARVANITI_RAW
from .io_utils import atomic_json, read_parquet, write_parquet
from .splits import author_patch_coords
TISSUE_FRACTION = 0.05

def author_tissue_mask(gray: np.ndarray) -> np.ndarray:
    import cv2
    blur = cv2.GaussianBlur(np.asarray(gray, np.uint8), (25, 25), 0)
    _, workspace = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(workspace, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((20, 20), np.uint8)
    padded = cv2.dilate(padded, kernel, iterations=5)
    padded = cv2.erode(padded, kernel, iterations=10)
    padded = cv2.dilate(padded, kernel, iterations=5)
    workspace = padded[100:-100, 100:-100]
    mask = np.full(workspace.shape, 4, dtype=np.uint8)
    mask[workspace == 255] = 0
    return mask

def resample_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask)).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image)

def write_tissue_masks(data_root: Path | None=None) -> dict[str, Any]:
    data_root = Path(data_root or ARVANITI_DATA)
    cores = read_parquet(data_root / '04_labels_splits' / 'cores.parquet')
    out = data_root / '02_nucleus_masks' / 'tissue_author'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for core in cores:
        rgb = np.asarray(Image.open(core['path']).convert('L'))
        native = author_tissue_mask(rgb)
        target = resample_mask(native, ARVANITI_CORE_PX, ARVANITI_CORE_PX)
        native_path = out / f"{core['core_id']}_native.npy"
        target_path = out / f"{core['core_id']}_1550.npy"
        np.save(native_path, native)
        np.save(target_path, target)
        rows.append({'core_id': core['core_id'], 'role': core['role'], 'tissue_fraction_native': float((native == 0).mean()), 'tissue_fraction_1550': float((target == 0).mean()), 'native_path': str(native_path), 'target_path': str(target_path)})
    write_parquet(data_root / '04_labels_splits_conditional_geometry' / 'tissue_masks.parquet', rows)
    return {'cores': len(rows), 'root': str(out)}

def freeze_inference_windows(data_root: Path | None=None) -> dict[str, Any]:
    data_root = Path(data_root or ARVANITI_DATA)
    cores = read_parquet(data_root / '04_labels_splits' / 'cores.parquet')
    supervised = read_parquet(data_root / '04_labels_splits' / 'windows.parquet')
    tissue_dir = data_root / '02_nucleus_masks' / 'tissue_author'
    out = data_root / '04_labels_splits_conditional_geometry'
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(out / 'supervised_windows.parquet', supervised)
    write_parquet(out / 'cores.parquet', cores)
    rows = []
    for core in cores:
        tissue_path = tissue_dir / f"{core['core_id']}_native.npy"
        if not tissue_path.is_file():
            gray = np.asarray(Image.open(core['path']).convert('L'))
            tissue = author_tissue_mask(gray)
        else:
            tissue = np.load(tissue_path)
        height, width = tissue.shape[:2]
        rgb = np.asarray(Image.open(core['path']).convert('RGB'))
        for patch_index, row0, col0 in author_patch_coords(width, height):
            crop = rgb[row0:row0 + 750, col0:col0 + 750]
            tissue_crop = tissue[row0:row0 + 750, col0:col0 + 750]
            if crop.shape[0] != 750 or crop.shape[1] != 750:
                continue
            fraction = float((tissue_crop == 0).mean())
            if fraction < TISSUE_FRACTION:
                continue
            rows.append({'graph_id': f"{core['core_id']}_patch_{patch_index:03d}", 'core_id': core['core_id'], 'board': core['board'], 'role': core['role'], 'author_patch_index': int(patch_index), 'row0_native': int(row0), 'col0_native': int(col0), 'size_native': 750, 'row0_target': int(round(row0 * 0.5)), 'col0_target': int(round(col0 * 0.5)), 'size_target': 375, 'rgb_path': core['path'], 'tissue_fraction': fraction, 'patient_id': core['core_id'], 'official_split': {'FIT': 'train', 'VAL': 'val', 'TEST': 'test'}[core['role']]})
    write_parquet(out / 'inference_windows.parquet', rows)
    covered = {row['core_id'] for row in rows}
    missing = [row['core_id'] for row in cores if row['core_id'] not in covered]
    payload = {'inference_windows': len(rows), 'cores_with_inference': len(covered), 'cores_missing_inference': missing, 'test_inference_cores': sum((1 for row in cores if row['role'] == 'TEST' and row['core_id'] in covered))}
    atomic_json(out / 'inference_lock.json', payload)
    return payload

def core_label_from_mask(mask: np.ndarray) -> dict[str, int]:
    values = np.asarray(mask).reshape(-1)
    values = values[values < 4]
    counts = np.bincount(values.astype(np.int64), minlength=4) if values.size else np.zeros(4, np.int64)
    primary, secondary = primary_secondary_from_counts(counts)
    gleason_sum = assign_group(primary, secondary)
    return {'primary': primary, 'secondary': secondary, 'gleason_sum': gleason_sum, 'qwk_label': gleason_sum_to_qwk_label(gleason_sum)}

def _window_boxes(windows: Iterable[dict[str, Any]], canvas: int=ARVANITI_CORE_PX) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for window in windows:
        row0 = int(window['row0_target'])
        col0 = int(window['col0_target'])
        size = int(window.get('size_target', 375))
        row1 = min(canvas, row0 + size)
        col1 = min(canvas, col0 + size)
        boxes.append((row0, row1, col0, col1))
    return boxes

def rasterize_window_probs(windows: Iterable[dict[str, Any]], probs: np.ndarray, canvas: int=ARVANITI_CORE_PX) -> tuple[np.ndarray, np.ndarray]:
    field = np.zeros((canvas, canvas, 4), dtype=np.float32)
    weight = np.zeros((canvas, canvas), dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    boxes = _window_boxes(windows, canvas)
    for index, (row0, row1, col0, col1) in enumerate(boxes):
        if row1 <= row0 or col1 <= col0:
            continue
        field[row0:row1, col0:col1] += probs[index]
        weight[row0:row1, col0:col1] += 1.0
    mean = field / np.clip(weight[..., None], 1.0, None)
    mean[weight == 0] = 0
    return (mean, weight)

def pool_window_alphas(windows: Iterable[dict[str, Any]], tissue_1550: np.ndarray, canvas: int=ARVANITI_CORE_PX) -> np.ndarray:
    boxes = _window_boxes(windows, canvas)
    weight = np.zeros((canvas, canvas), dtype=np.float32)
    for row0, row1, col0, col1 in boxes:
        if row1 <= row0 or col1 <= col0:
            continue
        weight[row0:row1, col0:col1] += 1.0
    inv = np.zeros_like(weight)
    nz = weight > 0
    inv[nz] = 1.0 / weight[nz]
    inv[np.asarray(tissue_1550) == 4] = 0.0
    alphas = np.zeros(len(boxes), dtype=np.float64)
    for index, (row0, row1, col0, col1) in enumerate(boxes):
        if row1 <= row0 or col1 <= col0:
            continue
        alphas[index] = float(inv[row0:row1, col0:col1].sum())
    return alphas

def pool_official_from_alphas(probs: np.ndarray, alphas: np.ndarray, *, thres: float=0.25) -> dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    w_sum = alphas @ probs[:, :4]
    primary, secondary, gleason_sum = gleason_summary_wsum(w_sum, thres=thres)
    return {'w_sum': [float(x) for x in w_sum], 'primary': primary, 'secondary': secondary, 'gleason_sum': gleason_sum, 'qwk_label': gleason_sum_to_qwk_label(gleason_sum)}

def pool_official(prob_map: np.ndarray, tissue_1550: np.ndarray, *, thres: float=0.25) -> dict[str, Any]:
    masked = np.asarray(prob_map, dtype=np.float64).copy()
    masked[np.asarray(tissue_1550) == 4] = 0
    w_sum = masked.reshape(-1, 4).sum(axis=0)
    primary, secondary, gleason_sum = gleason_summary_wsum(w_sum, thres=thres)
    return {'w_sum': [float(x) for x in w_sum], 'primary': primary, 'secondary': secondary, 'gleason_sum': gleason_sum, 'qwk_label': gleason_sum_to_qwk_label(gleason_sum)}
