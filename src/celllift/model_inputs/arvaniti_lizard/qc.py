from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
from .constants import ARVANITI_DATA, ARVANITI_RESULT, LIZARD_DATA, LIZARD_RESULT
from .io_utils import atomic_json, read_parquet, write_parquet
from .rgb import load_rgb

def _contours(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    contours = []
    for instance_id in np.unique(mask):
        if int(instance_id) == 0:
            continue
        ys, xs = np.nonzero(mask == instance_id)
        if len(xs) < 8:
            continue
        x0, x1 = (int(xs.min()), int(xs.max()))
        y0, y1 = (int(ys.min()), int(ys.max()))
        contours.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return contours

def overlay(rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(rgb).convert('RGBA')
    draw = ImageDraw.Draw(image)
    for poly in _contours(mask):
        draw.line(poly + [poly[0]], fill=(0, 255, 80, 255), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert('RGB').save(path)

def _count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0

def write_full_qc() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    ar_root = Path(ARVANITI_DATA)
    lz_root = Path(LIZARD_DATA)
    windows = read_parquet(ar_root / '00_manifest' / 'window_manifest.parquet')
    tiles = read_parquet(lz_root / '00_manifest' / 'tile_manifest.parquet')
    nuclei = read_parquet(lz_root / '04_labels_splits' / 'nucleus_labels.parquet')
    ar_qc = ar_root / '05_qc' / 'full'
    lz_qc = lz_root / '05_qc' / 'full'
    sampled = []
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in windows:
        by_role.setdefault(str(row.get('role', '')), []).append(row)
    for role in ('FIT', 'VAL', 'TEST'):
        sampled.extend(by_role.get(role, [])[:8])
    for row in sampled:
        window_rgb = load_rgb(row['rgb_path'])
        mask = np.load(row['nucleus_mask_path'])
        overlay(window_rgb, mask, ar_qc / f"{row['graph_id']}.png")
        if window_rgb.shape[0] != 375 or window_rgb.shape[1] != 375:
            raise RuntimeError(f"{row['graph_id']} window is {window_rgb.shape}, expected 375")
    owned_keys = {}
    dup = 0
    for row in nuclei:
        if not row['owned']:
            continue
        key = (row['roi_id'], int(row['nucleus_id']))
        if key in owned_keys:
            dup += 1
        owned_keys[key] = row['graph_id']
    lz_sampled = []
    lz_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in tiles:
        lz_by_source.setdefault(str(row.get('source', '')), []).append(row)
    for _source, rows in sorted(lz_by_source.items()):
        lz_sampled.extend(rows[:2])
    for row in lz_sampled[:24]:
        rgb = load_rgb(row.get('dino_rgb_path') or row['rgb_path'])
        mask = np.load(row['nucleus_mask_path'])
        overlay(rgb, mask, lz_qc / f"{row['graph_id']}.png")
    window_ids = {row['graph_id'] for row in windows}
    tile_ids = {row['graph_id'] for row in tiles}
    orphan_windows = sorted((path.stem for path in (ar_root / '01_rgb' / 'windows').glob('*.png') if path.stem not in window_ids))
    orphan_tiles = sorted((path.stem for path in (lz_root / '01_rgb' / 'tiles').glob('*.png') if path.stem not in tile_ids))
    reports['arvaniti'] = {'windows': len(windows), 'window_png': _count(ar_root / '01_rgb' / 'windows', '*.png'), 'core_masks': _count(ar_root / '02_nucleus_masks' / 'cores', '*.npy'), 'graphs': len(read_parquet(ar_root / '03_graph_cache' / 'graph_index.parquet')), 'projection_scene_pt': _count(ar_root / '04_projection_scene_inputs', '*.pt'), 'orphan_window_png': orphan_windows, 'excluded_windows': len(read_parquet(ar_root / '04_labels_splits' / 'excluded_windows.parquet')) if (ar_root / '04_labels_splits' / 'excluded_windows.parquet').is_file() else 0, 'overlays': len(sampled), 'checkpoint_sha256': '718d6445489266ec0708443262cae8b25aa2e458c963d11eaa9bb922aca06ee3', 'training_started': False}
    write_parquet(lz_root / '04_labels_splits' / 'excluded_tiles.parquet', [{'graph_id': graph_id, 'exclusion_reason': 'zero_nuclei'} for graph_id in orphan_tiles])
    reports['lizard'] = {'tiles': len(tiles), 'tile_png': _count(lz_root / '01_rgb' / 'tiles', '*.png'), 'graphs': len(read_parquet(lz_root / '03_graph_cache' / 'graph_index.parquet')), 'projection_scene_pt': _count(lz_root / '04_projection_scene_inputs', '*.pt'), 'owned': len(owned_keys), 'duplicate_owned': dup, 'orphan_tile_png': orphan_tiles, 'overlays': min(24, len(lz_sampled)), 'checkpoint_sha256': '718d6445489266ec0708443262cae8b25aa2e458c963d11eaa9bb922aca06ee3', 'training_started': False}
    if dup:
        raise RuntimeError(f'Lizard ownership leaked: {dup} duplicate owned nuclei')
    if reports['arvaniti']['projection_scene_pt'] != reports['arvaniti']['graphs']:
        raise RuntimeError(f"Arvaniti projection_scene {reports['arvaniti']['projection_scene_pt']} != graphs {reports['arvaniti']['graphs']}")
    if reports['lizard']['projection_scene_pt'] != reports['lizard']['graphs']:
        raise RuntimeError(f"Lizard projection_scene {reports['lizard']['projection_scene_pt']} != graphs {reports['lizard']['graphs']}")
    atomic_json(ar_root / '05_qc' / 'full_qc.json', reports)
    atomic_json(lz_root / '05_qc' / 'full_qc.json', reports)
    atomic_json(ar_root / '05_qc' / 'full_cache_status.json', reports)
    atomic_json(lz_root / '05_qc' / 'full_cache_status.json', reports)
    for result_root, data_root in ((ARVANITI_RESULT, ar_root), (LIZARD_RESULT, lz_root)):
        result_root.mkdir(parents=True, exist_ok=True)
        atomic_json(result_root / 'full_qc.json', reports)
        atomic_json(result_root / 'full_cache_status.json', reports)
    return reports

def write_engineering_qc() -> dict[str, Any]:
    reports = {}
    ar_root = Path(ARVANITI_DATA)
    windows = read_parquet(ar_root / '00_manifest' / 'engineering_windows.parquet')
    qc_dir = ar_root / '05_qc' / 'engineering'
    for row in windows[:24]:
        rgb = load_rgb(row['rgb_path'])
        mask = np.load(row['nucleus_mask_path'])
        overlay(rgb, mask, qc_dir / f"{row['graph_id']}.png")
        if rgb.shape[0] != 375 or rgb.shape[1] != 375:
            raise RuntimeError(f"{row['graph_id']} window is {rgb.shape}, expected 375")
    reports['arvaniti'] = {'overlays': min(24, len(windows)), 'windows': len(windows)}
    lz_root = Path(LIZARD_DATA)
    tiles = read_parquet(lz_root / '00_manifest' / 'engineering_tiles.parquet')
    nuclei = read_parquet(lz_root / '04_labels_splits' / 'engineering_nuclei.parquet')
    qc_dir = lz_root / '05_qc' / 'engineering'
    owned_keys = {}
    dup = 0
    for row in nuclei:
        if not row['owned']:
            continue
        key = (row['roi_id'], int(row['nucleus_id']))
        if key in owned_keys:
            dup += 1
        owned_keys[key] = row['graph_id']
    for row in tiles[:24]:
        rgb = load_rgb(row.get('dino_rgb_path') or row['rgb_path'])
        mask = np.load(row['nucleus_mask_path'])
        if rgb.shape[0] != mask.shape[0] or rgb.shape[1] != mask.shape[1]:
            rgb = load_rgb(row['dino_rgb_path'])
        overlay(rgb, mask, qc_dir / f"{row['graph_id']}.png")
    reports['lizard'] = {'overlays': min(24, len(tiles)), 'tiles': len(tiles), 'owned': len(owned_keys), 'duplicate_owned': dup}
    if dup:
        raise RuntimeError(f'Lizard ownership leaked: {dup} duplicate owned nuclei')
    atomic_json(ar_root / '05_qc' / 'engineering_qc.json', reports)
    atomic_json(lz_root / '05_qc' / 'engineering_qc.json', reports)
    return reports
