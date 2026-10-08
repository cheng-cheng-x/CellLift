from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import numpy as np
from .barcodes import filename_stem
from . import paths
from .io_utils import write_json
from .protocol import FIELD_UM, IMAGE_SIZE, MAX_TILES_PER_PATIENT, MIN_TILES_EVALUABLE, PROTOCOL_ID, TARGET_MPP, THUMB_MPP, TILE_SEED, TISSUE_BRIGHTNESS_MAX, TISSUE_FRACTION_MIN, TISSUE_SATURATION_MIN

def level0_size(native_mpp: float) -> int:
    return int(round(IMAGE_SIZE * TARGET_MPP / float(native_mpp)))

def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.zeros_like(maximum)
    np.divide(maximum - minimum, maximum, out=saturation, where=maximum > 0)
    brightness = array.mean(axis=2)
    return (brightness < TISSUE_BRIGHTNESS_MAX) & (saturation > TISSUE_SATURATION_MIN)

def candidate_windows(width: int, height: int, native_mpp: float) -> list[tuple[int, int, int]]:
    size = level0_size(native_mpp)
    if size <= 0 or width < size or height < size:
        return []
    return [(x, y, size) for y in range(0, height - size + 1, size) for x in range(0, width - size + 1, size)]

def thumbnail_fraction(mask: np.ndarray, x0: int, y0: int, size: int, width0: int, height0: int) -> float:
    thumb_h, thumb_w = mask.shape
    x1 = int(round(x0 * thumb_w / width0))
    y1 = int(round(y0 * thumb_h / height0))
    w1 = max(1, int(round(size * thumb_w / width0)))
    h1 = max(1, int(round(size * thumb_h / height0)))
    patch = mask[y1:y1 + h1, x1:x1 + w1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())

def kmeans_select(candidates: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    if k <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: row['tile_id'])
    if len(ordered) <= k:
        return ordered
    points = np.asarray([[row['x_um'], row['y_um']] for row in ordered], dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = points[rng.choice(len(points), size=k, replace=False)]
    for _ in range(40):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        new_centers = np.array([points[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    used: set[int] = set()
    chosen: list[dict[str, Any]] = []
    for center in centers:
        dist = ((points - center) ** 2).sum(axis=1)
        ranking = sorted(range(len(ordered)), key=lambda i: (i in used, float(dist[i]), -ordered[i]['tissue_frac'], ordered[i]['tile_id']))
        pick = ranking[0]
        used.add(pick)
        chosen.append(ordered[pick])
    return sorted(chosen, key=lambda row: row['tile_id'])

def waterfill(capacities: list[int], budget: int) -> list[int]:
    alloc = [0] * len(capacities)
    if not capacities or budget <= 0:
        return alloc
    remaining = True
    while sum(alloc) < budget and remaining:
        remaining = False
        for i, cap in enumerate(capacities):
            if alloc[i] < cap and sum(alloc) < budget:
                alloc[i] += 1
                remaining = True
    return alloc

def slide_candidates(slide: dict[str, Any], rgb: np.ndarray) -> list[dict[str, Any]]:
    width = int(slide['width0'])
    height = int(slide['height0'])
    native_mpp = float(slide['mpp_x'])
    mask = tissue_mask(rgb)
    stem = filename_stem(slide['slide_id'])
    kept = []
    for x0, y0, size in candidate_windows(width, height, native_mpp):
        frac = thumbnail_fraction(mask, x0, y0, size, width, height)
        if frac < TISSUE_FRACTION_MIN:
            continue
        tile_id = f'{stem}_x{x0}_y{y0}'
        kept.append({'tile_id': tile_id, 'graph_id': f'tcga_brca_set_encoding:{tile_id}', 'patient_id': slide['patient_id'], 'slide_id': slide['slide_id'], 'source_path': slide['source_path'], 'native_mpp': native_mpp, 'level0_x': x0, 'level0_y': y0, 'level0_size': size, 'target_mpp': TARGET_MPP, 'target_size': IMAGE_SIZE, 'field_um': FIELD_UM, 'tissue_frac': frac, 'x_um': (x0 + size / 2.0) * native_mpp, 'y_um': (y0 + size / 2.0) * native_mpp})
    return kept

def read_thumbnail(path: str, width0: int, height0: int, native_mpp: float) -> np.ndarray:
    import openslide
    slide = openslide.OpenSlide(path)
    try:
        thumb_w = max(1, int(round(width0 * native_mpp / THUMB_MPP)))
        thumb_h = max(1, int(round(height0 * native_mpp / THUMB_MPP)))
        return np.asarray(slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB'))
    finally:
        slide.close()

def select_patient_tiles(candidates: list[dict[str, Any]], seed: int=TILE_SEED) -> list[dict[str, Any]]:
    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_slide[row['slide_id']].append(row)
    slide_ids = sorted(by_slide)
    capacities = [len(by_slide[slide_id]) for slide_id in slide_ids]
    alloc = waterfill(capacities, MAX_TILES_PER_PATIENT)
    selected: list[dict[str, Any]] = []
    for slide_id, count in zip(slide_ids, alloc):
        selected.extend(kmeans_select(by_slide[slide_id], count, seed))
    return sorted(selected, key=lambda row: row['tile_id'])

def _load_split_map(root: Path) -> dict[str, str]:
    units = json.loads((paths.split_dir(root) / 'units.json').read_text(encoding='utf-8'))[0]
    mapping = {}
    for part in ('fit', 'val', 'test'):
        for pid in units[part]:
            mapping[pid] = part
    return mapping

def sample_slide_shard(slide: dict[str, Any], shard_dir: Path) -> dict[str, Any]:
    dest = shard_dir / f"{filename_stem(slide['slide_id'])}.json"
    if dest.exists():
        return json.loads(dest.read_text(encoding='utf-8'))
    record: dict[str, Any] = {'slide_id': slide['slide_id'], 'patient_id': slide['patient_id'], 'status': 'ok', 'n_candidates': 0, 'error': '', 'candidates': []}
    try:
        rgb = read_thumbnail(slide['source_path'], int(slide['width0']), int(slide['height0']), float(slide['mpp_x']))
        candidates = slide_candidates(slide, rgb)
        record['n_candidates'] = len(candidates)
        record['candidates'] = candidates
    except Exception as exc:
        record['status'] = 'error'
        record['error'] = f'{type(exc).__name__}: {exc}'
    dest.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    return record

def sample_tile_shards(root: Path | None=None, shard_index: int=0, shard_count: int=1) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    slides = pd.read_parquet(paths.inventory_dir(base) / 'eligible_slides.parquet').to_dict('records')
    slides = [row for i, row in enumerate(slides) if i % shard_count == shard_index]
    shard_dir = paths.tile_dir(base) / 'shards'
    done = 0
    errors = 0
    for slide in slides:
        record = sample_slide_shard(slide, shard_dir)
        done += 1
        errors += int(record['status'] != 'ok')
    return {'shard_index': shard_index, 'n_slides': done, 'n_errors': errors}

def merge_tiles(root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    labels = pd.read_parquet(paths.label_dir(base) / 'patient_labels.parquet')
    master_ids = set(labels.loc[labels['in_master'], 'patient_id'])
    split_map = _load_split_map(base)
    shard_dir = paths.tile_dir(base) / 'shards'
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slide_status = []
    for path in sorted(shard_dir.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        slide_status.append({k: record[k] for k in ('slide_id', 'patient_id', 'status', 'n_candidates', 'error')})
        if record['status'] == 'ok':
            for row in record['candidates']:
                if row['patient_id'] in master_ids:
                    by_patient[row['patient_id']].append(row)
    selected_rows = []
    counts = []
    for pid in sorted(master_ids):
        chosen = select_patient_tiles(by_patient.get(pid, []))
        for row in chosen:
            row = dict(row)
            row['split'] = split_map.get(pid, '')
            selected_rows.append(row)
        counts.append({'patient_id': pid, 'n_tiles': len(chosen), 'split': split_map.get(pid, '')})
    unevaluable = [row for row in counts if row['n_tiles'] < MIN_TILES_EVALUABLE]
    dest = paths.tile_dir(base)
    pd.DataFrame(selected_rows).to_parquet(dest / 'tile_manifest.parquet', index=False)
    pd.DataFrame(counts).to_parquet(dest / 'patient_tile_counts.parquet', index=False)
    summary = {'protocol_id': PROTOCOL_ID, 'n_tiles': len(selected_rows), 'n_patients': len(counts), 'n_unevaluable': len(unevaluable), 'min_tiles': MIN_TILES_EVALUABLE, 'max_tiles': MAX_TILES_PER_PATIENT, 'target_mpp': TARGET_MPP, 'image_size': IMAGE_SIZE, 'n_slide_errors': sum((1 for row in slide_status if row['status'] != 'ok'))}
    write_json(dest / 'unevaluable.json', unevaluable)
    write_json(dest / 'slide_status.json', slide_status)
    write_json(dest / 'summary.json', summary)
    return summary
