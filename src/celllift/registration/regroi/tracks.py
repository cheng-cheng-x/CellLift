from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from .core import read_tsv, result_root, stage_status, write_json, write_tsv
from .globalreg import warp_mask

def intervals(values: list[bool], breaks: set[int]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    for zero_index in range(len(values) + 1):
        value = values[zero_index] if zero_index < len(values) else False
        section = zero_index + 1
        if zero_index > 0 and zero_index in breaks and (start is not None):
            output.append((start, zero_index))
            start = section if value else None
            continue
        if value and start is None:
            start = section
        elif not value and start is not None:
            output.append((start, zero_index))
            start = None
    return output

def _integral_area(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    x0 = max(0, min(integral.shape[1] - 1, x0))
    x1 = max(0, min(integral.shape[1] - 1, x1))
    y0 = max(0, min(integral.shape[0] - 1, y0))
    y1 = max(0, min(integral.shape[0] - 1, y1))
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])

def _source_support(inverse_level0: np.ndarray, x: int, y: int, size: int, width: int, height: int) -> float:
    corners = np.array([[x, y, 1], [x + size, y, 1], [x, y + size, 1], [x + size, y + size, 1]], float).T
    points = inverse_level0 @ corners
    points = points[:2] / points[2]
    valid = (points[0] >= 0) & (points[0] < width) & (points[1] >= 0) & (points[1] < height)
    return float(valid.mean())

def run_track_enumerate(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    transforms = read_tsv(root / '00_manifests/global_transform_manifest.tsv')
    edges = read_tsv(root / '00_manifests/global_edge_manifest.tsv')
    if len(transforms) != 260 or len(edges) != 259:
        raise RuntimeError('global registration must complete before track enumeration')
    mask_root = root / '02_thumbnails_order_qc/masks_10um'
    preview_root = root / '04_global_registration/previews'
    global_masks, global_dark = ([], [])
    for index, row in enumerate(transforms):
        mask = np.asarray(Image.open(mask_root / f'{index + 1:03d}.png')) > 0
        matrix = np.asarray(json.loads(row['source_to_global_canvas10um_json']), float)
        global_masks.append(warp_mask(mask, matrix, mask.shape))
        preview = np.asarray(Image.open(preview_root / f'{index + 1:03d}.jpg').convert('L'))
        global_dark.append(preview < 35)
    tissue_integrals = [cv2.integral(mask.astype(np.uint8)) for mask in global_masks]
    dark_integrals = [cv2.integral(mask.astype(np.uint8)) for mask in global_dark]
    scale = float(cfg['physical']['mpp_um_per_px']) / float(cfg['physical']['global_mpp_um_per_px'])
    roi_px = int(cfg['physical']['roi_px'])
    thumb_roi = max(1, int(round(roi_px * scale)))
    width = int(cfg['physical']['width_px'])
    height = int(cfg['physical']['height_px'])
    cols = (width - roi_px) // roi_px + 1
    rows = (height - roi_px) // roi_px + 1
    if (cols, rows) != (14, 21):
        raise RuntimeError(f'frozen theoretical grid mismatch: {(cols, rows)}')
    min_tissue = float(cfg['tracks']['min_tissue_fraction'])
    max_dark = float(cfg['tracks']['max_dark_fraction'])
    availability_rows = []
    by_cell: dict[tuple[int, int], list[bool]] = {}
    metrics_by_cell_section: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row_index in range(rows):
        for col_index in range(cols):
            x = col_index * roi_px
            y = row_index * roi_px
            tx0, ty0 = (int(round(x * scale)), int(round(y * scale)))
            tx1, ty1 = (tx0 + thumb_roi, ty0 + thumb_roi)
            values = []
            for section_index, transform in enumerate(transforms, start=1):
                area = max(1, (tx1 - tx0) * (ty1 - ty0))
                tissue = _integral_area(tissue_integrals[section_index - 1], tx0, ty0, tx1, ty1) / area
                dark = _integral_area(dark_integrals[section_index - 1], tx0, ty0, tx1, ty1) / area
                inverse = np.asarray(json.loads(transform['global_level0_to_source_json']), float)
                support = _source_support(inverse, x, y, roi_px, width, height)
                available = tissue >= min_tissue and dark <= max_dark and (support == 1.0)
                values.append(available)
                record = {'grid_row': row_index, 'grid_col': col_index, 'global_x': x, 'global_y': y, 'section_id': f'{section_index:03d}', 'tissue_fraction': tissue, 'dark_fraction': dark, 'source_support_fraction': support, 'available': available, 'exclusion_reason': '' if available else 'low_tissue' if tissue < min_tissue else 'dark_artifact' if dark > max_dark else 'incomplete_source_support'}
                availability_rows.append(record)
                metrics_by_cell_section[row_index, col_index, section_index] = record
            by_cell[row_index, col_index] = values
    availability_path = root / '05_track_candidates/grid_layer_availability.tsv'
    write_tsv(availability_path, availability_rows)
    failed_breaks = {int(row['edge_index']) for row in edges if str(row['status']).lower() != 'pass'}
    minimum = int(cfg['tracks']['min_interval_layers'])
    tracks = []
    exclusions = []
    counter = 0
    for (row_index, col_index), values in sorted(by_cell.items()):
        for start, end in intervals(values, failed_breaks):
            length = end - start + 1
            if length < minimum:
                exclusions.append({'grid_row': row_index, 'grid_col': col_index, 'start_section': start, 'end_section': end, 'length': length, 'reason': 'continuous_interval_shorter_than_three'})
                continue
            counter += 1
            seed = (start + end) // 2
            tracks.append({'track_id': f'track_{counter:05d}', 'grid_row': row_index, 'grid_col': col_index, 'global_x': col_index * roi_px, 'global_y': row_index * roi_px, 'w': roi_px, 'h': roi_px, 'start_section': start, 'end_section': end, 'seed_section': seed, 'layer_count': length, 'edge_count': length - 1, 'triplet_capacity': length - 2, 'parent_row': row_index // 2, 'parent_col': col_index // 2, 'parent_id': f'parent_r{row_index // 2:02d}_c{col_index // 2:02d}', 'status': 'pending_local_registration'})
    write_tsv(root / '00_manifests/track_manifest.tsv', tracks)
    write_tsv(root / '05_track_candidates/excluded_intervals.tsv', exclusions)
    active_by_parent_edge: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    track_by_id = {row['track_id']: row for row in tracks}
    for track in tracks:
        for edge_index in range(int(track['start_section']), int(track['end_section'])):
            active_by_parent_edge[int(track['parent_row']), int(track['parent_col']), edge_index].append(track['track_id'])
    jobs = []
    for job_index, ((parent_row, parent_col, edge_index), members) in enumerate(sorted(active_by_parent_edge.items()), start=1):
        core_x, core_y = (parent_col * 2 * roi_px, parent_row * 2 * roi_px)
        context_size = int(cfg['tracks']['parent_context_px'])
        context_x = core_x - (context_size - int(cfg['tracks']['parent_core_px'])) // 2
        context_y = core_y - (context_size - int(cfg['tracks']['parent_core_px'])) // 2
        seeds = [int(track_by_id[item]['seed_section']) for item in members]
        distance = min((abs(edge_index - seed) for seed in seeds))
        jobs.append({'job_id': f'job_{job_index:07d}', 'parent_id': f'parent_r{parent_row:02d}_c{parent_col:02d}', 'parent_row': parent_row, 'parent_col': parent_col, 'left_section': edge_index, 'right_section': edge_index + 1, 'member_track_ids': ';'.join(sorted(members)), 'member_count': len(members), 'context_global_x': context_x, 'context_global_y': context_y, 'context_size_px': context_size, 'wave_distance_from_seed': distance, 'status': 'pending', 'attempts': 0, 'terminal_reason': ''})
    jobs.sort(key=lambda row: (int(row['wave_distance_from_seed']), int(row['left_section']), row['parent_id']))
    write_tsv(root / '00_manifests/edge_job_manifest.tsv', jobs)
    write_json(root / '05_track_candidates/track_enumeration_summary.json', {'status': 'PASS', 'grid_columns': cols, 'grid_rows': rows, 'theoretical_layer_rois': cols * rows * 260, 'theoretical_triplets': cols * rows * 258, 'track_count': len(tracks), 'excluded_short_intervals': len(exclusions), 'triplet_capacity': sum((int(row['triplet_capacity']) for row in tracks)), 'unique_local_parent_edge_jobs': len(jobs), 'independent_triplet_pair_jobs_avoided': 2 * sum((int(row['triplet_capacity']) for row in tracks)) - len(jobs)})
    stage_status(cfg, 'track_enumerate', 'PASS', track_count=len(tracks), job_count=len(jobs), triplet_capacity=sum((int(row['triplet_capacity']) for row in tracks)))
