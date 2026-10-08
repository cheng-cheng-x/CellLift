from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from .fusion import DEFAULT_LAP_IOU_GRID, DEFAULT_LAP_MARGIN_GRID, ABS_LOG_AREA_RATIO_LIMIT, LOCAL_RESIDUAL_LIMIT_PX, Thresholds
from .matching_core import Candidate, lap_jaccard

def _geometry_gate(c: Candidate, *, local_residual_limit_px: float=LOCAL_RESIDUAL_LIMIT_PX, abs_log_area_ratio_limit: float=ABS_LOG_AREA_RATIO_LIMIT) -> bool:
    if c.centroid_distance_px > local_residual_limit_px:
        return False
    if c.area_ratio <= 0:
        return False
    import math
    return abs(math.log(c.area_ratio)) <= abs_log_area_ratio_limit

@dataclass(frozen=True)
class CalibrationResult:
    status: str
    thresholds: Thresholds | None
    report: dict[str, object]

def calibrate_nucleus_lap_thresholds(true_candidates_by_edge: Mapping[str, Sequence[Candidate]], gold_pairs_by_edge: Mapping[str, set[tuple[int, int]]], null_candidates_by_edge: Mapping[str, Sequence[Candidate]], *, iou_grid: Sequence[float]=DEFAULT_LAP_IOU_GRID, margin_grid: Sequence[float]=DEFAULT_LAP_MARGIN_GRID, max_empirical_fdr: float=0.05, min_gold_retention: float=0.95) -> CalibrationResult:
    rows: list[dict[str, object]] = []
    total_gold = sum((len(v) for v in gold_pairs_by_edge.values()))
    for iou in iou_grid:
        for margin in margin_grid:
            recovered_gold = 0
            true_selected = 0
            null_selected = 0
            for edge_id, candidates in true_candidates_by_edge.items():
                selected = lap_jaccard([c for c in candidates if _geometry_gate(c)], iou, margin_threshold=margin)
                pairs = {(m.left_id, m.right_id) for m in selected}
                true_selected += len(pairs)
                recovered_gold += len(pairs & gold_pairs_by_edge.get(edge_id, set()))
            for candidates in null_candidates_by_edge.values():
                null_selected += len(lap_jaccard([c for c in candidates if _geometry_gate(c)], iou, margin_threshold=margin))
            fdr = null_selected / max(true_selected + null_selected, 1)
            retention = recovered_gold / max(total_gold, 1)
            rows.append({'nucleus_lap_iou': float(iou), 'nucleus_lap_margin': float(margin), 'true_selected': int(true_selected), 'null_selected': int(null_selected), 'gold_recovered': int(recovered_gold), 'gold_total': int(total_gold), 'empirical_fdr': float(fdr), 'gold_retention': float(retention), 'passes': bool(fdr <= max_empirical_fdr and retention >= min_gold_retention)})
    passing = [row for row in rows if row['passes']]
    if not passing:
        return CalibrationResult('NO_PASSING_THRESHOLD_DISABLE_LAP_SILVER', None, {'status': 'NO_PASSING_THRESHOLD_DISABLE_LAP_SILVER', 'selected_thresholds': {'nucleus_lap_iou': 1.0, 'nucleus_lap_margin': 1.0, 'local_residual_limit_px': LOCAL_RESIDUAL_LIMIT_PX, 'abs_log_area_ratio_limit': ABS_LOG_AREA_RATIO_LIMIT, 'lap_silver_enabled': False}, 'max_empirical_fdr': max_empirical_fdr, 'min_gold_retention': min_gold_retention, 'grid_results': rows})
    best = sorted(passing, key=lambda r: (int(r['true_selected']), float(r['nucleus_lap_iou']), float(r['nucleus_lap_margin'])), reverse=True)[0]
    thresholds = Thresholds(nucleus_lap_iou=float(best['nucleus_lap_iou']), nucleus_lap_margin=float(best['nucleus_lap_margin']))
    return CalibrationResult('PASS', thresholds, {'status': 'PASS', 'selected_thresholds': {'nucleus_lap_iou': thresholds.nucleus_lap_iou, 'nucleus_lap_margin': thresholds.nucleus_lap_margin, 'local_residual_limit_px': thresholds.local_residual_limit_px, 'abs_log_area_ratio_limit': thresholds.abs_log_area_ratio_limit, 'lap_silver_enabled': True}, 'max_empirical_fdr': max_empirical_fdr, 'min_gold_retention': min_gold_retention, 'grid_results': rows})

def _zero_fill_shift_labels(labels, *, dx: int, dy: int):
    import numpy as np
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError('label map must be 2D')
    out = np.zeros_like(arr)
    h, w = arr.shape
    if abs(dx) >= w or abs(dy) >= h:
        return out
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    if src_x0 < src_x1 and src_y0 < src_y1:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out

def _shift_geometry(geometry: Mapping[int, object], *, dx: int, dy: int) -> dict[int, object]:
    from dataclasses import replace
    shifted = {}
    for k, g in geometry.items():
        shifted[int(k)] = replace(g, centroid_x=float(g.centroid_x) + dx, centroid_y=float(g.centroid_y) + dy)
    return shifted

def _edge_shift(edge_id: str, shift_px: int) -> tuple[int, int]:
    from .common import stable_hash_u64
    if shift_px <= 0:
        raise ValueError('shift_px must be positive')
    shifts = [(shift_px, 0), (-shift_px, 0), (0, shift_px), (0, -shift_px), (shift_px, shift_px), (-shift_px, shift_px), (shift_px, -shift_px), (-shift_px, -shift_px)]
    return shifts[stable_hash_u64(edge_id) % len(shifts)]

def _aggregate_grid_rows(grid: list[dict[str, object]], true_candidates, gold_pairs: set[tuple[int, int]], null_candidates) -> None:
    for row in grid:
        iou = float(row['nucleus_lap_iou'])
        margin = float(row['nucleus_lap_margin'])
        selected = lap_jaccard([c for c in true_candidates if _geometry_gate(c)], iou, margin_threshold=margin)
        pairs = {(m.left_id, m.right_id) for m in selected}
        null_selected = lap_jaccard([c for c in null_candidates if _geometry_gate(c)], iou, margin_threshold=margin)
        row['true_selected'] = int(row['true_selected']) + len(pairs)
        row['null_selected'] = int(row['null_selected']) + len(null_selected)
        row['gold_recovered'] = int(row['gold_recovered']) + len(pairs & gold_pairs)

def calibrate_thresholds_from_manifests(*, layer_manifest: str, edge_manifest: str, output_json: str, max_edges: int | None=None, null_shift_px: int=8, iou_grid: Sequence[float]=DEFAULT_LAP_IOU_GRID, margin_grid: Sequence[float]=DEFAULT_LAP_MARGIN_GRID, max_empirical_fdr: float=0.05, min_gold_retention: float=0.95, progress_every: int=100) -> dict[str, object]:
    import sys
    from collections import Counter
    from celllift.runtime import ResourcePath as Path
    from .common import atomic_write_json, read_tsv
    from .fusion import _strict_gold_pairs_from_joint_support
    from .matching_core import candidate_metrics, coexist_mutual_max
    from .segmentation_context import load_layer_context_from_segmentation_row
    layer_rows = {int(r.get('layer_index', r.get('layer_idx'))): r for r in read_tsv(layer_manifest)}
    edge_rows = read_tsv(edge_manifest)
    if max_edges is not None:
        edge_rows = edge_rows[:int(max_edges)]
    grid_rows: list[dict[str, object]] = []
    for iou in iou_grid:
        for margin in margin_grid:
            grid_rows.append({'nucleus_lap_iou': float(iou), 'nucleus_lap_margin': float(margin), 'true_selected': 0, 'null_selected': 0, 'gold_recovered': 0, 'gold_total': 0, 'empirical_fdr': 1.0, 'gold_retention': 0.0, 'passes': False})
    counters: Counter[str] = Counter()
    total_gold = 0
    skipped: list[dict[str, object]] = []
    for idx, edge in enumerate(edge_rows, start=1):
        eid = str(edge['edge_id'])
        try:
            left = load_layer_context_from_segmentation_row(layer_rows[int(edge['left_layer_index'])])
            right = load_layer_context_from_segmentation_row(layer_rows[int(edge['right_layer_index'])])
            nucleus_candidates = candidate_metrics(left.nucleus_labels, right.nucleus_labels, left.nucleus_geometry, right.nucleus_geometry)
            cell_candidates = candidate_metrics(left.cell_labels, right.cell_labels, left.cell_geometry, right.cell_geometry)
            anchor_matches = coexist_mutual_max(nucleus_candidates, method='coexist_nucleus_anchor')
            cell_matches = lap_jaccard(cell_candidates, 0.1, margin_threshold=0.0, method='lap_jaccard_cell_support')
            cell_by_cell_id = {(m.left_id, m.right_id): m for m in cell_matches}
            gold_pairs = _strict_gold_pairs_from_joint_support(anchor_matches, left, right, cell_by_cell_id)
            dx, dy = _edge_shift(eid, null_shift_px)
            shifted_right_labels = _zero_fill_shift_labels(right.nucleus_labels, dx=dx, dy=dy)
            shifted_right_geometry = _shift_geometry(right.nucleus_geometry, dx=dx, dy=dy)
            null_candidates = [c for c in candidate_metrics(left.nucleus_labels, shifted_right_labels, left.nucleus_geometry, shifted_right_geometry) if (c.left_id, c.right_id) not in gold_pairs]
            total_gold += len(gold_pairs)
            _aggregate_grid_rows(grid_rows, nucleus_candidates, gold_pairs, null_candidates)
            counters['edges_used'] += 1
            counters['gold_pairs'] += len(gold_pairs)
            counters['true_candidates'] += len(nucleus_candidates)
            counters['null_candidates'] += len(null_candidates)
        except Exception as exc:
            counters['edges_skipped'] += 1
            if len(skipped) < 100:
                skipped.append({'edge_id': eid, 'reason': type(exc).__name__, 'message': str(exc)})
        if progress_every and idx % progress_every == 0:
            print(f"calibration progress: {idx}/{len(edge_rows)} edges, used={counters['edges_used']}, skipped={counters['edges_skipped']}, gold={total_gold}", file=sys.stderr, flush=True)
    for row in grid_rows:
        row['gold_total'] = int(total_gold)
        fdr = int(row['null_selected']) / max(int(row['true_selected']) + int(row['null_selected']), 1)
        retention = int(row['gold_recovered']) / max(total_gold, 1)
        row['empirical_fdr'] = float(fdr)
        row['gold_retention'] = float(retention)
        row['passes'] = bool(fdr <= max_empirical_fdr and retention >= min_gold_retention)
    passing = [row for row in grid_rows if row['passes']]
    if passing:
        best = sorted(passing, key=lambda r: (int(r['true_selected']), float(r['nucleus_lap_iou']), float(r['nucleus_lap_margin'])), reverse=True)[0]
        selected = {'nucleus_lap_iou': float(best['nucleus_lap_iou']), 'nucleus_lap_margin': float(best['nucleus_lap_margin']), 'local_residual_limit_px': LOCAL_RESIDUAL_LIMIT_PX, 'abs_log_area_ratio_limit': ABS_LOG_AREA_RATIO_LIMIT, 'lap_silver_enabled': True}
        status = 'PASS'
    else:
        selected = {'nucleus_lap_iou': 1.0, 'nucleus_lap_margin': 1.0, 'local_residual_limit_px': LOCAL_RESIDUAL_LIMIT_PX, 'abs_log_area_ratio_limit': ABS_LOG_AREA_RATIO_LIMIT, 'lap_silver_enabled': False}
        status = 'NO_PASSING_THRESHOLD_DISABLE_LAP_SILVER'
    report: dict[str, object] = {'status': status, 'selected_thresholds': selected, **selected, 'calibration_method': 'strict_gold_reference_plus_deterministic_spatial_shift_null', 'null_shift_px': int(null_shift_px), 'max_empirical_fdr': float(max_empirical_fdr), 'min_gold_retention': float(min_gold_retention), 'layer_manifest': str(Path(layer_manifest)), 'edge_manifest': str(Path(edge_manifest)), 'requested_edge_rows': len(edge_rows), 'counters': dict(sorted(counters.items())), 'skipped_examples': skipped, 'grid_results': grid_rows}
    atomic_write_json(output_json, report)
    return report
