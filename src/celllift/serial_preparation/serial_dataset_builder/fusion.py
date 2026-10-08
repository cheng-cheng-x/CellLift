from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from .matching_core import Candidate, Geometry, LayerContext, Match, coexist_mutual_max, endpoint_conflicts, lap_jaccard
DEFAULT_LAP_IOU_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_LAP_MARGIN_GRID = [0.0, 0.02, 0.05, 0.1]
LOCAL_RESIDUAL_LIMIT_PX = 12.0
ABS_LOG_AREA_RATIO_LIMIT = 1.2

@dataclass(frozen=True)
class Thresholds:
    nucleus_lap_iou: float
    nucleus_lap_margin: float
    local_residual_limit_px: float = LOCAL_RESIDUAL_LIMIT_PX
    abs_log_area_ratio_limit: float = ABS_LOG_AREA_RATIO_LIMIT
    lap_silver_enabled: bool = True

def _candidate_for(match: Match | None) -> Candidate | None:
    return match.candidate if match is not None else None

def _local_residual(candidate: Candidate | None) -> float | None:
    if candidate is None:
        return None
    return float(candidate.centroid_distance_px)

def _log_area_ratio(candidate: Candidate | None, fallback_left: Geometry | None=None, fallback_right: Geometry | None=None) -> float | None:
    if candidate is not None and candidate.area_ratio > 0:
        return abs(float(math.log(candidate.area_ratio)))
    if fallback_left is not None and fallback_right is not None and (fallback_left.area > 0) and (fallback_right.area > 0):
        return abs(float(math.log(fallback_left.area / fallback_right.area)))
    return None

def _assigned_cell(layer: LayerContext, nucleus_id: int | None) -> tuple[int | None, str]:
    if nucleus_id is None:
        return (None, 'missing')
    cid = layer.nucleus_to_cell.get(int(nucleus_id))
    if cid is None:
        return (None, 'missing')
    reason = layer.cell_exclusion_reason.get(int(cid))
    if reason:
        return (int(cid), 'artifact')
    if int(cid) not in layer.cell_geometry:
        return (int(cid), 'missing')
    return (int(cid), 'valid')

def cell_relation_for_nucleus_link(left: LayerContext, right: LayerContext, left_nucleus_id: int, right_nucleus_id: int, cell_matches_by_cell_id: Mapping[tuple[int, int], Match]) -> tuple[str, int | None, int | None, str | None, float | None]:
    left_cell, left_state = _assigned_cell(left, left_nucleus_id)
    right_cell, right_state = _assigned_cell(right, right_nucleus_id)
    if left_state == 'artifact' or right_state == 'artifact':
        if left_state == 'artifact' and right_state == 'artifact':
            relation = 'artifact_filtered_both'
        elif left_state == 'artifact':
            relation = 'artifact_filtered_left'
        else:
            relation = 'artifact_filtered_right'
        return (relation, left_cell, right_cell, None, None)
    if left_state != 'valid' or right_state != 'valid':
        if left_state != 'valid' and right_state != 'valid':
            relation = 'missing_both'
        elif left_state != 'valid':
            relation = 'missing_left'
        else:
            relation = 'missing_right'
        return (relation, left_cell, right_cell, None, None)
    assert left_cell is not None and right_cell is not None
    match = cell_matches_by_cell_id.get((left_cell, right_cell))
    if match is not None:
        return ('agree', left_cell, right_cell, match.method, float(match.score))
    return ('conflict', left_cell, right_cell, None, None)

def _row(*, edge: Mapping[str, Any], left: LayerContext, right: LayerContext, left_nucleus_id: int | None, right_nucleus_id: int | None, quality_class: str, nucleus_method: str | None, nucleus_match: Match | None, cell_relation: str, left_cell_id: int | None, right_cell_id: int | None, cell_method: str | None, cell_score: float | None, accepted: bool, reason_code: str) -> dict[str, Any]:
    candidate = _candidate_for(nucleus_match)
    return {'edge_id': str(edge['edge_id']), 'optimized_track_id': str(edge.get('optimized_track_id', edge.get('track_id', ''))), 'source_track_id': str(edge.get('source_track_id', '')), 'left_layer_index': int(edge['left_layer_index']), 'right_layer_index': int(edge['right_layer_index']), 'left_roi_layer_id': left.roi_layer_id, 'right_roi_layer_id': right.roi_layer_id, 'left_section_id': int(edge['left_section_id']), 'right_section_id': int(edge['right_section_id']), 'left_nucleus_id': left_nucleus_id, 'right_nucleus_id': right_nucleus_id, 'left_cell_id': left_cell_id, 'right_cell_id': right_cell_id, 'quality_class': quality_class, 'nucleus_method': nucleus_method, 'nucleus_iou': None if candidate is None else float(candidate.iou), 'nucleus_dice': None if candidate is None else float(candidate.dice), 'nucleus_margin': None if nucleus_match is None else float(nucleus_match.margin), 'local_residual_px': _local_residual(candidate), 'absolute_log_area_ratio': _log_area_ratio(candidate), 'cell_relation': cell_relation, 'cell_method': cell_method, 'cell_score': cell_score, 'nucleus_supervision_valid': bool(accepted and quality_class in {'gold', 'silver_nucleus'}), 'cell_supervision_valid': bool(accepted and cell_relation == 'agree'), 'accepted': bool(accepted), 'audit_status': 'accepted' if accepted else 'audit_only', 'reason_code': reason_code}

def _passes_geometry_gates(match: Match, thresholds: Thresholds) -> tuple[bool, str]:
    candidate = match.candidate
    if candidate is None:
        return (True, 'pass')
    local_residual = _local_residual(candidate)
    if local_residual is not None and local_residual > thresholds.local_residual_limit_px:
        return (False, 'failed_local_residual_gate')
    log_area = _log_area_ratio(candidate)
    if log_area is not None and log_area > thresholds.abs_log_area_ratio_limit:
        return (False, 'failed_abs_log_area_ratio_gate')
    return (True, 'pass')

def _strict_gold_pairs_from_joint_support(anchor_matches: Iterable[Match], left: LayerContext, right: LayerContext, cell_by_cell_id: Mapping[tuple[int, int], Match]) -> set[tuple[int, int]]:
    gold: set[tuple[int, int]] = set()
    for match in anchor_matches:
        relation, _, _, _, _ = cell_relation_for_nucleus_link(left, right, match.left_id, match.right_id, cell_by_cell_id)
        if relation == 'agree':
            gold.add((int(match.left_id), int(match.right_id)))
    return gold

def fuse_edge_minimal(edge: Mapping[str, Any], left: LayerContext, right: LayerContext, nucleus_candidates: Iterable[Candidate], cell_candidates: Iterable[Candidate], thresholds: Thresholds, *, strict_gold_pairs: Iterable[tuple[int, int]] | None=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nucleus_values = list(nucleus_candidates)
    cell_values = list(cell_candidates)
    anchor_matches = coexist_mutual_max(nucleus_values, method='coexist_nucleus_anchor')
    lap_matches = lap_jaccard(nucleus_values, thresholds.nucleus_lap_iou, margin_threshold=thresholds.nucleus_lap_margin, method='lap_jaccard_nucleus_recovery') if thresholds.lap_silver_enabled else []
    cell_matches = lap_jaccard(cell_values, 0.1, margin_threshold=0.0, method='lap_jaccard_cell_support')
    nucleus_by_id_anchor = {(m.left_id, m.right_id): m for m in anchor_matches}
    nucleus_by_id_lap = {(m.left_id, m.right_id): m for m in lap_matches}
    cell_by_cell_id = {(m.left_id, m.right_id): m for m in cell_matches}
    if strict_gold_pairs is None:
        gold_pairs = _strict_gold_pairs_from_joint_support(anchor_matches, left, right, cell_by_cell_id)
    else:
        gold_pairs = {(int(l), int(r)) for l, r in strict_gold_pairs}
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_left_nuclei: set[int] = set()
    used_right_nuclei: set[int] = set()

    def add_nucleus_link(left_nid: int, right_nid: int, quality: str, match: Match, reason: str) -> bool:
        if left_nid in used_left_nuclei or right_nid in used_right_nuclei:
            audit.append(_row(edge=edge, left=left, right=right, left_nucleus_id=left_nid, right_nucleus_id=right_nid, quality_class='unknown', nucleus_method=match.method, nucleus_match=match, cell_relation='not_evaluated', left_cell_id=None, right_cell_id=None, cell_method=None, cell_score=None, accepted=False, reason_code='unresolved_endpoint_conflict'))
            return False
        relation, lc, rc, cmethod, cscore = cell_relation_for_nucleus_link(left, right, left_nid, right_nid, cell_by_cell_id)
        accepted.append(_row(edge=edge, left=left, right=right, left_nucleus_id=left_nid, right_nucleus_id=right_nid, quality_class=quality, nucleus_method=match.method, nucleus_match=match, cell_relation=relation, left_cell_id=lc, right_cell_id=rc, cell_method=cmethod, cell_score=cscore, accepted=True, reason_code=reason))
        used_left_nuclei.add(left_nid)
        used_right_nuclei.add(right_nid)
        return True
    for pair in sorted(gold_pairs):
        match = nucleus_by_id_anchor.get(pair) or nucleus_by_id_lap.get(pair)
        if match is None:
            match = Match(pair[0], pair[1], 1.0, 1.0, 'frozen_strict_gold', None)
        ok, fail_reason = _passes_geometry_gates(match, thresholds)
        if not ok:
            audit.append(_row(edge=edge, left=left, right=right, left_nucleus_id=pair[0], right_nucleus_id=pair[1], quality_class='unknown', nucleus_method=match.method, nucleus_match=match, cell_relation='not_evaluated', left_cell_id=None, right_cell_id=None, cell_method=None, cell_score=None, accepted=False, reason_code=fail_reason))
            continue
        add_nucleus_link(pair[0], pair[1], 'gold', match, 'frozen_strict_gold_recomputed_on_serial')
    silver_candidates: dict[tuple[int, int], Match] = {}
    for m in anchor_matches:
        silver_candidates.setdefault((m.left_id, m.right_id), m)
    for m in lap_matches:
        silver_candidates.setdefault((m.left_id, m.right_id), m)
    conflicts = endpoint_conflicts({pair: match for pair, match in silver_candidates.items() if pair not in gold_pairs})
    for pair, match in sorted(silver_candidates.items()):
        if pair in gold_pairs:
            continue
        if pair in conflicts:
            audit.append(_row(edge=edge, left=left, right=right, left_nucleus_id=pair[0], right_nucleus_id=pair[1], quality_class='unknown', nucleus_method=match.method, nucleus_match=match, cell_relation='not_evaluated', left_cell_id=None, right_cell_id=None, cell_method=None, cell_score=None, accepted=False, reason_code='unresolved_same_tier_endpoint_conflict'))
            continue
        ok, fail_reason = _passes_geometry_gates(match, thresholds)
        if not ok:
            audit.append(_row(edge=edge, left=left, right=right, left_nucleus_id=pair[0], right_nucleus_id=pair[1], quality_class='unknown', nucleus_method=match.method, nucleus_match=match, cell_relation='not_evaluated', left_cell_id=None, right_cell_id=None, cell_method=None, cell_score=None, accepted=False, reason_code=fail_reason))
            continue
        if match.method == 'lap_jaccard_nucleus_recovery' and (match.score < thresholds.nucleus_lap_iou or match.margin < thresholds.nucleus_lap_margin):
            audit.append(_row(edge=edge, left=left, right=right, left_nucleus_id=pair[0], right_nucleus_id=pair[1], quality_class='unknown', nucleus_method=match.method, nucleus_match=match, cell_relation='not_evaluated', left_cell_id=None, right_cell_id=None, cell_method=None, cell_score=None, accepted=False, reason_code='failed_frozen_nucleus_lap_threshold'))
            continue
        add_nucleus_link(pair[0], pair[1], 'silver_nucleus', match, 'nucleus_evidence_without_cell_veto')
    for (lc, rc), cmatch in sorted(cell_by_cell_id.items()):
        left_nuclei = left.cell_assigned_nuclei.get(lc, [])
        right_nuclei = right.cell_assigned_nuclei.get(rc, [])
        if len(left_nuclei) == 1 and len(right_nuclei) == 1:
            if left_nuclei[0] in used_left_nuclei and right_nuclei[0] in used_right_nuclei:
                continue
        audit.append({'edge_id': str(edge['edge_id']), 'left_cell_id': lc, 'right_cell_id': rc, 'left_nucleus_id': left_nuclei[0] if len(left_nuclei) == 1 else None, 'right_nucleus_id': right_nuclei[0] if len(right_nuclei) == 1 else None, 'quality_class': 'cell_only', 'cell_method': cmatch.method, 'cell_score': float(cmatch.score), 'accepted': False, 'audit_status': 'audit_only', 'reason_code': 'cell_only_not_nucleus_positive'})
    return (accepted, audit)

def summarize_links(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[f"quality:{row.get('quality_class')}"] += 1
        counter[f"cell_relation:{row.get('cell_relation')}"] += 1
        if row.get('cell_supervision_valid'):
            counter['cell_supervision_valid'] += 1
        if row.get('nucleus_supervision_valid'):
            counter['nucleus_supervision_valid'] += 1
    return dict(sorted(counter.items()))
