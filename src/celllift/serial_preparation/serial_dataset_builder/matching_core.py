from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from .common import cell_entity_id, nucleus_entity_id

@dataclass(frozen=True)
class Geometry:
    instance_id: int
    area: int
    centroid_x: float
    centroid_y: float
    border: bool = False

@dataclass(frozen=True)
class Candidate:
    left_id: int
    right_id: int
    intersection_px: int
    dice: float
    iou: float
    source_containment: float
    target_containment: float
    centroid_distance_px: float
    area_ratio: float
    left_centroid_x: float = 0.0
    left_centroid_y: float = 0.0
    right_centroid_x: float = 0.0
    right_centroid_y: float = 0.0

@dataclass(frozen=True)
class Match:
    left_id: int
    right_id: int
    score: float
    margin: float
    method: str
    candidate: Candidate | None = None

@dataclass
class LayerContext:
    layer_index: int
    roi_layer_id: str
    nucleus_labels: np.ndarray | None
    cell_labels: np.ndarray | None
    nucleus_geometry: dict[int, Geometry]
    cell_geometry: dict[int, Geometry]
    all_cell_geometry: dict[int, Geometry]
    nucleus_to_cell: dict[int, int]
    cell_assigned_nuclei: dict[int, list[int]]
    cell_exclusion_reason: dict[int, str]

    @property
    def nucleus_to_entity(self) -> dict[int, int]:
        return {nid: nucleus_entity_id(self.layer_index, nid) for nid in self.nucleus_geometry}

    @property
    def cell_to_entity(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for cid in self.cell_geometry:
            nuclei = [nid for nid in self.cell_assigned_nuclei.get(cid, []) if nid in self.nucleus_geometry]
            if len(nuclei) == 1:
                result[cid] = nucleus_entity_id(self.layer_index, nuclei[0])
            else:
                result[cid] = cell_entity_id(self.layer_index, cid)
        return result

def geometry_from_label_map(labels: np.ndarray, *, border_ids: set[int] | None=None) -> dict[int, Geometry]:
    source = np.asarray(labels)
    if source.ndim != 2:
        raise ValueError('label map must be 2D')
    flat = source.ravel().astype(np.int64, copy=False)
    if flat.size == 0 or int(flat.max(initial=0)) == 0:
        return {}
    max_id = int(flat.max())
    counts = np.bincount(flat, minlength=max_id + 1)
    height, width = source.shape
    xs = np.tile(np.arange(width, dtype=np.float64), height)
    ys = np.repeat(np.arange(height, dtype=np.float64), width)
    xsum = np.bincount(flat, weights=xs, minlength=max_id + 1)
    ysum = np.bincount(flat, weights=ys, minlength=max_id + 1)
    result: dict[int, Geometry] = {}
    for raw_id in np.flatnonzero(counts):
        iid = int(raw_id)
        if iid == 0:
            continue
        area = int(counts[iid])
        result[iid] = Geometry(instance_id=iid, area=area, centroid_x=float(xsum[iid] / area), centroid_y=float(ysum[iid] / area), border=bool(border_ids and iid in border_ids))
    return result

def geometry_from_instances(payload: Mapping[str, Any]) -> dict[int, Geometry]:
    result: dict[int, Geometry] = {}
    for raw in payload.get('instances', []) or []:
        item = dict(raw)
        iid = int(item.get('instance_id', item.get('cell_id', item.get('nucleus_id'))))
        centroid = item.get('centroid_xy') or (item.get('centroid_x', 0.0), item.get('centroid_y', 0.0))
        result[iid] = Geometry(instance_id=iid, area=int(item.get('area_px', item.get('area', 0))), centroid_x=float(centroid[0]), centroid_y=float(centroid[1]), border=bool(item.get('border_flag', item.get('border', False))))
    return result

def pair_counts(left: np.ndarray, right: np.ndarray) -> dict[tuple[int, int], int]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError('label maps must have identical shape')
    valid = (left > 0) & (right > 0)
    if not bool(valid.any()):
        return {}
    encoded = left[valid].astype(np.uint64) << np.uint64(32) | right[valid].astype(np.uint64)
    values, counts = np.unique(encoded, return_counts=True)
    return {(int(v >> np.uint64(32)), int(v & np.uint64(4294967295))): int(c) for v, c in zip(values, counts)}

def candidate_metrics(left_labels: np.ndarray, right_labels: np.ndarray, left_geometry: Mapping[int, Geometry], right_geometry: Mapping[int, Geometry]) -> list[Candidate]:
    result: list[Candidate] = []
    for (left_id, right_id), inter in sorted(pair_counts(left_labels, right_labels).items()):
        if left_id not in left_geometry or right_id not in right_geometry:
            continue
        lg = left_geometry[left_id]
        rg = right_geometry[right_id]
        if lg.area <= 0 or rg.area <= 0 or inter <= 0:
            continue
        union = lg.area + rg.area - inter
        result.append(Candidate(left_id=left_id, right_id=right_id, intersection_px=inter, dice=2.0 * inter / (lg.area + rg.area), iou=inter / union if union > 0 else 0.0, source_containment=inter / lg.area, target_containment=inter / rg.area, centroid_distance_px=float(math.hypot(lg.centroid_x - rg.centroid_x, lg.centroid_y - rg.centroid_y)), area_ratio=float(lg.area / rg.area) if rg.area else 0.0, left_centroid_x=float(lg.centroid_x), left_centroid_y=float(lg.centroid_y), right_centroid_x=float(rg.centroid_x), right_centroid_y=float(rg.centroid_y)))
    return result

def _margin(left_id: int, right_id: int, score: float, candidates: Sequence[Candidate], attr: str='iou') -> float:
    competitors = [float(getattr(c, attr)) for c in candidates if (c.left_id == left_id or c.right_id == right_id) and (not (c.left_id == left_id and c.right_id == right_id))]
    return float(score - max(competitors)) if competitors else 1.0

def coexist_mutual_max(candidates: Iterable[Candidate], *, method: str='coexist_nucleus_anchor') -> list[Match]:
    values = [c for c in candidates if c.intersection_px > 0]
    by_left: dict[int, list[Candidate]] = defaultdict(list)
    by_right: dict[int, list[Candidate]] = defaultdict(list)
    for c in values:
        by_left[c.left_id].append(c)
        by_right[c.right_id].append(c)

    def best(items: list[Candidate]) -> tuple[Candidate, bool]:
        ordered = sorted(items, key=lambda c: (-c.intersection_px, c.centroid_distance_px, c.right_id, c.left_id))
        chosen = ordered[0]
        tied = len(ordered) > 1 and ordered[1].intersection_px == chosen.intersection_px and math.isclose(ordered[1].centroid_distance_px, chosen.centroid_distance_px, rel_tol=0.0, abs_tol=1e-12)
        return (chosen, tied)
    fwd = {lid: best(items) for lid, items in by_left.items()}
    rev = {rid: best(items) for rid, items in by_right.items()}
    out: list[Match] = []
    for lid, (c, fwd_tied) in sorted(fwd.items()):
        reverse = rev.get(c.right_id)
        if reverse is None:
            continue
        rbest, rev_tied = reverse
        if rbest.left_id != lid or fwd_tied or rev_tied:
            continue
        out.append(Match(lid, c.right_id, float(c.dice), _margin(lid, c.right_id, float(c.dice), values, 'dice'), method, c))
    return out

def lap_jaccard(candidates: Iterable[Candidate], threshold: float, *, margin_threshold: float=0.0, method: str='lap_jaccard_nucleus_recovery') -> list[Match]:
    values = [c for c in candidates if c.intersection_px > 0 and c.iou >= threshold]
    if not values:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        left_ids = sorted({c.left_id for c in values})
        right_ids = sorted({c.right_id for c in values})
        left_pos = {v: i for i, v in enumerate(left_ids)}
        right_pos = {v: i for i, v in enumerate(right_ids)}
        cost = np.full((len(left_ids), len(right_ids)), 1000000.0, dtype=np.float64)
        by_pair = {(c.left_id, c.right_id): c for c in values}
        for c in values:
            cost[left_pos[c.left_id], right_pos[c.right_id]] = -float(c.iou)
        rows, cols = linear_sum_assignment(cost)
        selected = [by_pair[left_ids[r], right_ids[c]] for r, c in zip(rows, cols) if cost[r, c] < 1000000.0]
    except Exception:
        selected = []
        used_left: set[int] = set()
        used_right: set[int] = set()
        for c in sorted(values, key=lambda x: (-x.iou, -x.intersection_px, x.centroid_distance_px, x.left_id, x.right_id)):
            if c.left_id in used_left or c.right_id in used_right:
                continue
            selected.append(c)
            used_left.add(c.left_id)
            used_right.add(c.right_id)
    out: list[Match] = []
    for c in sorted(selected, key=lambda x: (x.left_id, x.right_id)):
        margin = _margin(c.left_id, c.right_id, float(c.iou), values, 'iou')
        if margin >= margin_threshold:
            out.append(Match(c.left_id, c.right_id, float(c.iou), margin, method, c))
    return out

def map_matches(matches: Iterable[Match], left_map: Mapping[int, int], right_map: Mapping[int, int]) -> dict[tuple[int, int], Match]:
    result: dict[tuple[int, int], Match] = {}
    for match in matches:
        if match.left_id in left_map and match.right_id in right_map:
            result[left_map[match.left_id], right_map[match.right_id]] = match
    return result

def endpoint_conflicts(matches: Mapping[tuple[int, int], Match]) -> set[tuple[int, int]]:
    by_left: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_right: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in matches:
        by_left[pair[0]].append(pair)
        by_right[pair[1]].append(pair)
    conflicts: set[tuple[int, int]] = set()
    for pairs in list(by_left.values()) + list(by_right.values()):
        if len(pairs) > 1:
            conflicts.update(pairs)
    return conflicts
