from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter
from typing import Any, Iterable, Mapping
LARGE_NUCLEUS_FREE_CELL_AREA_PX = 816
EXCLUSION_REASON = 'large_nucleus_free_artifact'

def is_large_nucleus_free_cell(cell_relation: Mapping[str, Any], *, area_threshold_px: int=LARGE_NUCLEUS_FREE_CELL_AREA_PX) -> bool:
    assigned = cell_relation.get('assigned_nucleus_count')
    if assigned is None:
        assigned = len(cell_relation.get('assigned_nucleus_ids', []) or [])
    area = int(cell_relation.get('cell_area_px', cell_relation.get('area_px', 0)) or 0)
    return int(assigned) == 0 and area > int(area_threshold_px)

def mark_artifact_cells(pair_payload: Mapping[str, Any], *, area_threshold_px: int=LARGE_NUCLEUS_FREE_CELL_AREA_PX) -> dict[str, Any]:
    payload = dict(pair_payload)
    new_cell_relations: list[dict[str, Any]] = []
    artifact_ids: list[int] = []
    for raw in payload.get('cell_relations', []) or []:
        row = dict(raw)
        if is_large_nucleus_free_cell(row, area_threshold_px=area_threshold_px):
            row['cell_valid_for_matching'] = False
            row['cell_exclusion_reason'] = EXCLUSION_REASON
            artifact_ids.append(int(row['cell_id']))
        else:
            row['cell_valid_for_matching'] = True
            row['cell_exclusion_reason'] = ''
        new_cell_relations.append(row)
    payload['cell_relations'] = new_cell_relations
    raw_count = len(new_cell_relations)
    valid_count = sum((1 for row in new_cell_relations if row.get('cell_valid_for_matching')))
    summary = dict(payload.get('summary', {}) or {})
    summary.update({'artifact_filter_rule': f'assigned_nucleus_count==0 AND cell_area_px>{area_threshold_px}', 'raw_cell_count': raw_count, 'valid_cell_count': valid_count, 'artifact_cell_count': raw_count - valid_count, 'large_nucleus_free_artifact_cell_ids': artifact_ids})
    payload['summary'] = summary
    return payload

def layer_artifact_summary(pair_payload: Mapping[str, Any]) -> dict[str, int]:
    rows = list(pair_payload.get('cell_relations', []) or [])
    raw_count = len(rows)
    artifact_count = sum((1 for row in rows if row.get('cell_exclusion_reason') == EXCLUSION_REASON or row.get('cell_valid_for_matching') is False))
    return {'raw_cell_count': raw_count, 'valid_cell_count': raw_count - artifact_count, 'artifact_cell_count': artifact_count}

def exclusion_manifest_rows(layer_rows: Iterable[Mapping[str, Any]], payload_by_layer: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in layer_rows:
        rid = str(layer['roi_layer_id'])
        payload = payload_by_layer.get(rid, {})
        for cell in payload.get('cell_relations', []) or []:
            if cell.get('cell_exclusion_reason') == EXCLUSION_REASON or cell.get('cell_valid_for_matching') is False:
                rows.append({'roi_layer_id': rid, 'optimized_track_id': layer.get('optimized_track_id', layer.get('track_id', '')), 'section_id': layer.get('section_id', ''), 'cell_id': int(cell['cell_id']), 'cell_area_px': int(cell.get('cell_area_px', 0)), 'assigned_nucleus_count': int(cell.get('assigned_nucleus_count', len(cell.get('assigned_nucleus_ids', []) or []))), 'cell_exclusion_reason': EXCLUSION_REASON})
    return rows
