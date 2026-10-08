from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .artifact_filter import EXCLUSION_REASON, mark_artifact_cells
from .common import parse_int, read_gzip_json
from .matching_core import LayerContext, geometry_from_instances

def load_layer_context_from_segmentation_row(row: Mapping[str, Any], *, apply_artifact_filter: bool=True) -> LayerContext:
    nucleus_labels = np.load(str(row['nucleus_mask_path']), mmap_mode='r', allow_pickle=False)
    cell_labels = np.load(str(row['cell_mask_path']), mmap_mode='r', allow_pickle=False)
    nucleus_payload = read_gzip_json(row['nucleus_instances_path'])
    cell_payload = read_gzip_json(row['cell_instances_path'])
    pair_payload = read_gzip_json(row['pairs_path'])
    if apply_artifact_filter:
        pair_payload = mark_artifact_cells(pair_payload)
    nucleus_geometry = geometry_from_instances(nucleus_payload)
    all_cell_geometry = geometry_from_instances(cell_payload)
    nucleus_to_cell: dict[int, int] = {}
    for rel in pair_payload.get('nucleus_relations', []) or []:
        cell_id = rel.get('cell_id')
        if cell_id not in {None, ''}:
            nucleus_to_cell[int(rel['nucleus_id'])] = int(cell_id)
    cell_assigned_nuclei: dict[int, list[int]] = {}
    cell_exclusion_reason: dict[int, str] = {}
    valid_cell_geometry = dict(all_cell_geometry)
    for rel in pair_payload.get('cell_relations', []) or []:
        cid = int(rel['cell_id'])
        cell_assigned_nuclei[cid] = [int(v) for v in rel.get('assigned_nucleus_ids', []) or []]
        reason = str(rel.get('cell_exclusion_reason', '') or '')
        if rel.get('cell_valid_for_matching') is False or reason == EXCLUSION_REASON:
            cell_exclusion_reason[cid] = reason or EXCLUSION_REASON
            valid_cell_geometry.pop(cid, None)
    return LayerContext(layer_index=parse_int(row.get('layer_index', row.get('layer_idx'))), roi_layer_id=str(row['roi_layer_id']), nucleus_labels=nucleus_labels, cell_labels=cell_labels, nucleus_geometry=nucleus_geometry, cell_geometry=valid_cell_geometry, all_cell_geometry=all_cell_geometry, nucleus_to_cell=nucleus_to_cell, cell_assigned_nuclei=cell_assigned_nuclei, cell_exclusion_reason=cell_exclusion_reason)

def layer_summary_from_segmentation_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = mark_artifact_cells(read_gzip_json(row['pairs_path']))
    summary = payload.get('summary', {}) or {}
    return {'layer_index': parse_int(row.get('layer_index', row.get('layer_idx'))), 'roi_layer_id': row['roi_layer_id'], 'optimized_track_id': row.get('optimized_track_id', row.get('track_id', '')), 'source_track_id': row.get('source_track_id', ''), 'section_id': parse_int(row['section_id']), 'registered_png_path': row.get('registered_png_path', row.get('source_png_path', '')), 'registered_png_sha256': row.get('registered_png_sha256', row.get('source_png_sha256', '')), 'nucleus_mask_path': row.get('nucleus_mask_path', ''), 'nucleus_mask_sha256': row.get('nucleus_mask_sha256', ''), 'cell_mask_path': row.get('cell_mask_path', ''), 'cell_mask_sha256': row.get('cell_mask_sha256', ''), 'pairs_path': row.get('pairs_path', ''), 'pairs_sha256': row.get('pairs_sha256', ''), 'nucleus_count': parse_int(row.get('nucleus_count')), 'raw_cell_count': parse_int(summary.get('raw_cell_count', row.get('cell_count'))), 'valid_cell_count': parse_int(summary.get('valid_cell_count', row.get('cell_count'))), 'artifact_cell_count': parse_int(summary.get('artifact_cell_count', 0)), 'registration_visual_qc': row.get('registration_visual_qc', 'PASS'), 'segmentation_status': row.get('segmentation_status', 'complete'), 'algorithm_fingerprint': row.get('algorithm_fingerprint', '')}
