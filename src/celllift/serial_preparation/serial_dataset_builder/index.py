from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from typing import Any, Iterable, Mapping

def build_nucleus_link_index(layer_rows: Iterable[Mapping[str, Any]], link_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[int, int], dict[str, Any]] = {}
    layer_by_index = {int(r['layer_index']): r for r in layer_rows}
    for layer in layer_by_index.values():
        ids = layer.get('nucleus_ids')
        if ids is None:
            ids = range(1, int(layer.get('nucleus_count', 0)) + 1)
        for nid in ids:
            key = (int(layer['layer_index']), int(nid))
            index[key] = {'layer_index': key[0], 'roi_layer_id': layer.get('roi_layer_id', ''), 'nucleus_id': key[1], 'lower_edge_id': None, 'lower_neighbor_layer_index': None, 'lower_neighbor_nucleus_id': None, 'lower_quality_class': 'unknown', 'lower_cell_relation': 'not_evaluated', 'lower_cell_supervision_valid': False, 'upper_edge_id': None, 'upper_neighbor_layer_index': None, 'upper_neighbor_nucleus_id': None, 'upper_quality_class': 'unknown', 'upper_cell_relation': 'not_evaluated', 'upper_cell_supervision_valid': False, 'status': 'unknown'}
    for row in link_rows:
        if not row.get('accepted') or row.get('quality_class') not in {'gold', 'silver_nucleus'}:
            continue
        left_key = (int(row['left_layer_index']), int(row['left_nucleus_id']))
        right_key = (int(row['right_layer_index']), int(row['right_nucleus_id']))
        if left_key in index:
            index[left_key].update({'upper_edge_id': row['edge_id'], 'upper_neighbor_layer_index': right_key[0], 'upper_neighbor_nucleus_id': right_key[1], 'upper_quality_class': row['quality_class'], 'upper_cell_relation': row.get('cell_relation', 'not_evaluated'), 'upper_cell_supervision_valid': bool(row.get('cell_supervision_valid')), 'status': 'linked'})
        if right_key in index:
            index[right_key].update({'lower_edge_id': row['edge_id'], 'lower_neighbor_layer_index': left_key[0], 'lower_neighbor_nucleus_id': left_key[1], 'lower_quality_class': row['quality_class'], 'lower_cell_relation': row.get('cell_relation', 'not_evaluated'), 'lower_cell_supervision_valid': bool(row.get('cell_supervision_valid')), 'status': 'linked'})
    return [index[key] for key in sorted(index)]

def assert_endpoint_degree_at_most_one(link_rows: Iterable[Mapping[str, Any]]) -> None:
    left_seen: set[tuple[int, int, str]] = set()
    right_seen: set[tuple[int, int, str]] = set()
    for row in link_rows:
        if not row.get('accepted') or row.get('quality_class') not in {'gold', 'silver_nucleus'}:
            continue
        edge_id = str(row['edge_id'])
        lk = (int(row['left_layer_index']), int(row['left_nucleus_id']), edge_id)
        rk = (int(row['right_layer_index']), int(row['right_nucleus_id']), edge_id)
        if lk in left_seen:
            raise AssertionError(f'duplicate left endpoint {lk}')
        if rk in right_seen:
            raise AssertionError(f'duplicate right endpoint {rk}')
        left_seen.add(lk)
        right_seen.add(rk)
