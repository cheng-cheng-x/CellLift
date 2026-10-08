from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from collections import Counter
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from .common import atomic_write_json, read_json, read_tsv, sha256_file, stable_bucket
from .fusion import Thresholds, fuse_edge_minimal
from .matching_core import candidate_metrics
from .production_io import write_table
from .schemas import NUCLEUS_LINK_FIELDS, pyarrow_schemas
from .segmentation_context import load_layer_context_from_segmentation_row

def thresholds_from_json(path: str | Path) -> Thresholds:
    payload = read_json(path, {}) or {}
    return Thresholds(nucleus_lap_iou=float(payload.get('nucleus_lap_iou', payload.get('selected_thresholds', {}).get('nucleus_lap_iou', 0.1))), nucleus_lap_margin=float(payload.get('nucleus_lap_margin', payload.get('selected_thresholds', {}).get('nucleus_lap_margin', 0.02))), local_residual_limit_px=float(payload.get('local_residual_limit_px', payload.get('selected_thresholds', {}).get('local_residual_limit_px', 12.0))), abs_log_area_ratio_limit=float(payload.get('abs_log_area_ratio_limit', payload.get('selected_thresholds', {}).get('abs_log_area_ratio_limit', 1.2))), lap_silver_enabled=bool(payload.get('lap_silver_enabled', payload.get('selected_thresholds', {}).get('lap_silver_enabled', True))))

def process_edge(edge: Mapping[str, Any], layers_by_index: Mapping[int, Mapping[str, Any]], thresholds: Thresholds) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = load_layer_context_from_segmentation_row(layers_by_index[int(edge['left_layer_index'])])
    right = load_layer_context_from_segmentation_row(layers_by_index[int(edge['right_layer_index'])])
    nucleus_candidates = candidate_metrics(left.nucleus_labels, right.nucleus_labels, left.nucleus_geometry, right.nucleus_geometry)
    cell_candidates = candidate_metrics(left.cell_labels, right.cell_labels, left.cell_geometry, right.cell_geometry)
    return fuse_edge_minimal(edge, left, right, nucleus_candidates, cell_candidates, thresholds)

def process_manifest_shard(*, layer_manifest: str | Path, edge_manifest: str | Path, thresholds_json: str | Path, output_root: str | Path, shard_count: int=1, shard_index: int=0, prefer_parquet: bool=True) -> dict[str, Any]:
    layers = {int(row.get('layer_index', row.get('layer_idx'))): row for row in read_tsv(layer_manifest)}
    edges = [row for row in read_tsv(edge_manifest) if stable_bucket(str(row['edge_id']), shard_count) == shard_index]
    thresholds = thresholds_from_json(thresholds_json)
    out = Path(output_root)
    schemas = None
    try:
        schemas = pyarrow_schemas()
    except ModuleNotFoundError:
        schemas = {}
    counts = Counter()
    link_paths: list[str] = []
    audit_paths: list[str] = []
    for edge in edges:
        accepted, audit = process_edge(edge, layers, thresholds)
        counts['edges_processed'] += 1
        counts['accepted_links'] += len(accepted)
        counts['audit_rows'] += len(audit)
        for row in accepted:
            counts[f"quality_{row['quality_class']}"] += 1
            counts[f"cell_relation_{row.get('cell_relation')}"] += 1
        bucket = int(edge.get('matching_bucket', stable_bucket(str(edge['edge_id']), 64)))
        link_path = out / 'nucleus_links' / f'bucket_{bucket:02d}' / f"{edge['edge_id']}.parquet"
        audit_path = out / 'audit_links' / f'bucket_{bucket:02d}' / f"{edge['edge_id']}.parquet"
        write_table(link_path, accepted, schema=schemas.get('nucleus_links'), prefer_parquet=prefer_parquet)
        write_table(audit_path, audit, schema=None, prefer_parquet=prefer_parquet)
        link_paths.append(str(link_path))
        audit_paths.append(str(audit_path))
    summary = {'status': 'PASS', 'shard_count': shard_count, 'shard_index': shard_index, 'edge_count': len(edges), 'counts': dict(sorted(counts.items())), 'link_output_count': len(link_paths), 'audit_output_count': len(audit_paths)}
    atomic_write_json(out / 'runtime' / f'match_shard_{shard_index:03d}.json', summary)
    return summary

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser(description='RSG-6 edge matching worker')
    p.add_argument('--layer-manifest', required=True)
    p.add_argument('--edge-manifest', required=True)
    p.add_argument('--thresholds-json', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--tsv', action='store_true', help='force TSV fallback outputs')
    args = p.parse_args(argv)
    summary = process_manifest_shard(layer_manifest=args.layer_manifest, edge_manifest=args.edge_manifest, thresholds_json=args.thresholds_json, output_root=args.output_root, shard_count=args.shard_count, shard_index=args.shard_index, prefer_parquet=not args.tsv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
