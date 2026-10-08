from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from pathlib import Path
from .artifact_filter import mark_artifact_cells
from .common import atomic_write_gzip_json, atomic_write_json, atomic_write_tsv, read_gzip_json, read_json
from .calibration import calibrate_thresholds_from_manifests
from .fusion import Thresholds
from .dataset_ops import build_artifact_filtered_layer_manifest, compact_match_outputs, validate_final_dataset, write_dataset_readme
from .registration import build_edge_job_manifest, build_registration_qc_tables, build_registration_visual_qc_contact_sheets, build_segmentation_source_manifest, freeze_snapshot, validate_rsg_manifests

def cmd_snapshot(args: argparse.Namespace) -> int:
    result = freeze_snapshot(args.rsg_root, args.result_root, code_version=args.code_version)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_rsg_manifests(args.rsg_root, check_png_pixels=args.check_png_pixels, max_rows=args.max_rows)
    out = Path(args.result_root) / '01_registration_qc' / 'manifest_integrity.json'
    excluded = result.pop('_excluded_registration_edges_full', []) or []
    if excluded:
        atomic_write_tsv(Path(args.result_root) / '00_snapshot' / 'excluded_registration_edges.tsv', excluded)
        atomic_write_tsv(Path(args.result_root) / '01_registration_qc' / 'excluded_registration_edges.tsv', excluded)
    atomic_write_json(out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'].startswith('PASS') else 2

def cmd_registration_qc(args: argparse.Namespace) -> int:
    result = build_registration_qc_tables(args.rsg_root, args.result_root, scan_limit=args.scan_limit, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not str(result.get('status', '')).startswith('FAIL') else 2

def cmd_registration_contact_sheets(args: argparse.Namespace) -> int:
    result = build_registration_visual_qc_contact_sheets(args.rsg_root, args.result_root, top_layer_count=args.top_layer_count, edge_count=args.edge_count)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0

def cmd_seg_manifest(args: argparse.Namespace) -> int:
    rows = build_segmentation_source_manifest(args.rsg_root, args.result_root, bucket_count=args.buckets, compute_sha=not args.no_sha)
    print(json.dumps({'status': 'PASS', 'rows': len(rows)}, indent=2, sort_keys=True))
    return 0

def cmd_edge_manifest(args: argparse.Namespace) -> int:
    rows = build_edge_job_manifest(args.rsg_root, args.result_root, bucket_count=args.buckets)
    print(json.dumps({'status': 'PASS', 'rows': len(rows)}, indent=2, sort_keys=True))
    return 0

def cmd_filter_pairs(args: argparse.Namespace) -> int:
    payload = read_gzip_json(args.input_pairs)
    marked = mark_artifact_cells(payload, area_threshold_px=args.area_threshold_px)
    atomic_write_gzip_json(args.output_pairs, marked)
    print(json.dumps(marked.get('summary', {}), indent=2, sort_keys=True))
    return 0

def cmd_build_layer_dataset(args: argparse.Namespace) -> int:
    result = build_artifact_filtered_layer_manifest(result_root=args.result_root, source_manifest=args.source_manifest, cellpose_manifest=args.cellpose_manifest, prefer_parquet=not args.tsv)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 2

def cmd_compact_dataset(args: argparse.Namespace) -> int:
    result = compact_match_outputs(result_root=args.result_root, prefer_parquet=not args.tsv)
    readme = write_dataset_readme(result_root=args.result_root)
    result['readme'] = readme
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] in {'PASS', 'PARTIAL'} else 2

def cmd_validate_final_dataset(args: argparse.Namespace) -> int:
    result = validate_final_dataset(result_root=args.result_root, expected_layers=args.expected_layers, expected_edges=args.expected_edges)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 2

def cmd_calibrate_thresholds(args: argparse.Namespace) -> int:
    result = calibrate_thresholds_from_manifests(layer_manifest=args.layer_manifest, edge_manifest=args.edge_manifest, output_json=args.output, max_edges=args.max_edges, null_shift_px=args.null_shift_px, max_empirical_fdr=args.max_empirical_fdr, min_gold_retention=args.min_gold_retention, progress_every=args.progress_every)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] in {'PASS', 'NO_PASSING_THRESHOLD_DISABLE_LAP_SILVER'} else 2

def cmd_write_default_thresholds(args: argparse.Namespace) -> int:
    thresholds = {'status': 'PILOT_ONLY_REQUIRES_CALIBRATION_FOR_PRODUCTION', 'nucleus_lap_iou': args.nucleus_lap_iou, 'nucleus_lap_margin': args.nucleus_lap_margin, 'local_residual_limit_px': 12.0, 'abs_log_area_ratio_limit': 1.2}
    atomic_write_json(args.output, thresholds)
    print(json.dumps(thresholds, indent=2, sort_keys=True))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='RSG-6 prostate cross-layer dataset builder')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('snapshot')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--code-version', default='serial_cross_layer_dataset_set_encoding')
    s.set_defaults(func=cmd_snapshot)
    s = sub.add_parser('validate-rsg')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--check-png-pixels', action='store_true')
    s.add_argument('--max-rows', type=int)
    s.set_defaults(func=cmd_validate)
    s = sub.add_parser('registration-qc')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--scan-limit', type=int)
    s.add_argument('--workers', type=int, default=8)
    s.set_defaults(func=cmd_registration_qc)
    s = sub.add_parser('registration-contact-sheets')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--top-layer-count', type=int, default=32)
    s.add_argument('--edge-count', type=int, default=48)
    s.set_defaults(func=cmd_registration_contact_sheets)
    s = sub.add_parser('build-segmentation-manifest')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--buckets', type=int, default=64)
    s.add_argument('--no-sha', action='store_true')
    s.set_defaults(func=cmd_seg_manifest)
    s = sub.add_parser('build-edge-manifest')
    s.add_argument('--rsg-root', required=True)
    s.add_argument('--result-root', required=True)
    s.add_argument('--buckets', type=int, default=64)
    s.set_defaults(func=cmd_edge_manifest)
    s = sub.add_parser('filter-pairs')
    s.add_argument('--input-pairs', required=True)
    s.add_argument('--output-pairs', required=True)
    s.add_argument('--area-threshold-px', type=int, default=816)
    s.set_defaults(func=cmd_filter_pairs)
    s = sub.add_parser('build-layer-dataset')
    s.add_argument('--result-root', required=True)
    s.add_argument('--source-manifest')
    s.add_argument('--cellpose-manifest')
    s.add_argument('--tsv', action='store_true')
    s.set_defaults(func=cmd_build_layer_dataset)
    s = sub.add_parser('compact-dataset')
    s.add_argument('--result-root', required=True)
    s.add_argument('--tsv', action='store_true')
    s.set_defaults(func=cmd_compact_dataset)
    s = sub.add_parser('validate-final-dataset')
    s.add_argument('--result-root', required=True)
    s.add_argument('--expected-layers', type=int, default=11016)
    s.add_argument('--expected-edges', type=int, default=8972)
    s.set_defaults(func=cmd_validate_final_dataset)
    s = sub.add_parser('calibrate-thresholds')
    s.add_argument('--layer-manifest', required=True)
    s.add_argument('--edge-manifest', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--max-edges', type=int)
    s.add_argument('--null-shift-px', type=int, default=8)
    s.add_argument('--max-empirical-fdr', type=float, default=0.05)
    s.add_argument('--min-gold-retention', type=float, default=0.95)
    s.add_argument('--progress-every', type=int, default=100)
    s.set_defaults(func=cmd_calibrate_thresholds)
    s = sub.add_parser('write-pilot-thresholds')
    s.add_argument('--output', required=True)
    s.add_argument('--nucleus-lap-iou', type=float, default=0.1)
    s.add_argument('--nucleus-lap-margin', type=float, default=0.02)
    s.set_defaults(func=cmd_write_default_thresholds)
    return p

def main(argv: list[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
if __name__ == '__main__':
    raise SystemExit(main())
