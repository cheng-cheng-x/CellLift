from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
from .core import read_json, read_tsv, result_root, run_logged, stage_status, valis_shell, write_json, write_tsv

def _record_render_failures(root: Path, render_failures: list[dict[str, str]]) -> None:
    manifest = root / '00_manifests/failure_manifest.tsv'
    existing = read_tsv(manifest) if manifest.exists() else []
    existing = [row for row in existing if row.get('record_type') != 'registered_layer_render_failure']
    existing.extend(({'record_type': 'registered_layer_render_failure', 'track_id': row.get('source_track_id', ''), 'optimized_track_id': row.get('optimized_track_id', ''), 'left_id': '', 'right_id': '', 'final_status': 'failed', 'failure_reasons': row.get('reason', 'render_failure')} for row in render_failures))
    write_tsv(manifest, existing)

def _pass_intervals(start: int, end: int, accepted_edges: set[int]) -> list[tuple[int, int]]:
    output = []
    interval_start = start
    for edge in range(start, end):
        if edge not in accepted_edges:
            if edge - interval_start + 1 >= 3:
                output.append((interval_start, edge))
            interval_start = edge + 1
    if end - interval_start + 1 >= 3:
        output.append((interval_start, end))
    return output

def _render_worker_config_path(cfg: dict[str, Any]) -> Path:
    configured = cfg.get('render_worker_config')
    if configured:
        return Path(configured)
    worker_config_name = 'prostate_track_roi1024_native_conditional_geometry_hier8192_2048.worker.json' if str(cfg.get('pipeline_mode', 'set_encoding')) == 'hierarchical_conditional_geometry' else 'prostate_track_roi1024_native_set_encoding.worker.json'
    return Path(cfg['code_root']) / 'configs' / worker_config_name

def run_track_optimize(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    tracks = read_tsv(root / '00_manifests/track_manifest.tsv')
    edges = read_tsv(root / '00_manifests/edge_transform_manifest.tsv')
    by_track: dict[str, set[int]] = {}
    for row in edges:
        if row['final_status'] == 'accepted':
            by_track.setdefault(row['track_id'], set()).add(int(row['left_id']))
    optimized = []
    discarded = []
    counter = 0
    for track in tracks:
        start, end = (int(track['start_section']), int(track['end_section']))
        intervals = _pass_intervals(start, end, by_track.get(track['track_id'], set()))
        kept_layers = sum((right - left + 1 for left, right in intervals))
        for left, right in intervals:
            counter += 1
            optimized.append({'optimized_track_id': f'otrack_{counter:05d}', 'source_track_id': track['track_id'], 'grid_row': track['grid_row'], 'grid_col': track['grid_col'], 'global_x': track['global_x'], 'global_y': track['global_y'], 'start_section': left, 'end_section': right, 'seed_section': (left + right) // 2, 'layer_count': right - left + 1, 'triplet_count': right - left - 1, 'keyframe_interval': cfg['tracks']['keyframe_interval'], 'status': 'pending_render'})
        if intervals != [(start, end)]:
            discarded.append({'source_track_id': track['track_id'], 'original_start': start, 'original_end': end, 'retained_interval_count': len(intervals), 'retained_layer_count': kept_layers, 'discarded_layer_count': end - start + 1 - kept_layers, 'reason': 'failed_local_edge_split'})
    manifest = root / '00_manifests/optimized_track_manifest.tsv'
    write_tsv(manifest, optimized)
    write_tsv(root / '07_track_optimization/track_splits.tsv', discarded)
    config_path = _render_worker_config_path(cfg)
    helper = Path(cfg['code_root']) / 'scripts/render_worker.py'
    command = valis_shell(cfg, helper, ['--config', str(config_path), '--track-manifest', str(manifest), '--workers', str(min(int(cfg['runtime']['cpu_qc_workers']), 16))])
    run_logged(command, root / 'logs/render_registered_layers.log')
    render_summary = read_json(root / '07_track_optimization/render_summary.json', {})
    render_failures = read_tsv(root / '07_track_optimization/render_failures.tsv')
    _record_render_failures(root, render_failures)
    status = render_summary.get('status', 'INCOMPLETE')
    write_json(root / '07_track_optimization/track_optimization_summary.json', {'status': status, 'source_track_count': len(tracks), 'optimized_track_count': len(optimized), 'triplet_capacity_after_edge_qc': sum((int(row['triplet_count']) for row in optimized)), 'tracks_with_splits': len(discarded), 'render_failure_count': len(render_failures), 'render_summary': render_summary})
    stage_status(cfg, 'track_optimize', status, optimized_tracks=len(optimized), rendered_layers=render_summary.get('rendered_layer_count', 0))
