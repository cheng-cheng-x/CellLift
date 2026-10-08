from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import socket
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
from celllift.segmentation.common import algorithm_fingerprint, assigned_worker, fingerprint, load_config, read_json, read_tsv, sha256_file, write_json, write_tsv

def _validate_roots(cfg: dict[str, Any]) -> None:
    result = Path(cfg['result_root']).resolve()
    source = Path(cfg['source_experiment_root']).resolve()
    if result == source or source in result.parents or result in source.parents:
        raise RuntimeError(f'source and result roots must be isolated: {source} / {result}')
    expected_parent = (Path(cfg['project_root']) / _public_resource('artifact_0063')).resolve()
    if expected_parent not in result.parents:
        raise RuntimeError(f'unexpected result root: {result}')

def _qc_ids(rows: list[dict[str, str]], count: int) -> list[str]:
    ordered = sorted(rows, key=lambda row: (int(row['section_id']), row['track_id'], row['roi_layer_id']))
    if count >= len(ordered):
        return [row['roi_layer_id'] for row in ordered]
    indices = {int(round(index * (len(ordered) - 1) / (count - 1))) for index in range(count)}
    return [ordered[index]['roi_layer_id'] for index in sorted(indices)]

def run(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    _validate_roots(cfg)
    result = Path(cfg['result_root'])
    result.mkdir(parents=True, exist_ok=True)
    for relative in ('00_snapshot', '01_nuclei/masks', '01_nuclei/instances', '02_cells/masks', '02_cells/instances', '03_pairs/records', '04_status/roi', '04_status/workers', '05_manifests', '05_qc/overlays', '06_audit', 'logs/workers', 'runtime/agents', 'runtime/workers', 'runtime/matplotlib'):
        (result / relative).mkdir(parents=True, exist_ok=True)
    snapshot_root = result / '00_snapshot'
    lock_path = snapshot_root / 'snapshot_lock.json'
    source_manifest = Path(cfg['source_manifest'])
    source_manifest_sha = sha256_file(source_manifest)
    config_sha = sha256_file(config_path)
    expected_lock = {'experiment_id': cfg['experiment_id'], 'source_manifest_path': str(source_manifest), 'source_manifest_sha256': source_manifest_sha, 'config_sha256': config_sha, 'algorithm_fingerprint': algorithm_fingerprint(cfg), 'expected_roi_layer_count': int(cfg['expected_roi_layer_count'])}
    existing_lock = read_json(lock_path, {})
    if existing_lock:
        for key, expected in expected_lock.items():
            if existing_lock.get(key) != expected:
                raise RuntimeError(f'snapshot lock mismatch for {key}: {existing_lock.get(key)!r} != {expected!r}')
        summary = read_json(snapshot_root / 'snapshot_summary.json', {})
        if summary.get('status') != 'PASS':
            raise RuntimeError('existing snapshot summary is not PASS')
        return summary
    rows = read_tsv(source_manifest)
    expected_count = int(cfg['expected_roi_layer_count'])
    if len(rows) != expected_count:
        raise RuntimeError(f'source manifest count {len(rows)} != {expected_count}')
    identifiers = [row['roi_layer_id'] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError('source manifest contains duplicate roi_layer_id')
    for row in rows:
        if row.get('render_status') != 'complete':
            raise RuntimeError(f"incomplete source render: {row['roi_layer_id']}")
        if int(row['width']) != int(cfg['physical']['width_px']) or int(row['height']) != int(cfg['physical']['height_px']) or row['mode'] != 'RGB' or (abs(float(row['mpp_um_per_px']) - float(cfg['physical']['mpp_um_per_px'])) > 1e-09):
            raise RuntimeError(f"source geometry mismatch: {row['roi_layer_id']}")
        if not Path(row['png_path']).is_file():
            raise FileNotFoundError(row['png_path'])
    fields = list(rows[0])
    write_tsv(snapshot_root / 'roi_layer_manifest.tsv', rows, fields)
    assignment_rows = []
    worker_counts: dict[int, int] = {}
    slot_counts: dict[int, int] = {}
    for row in rows:
        worker_id, virtual_slot = assigned_worker(cfg, row['roi_layer_id'])
        worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
        slot_counts[virtual_slot] = slot_counts.get(virtual_slot, 0) + 1
        assignment_rows.append({'roi_layer_id': row['roi_layer_id'], 'component_id': row['component_id'], 'track_id': row['track_id'], 'section_id': row['section_id'], 'worker_id': worker_id, 'virtual_slot': virtual_slot, 'source_png_path': row['png_path'], 'source_png_sha256': row['png_sha256']})
    write_tsv(snapshot_root / 'worker_assignment.tsv', assignment_rows)
    qc_ids = _qc_ids(rows, int(cfg['qc']['snapshot_sample_count']))
    qc_path = snapshot_root / 'qc_roi_layer_ids.txt'
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = qc_path.with_name(f'.{qc_path.name}.tmp.{os.getpid()}')
    temporary.write_text('\n'.join(qc_ids) + '\n', encoding='utf-8')
    os.replace(temporary, qc_path)
    write_json(snapshot_root / 'config.snapshot.json', {**cfg, 'config_source': str(config_path)})
    lock = {**expected_lock, 'created_epoch': time.time(), 'hostname': socket.gethostname()}
    write_json(lock_path, lock)
    summary = {'status': 'PASS', 'experiment_id': cfg['experiment_id'], 'roi_layer_count': len(rows), 'unique_roi_layer_count': len(set(identifiers)), 'worker_counts': {str(key): value for key, value in sorted(worker_counts.items())}, 'virtual_slot_counts': {str(key): value for key, value in sorted(slot_counts.items())}, 'qc_sample_count': len(qc_ids), 'source_manifest_sha256': source_manifest_sha, 'snapshot_manifest_sha256': sha256_file(snapshot_root / 'roi_layer_manifest.tsv'), 'assignment_manifest_sha256': sha256_file(snapshot_root / 'worker_assignment.tsv'), 'algorithm_fingerprint': algorithm_fingerprint(cfg), 'config_fingerprint': fingerprint(cfg)}
    write_json(snapshot_root / 'snapshot_summary.json', summary)
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary['status'] == 'PASS' else 2
if __name__ == '__main__':
    raise SystemExit(main())
