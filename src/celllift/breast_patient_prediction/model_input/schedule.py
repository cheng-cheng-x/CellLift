from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import socket
import subprocess
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
from .. import paths
from ..io_utils import atomic_json, read_json
from ..protocol import HOST_PRIORITY
from . import layout

def _owner() -> dict[str, Any]:
    return {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time()}

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    except Exception:
        return True
    return True

def try_claim(base: Path, stage: str, slide_id: str, *, stale_s: float=6 * 3600, retry_failed: bool=False) -> bool:
    directory = layout.claim_dir(base, stage, slide_id)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'owner.json'
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding='utf-8'))
        except Exception:
            payload = {}
        status = payload.get('status')
        age = time.time() - float(payload.get('time') or 0)
        same_host = payload.get('host') == socket.gethostname()
        dead_owner = same_host and payload.get('pid') is not None and (not _pid_alive(int(payload['pid'])))
        if status == 'running' and age < stale_s and (not dead_owner):
            return False
        if status == 'fail' and (not retry_failed):
            return False
        marker.unlink(missing_ok=True)
    payload = {**_owner(), 'stage': stage, 'slide_id': slide_id, 'status': 'running'}
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + '\n').encode('utf-8')
    try:
        fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, encoded)
        os.close(fd)
        return True
    except Exception:
        os.close(fd)
        marker.unlink(missing_ok=True)
        return False

def finish_claim(base: Path, stage: str, slide_id: str, status: str, detail: dict[str, Any] | None=None) -> None:
    directory = layout.claim_dir(base, stage, slide_id)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'owner.json', {**_owner(), 'stage': stage, 'slide_id': slide_id, 'status': status, **(detail or {})})

def _npz_count(path: Path, key: str='tile_ids') -> int:
    if not path.is_file():
        return -1
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return int(len(data[key]))

def stage_done(base: Path, stage: str, job: dict[str, Any]) -> bool:
    slide_id = job['slide_id']
    n = len(job['tiles'])
    if stage == 'mask':
        return _npz_count(layout.mask_npz_path(base, slide_id)) == n and layout.instances_path(base, slide_id).is_file()
    if stage == 'graph':
        return _npz_count(layout.graph_path(base, slide_id)) == n
    if stage == 'scene':
        return layout.scene_path(base, slide_id).is_file() and _npz_count(layout.geometry_path(base, slide_id), 'tile_ids') == n and (_npz_count(layout.scene_graph_path(base, slide_id), 'tile_ids') == n)
    return False

def rgb_complete(job: dict[str, Any]) -> bool:
    base = Path(job['model_input_root'])
    status_path = layout.slide_status_path(base, job['slide_id'])
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding='utf-8'))
        except Exception:
            payload = {}
        if payload.get('status') == 'done' and int(payload.get('n_ok') or 0) == len(job['tiles']):
            return True
    for row in job['tiles']:
        path = layout.rgb_path(base, job['slide_id'], str(row['tile_id']))
        if not path.is_file() or path.stat().st_size < 1024:
            return False
    return True

def write_job_files(data_root: Path | None=None) -> dict[str, Any]:
    from . import jobs
    base = layout.ensure(data_root)
    manifest = jobs.load_tile_manifest(data_root)
    inventory = jobs.load_slide_inventory(data_root)
    rows = [jobs.enrich_tile(row, inventory, base) for row in manifest.to_dict('records')]
    grouped = jobs.slide_groups(rows)
    job_root = base / 'logs' / 'jobs'
    job_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for slide_id, tiles in grouped.items():
        payload = {'slide_id': slide_id, 'model_input_root': str(base), 'tiles': tiles, 'source_path': tiles[0]['wsi_source_path'], 'mpp_x': tiles[0]['mpp_x'], 'mpp_y': tiles[0]['mpp_y'], 'mpp_xy_mismatch': abs(float(tiles[0]['mpp_x']) - float(tiles[0]['mpp_y'])) > 1e-06, 'batch_size': 8, 'cpu_threads': 8}
        atomic_json(job_root / f'{layout._slide_stem(slide_id)}.json', payload)
        written += 1
    return {'jobs': written, 'root': str(job_root)}

def load_job_files(data_root: Path | None=None) -> list[dict[str, Any]]:
    base = layout.ensure(data_root)
    job_root = base / 'logs' / 'jobs'
    if not job_root.is_dir() or not any(job_root.glob('*.json')):
        write_job_files(data_root)
    return [read_json(path) for path in sorted(job_root.glob('*.json'))]

def run_stage(job: dict[str, Any], stage: str, *, device: str='cuda', modules=None, encoder=None, model=None) -> dict[str, Any]:
    if stage == 'rgb':
        from .rgb import export_slide
        return export_slide(job)
    if stage == 'mask':
        from .segment import segment_slide_job
        return segment_slide_job(job, model=model)
    if stage == 'graph':
        from .graphs import build_slide_graphs
        return build_slide_graphs(job)
    if stage == 'scene':
        from .encode import infer_slide
        return infer_slide(job, device=device, modules=modules, encoder=encoder, model=model)['summary']
    raise ValueError(stage)

def _ready(job: dict[str, Any], stage: str, base: Path) -> bool:
    if stage == 'rgb':
        return True
    if stage == 'mask':
        return rgb_complete(job)
    if stage == 'graph':
        return stage_done(base, 'mask', job)
    if stage == 'scene':
        return stage_done(base, 'graph', job)
    return False

def _finished(job: dict[str, Any], stage: str, base: Path) -> bool:
    if stage == 'rgb':
        return rgb_complete(job)
    return stage_done(base, stage, job)

def worker_loop(stage: str, data_root: Path | None=None, *, device: str='cuda', retry_failed: bool=False, idle_sleep_s: float=20.0, max_idle_rounds: int=9000) -> dict[str, Any]:
    jobs_list = load_job_files(data_root)
    base = layout.ensure(data_root)
    modules = encoder = model = None
    if stage == 'mask':
        from .segment import load_cellpose_model
        model = load_cellpose_model()
    if stage == 'scene':
        from .encode import load_encoder
        from celllift.matched_geometry_controls.inference import load_model
        from ..protocol import ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST
        modules, encoder = load_encoder(device)
        model, _ = load_model(modules, ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST, device)
    done = 0
    failed = 0
    idle_rounds = 0
    unfinished = 0
    print(f'worker start stage={stage} host={socket.gethostname()} pid={os.getpid()} jobs={len(jobs_list)} retry_failed={retry_failed}', flush=True)
    while idle_rounds < max_idle_rounds:
        progress = 0
        unfinished = 0
        for job in jobs_list:
            if _finished(job, stage, base):
                continue
            unfinished += 1
            if not _ready(job, stage, base):
                continue
            if not try_claim(base, stage, job['slide_id'], retry_failed=retry_failed):
                continue
            print(f"claim {stage} {job['slide_id']} n={len(job['tiles'])}", flush=True)
            try:
                run_stage(job, stage, device=device, modules=modules, encoder=encoder, model=model)
                finish_claim(base, stage, job['slide_id'], 'done')
                done += 1
                progress += 1
            except Exception as exc:
                finish_claim(base, stage, job['slide_id'], 'fail', {'error': f'{type(exc).__name__}: {exc}'})
                failed += 1
                progress += 1
        if unfinished == 0:
            break
        if progress:
            print(f'worker {stage} done={done} failed={failed} unfinished={unfinished} idle={idle_rounds}', flush=True)
        if progress == 0:
            idle_rounds += 1
            if idle_rounds == 1 or idle_rounds % 30 == 0:
                print(f'worker {stage} idle={idle_rounds} unfinished={unfinished} waiting_for_upstream', flush=True)
            if idle_rounds % 90 == 0:
                retry_failed = True
            time.sleep(idle_sleep_s)
        else:
            idle_rounds = 0
    return {'stage': stage, 'host': socket.gethostname(), 'done': done, 'failed': failed, 'remaining_unfinished_pass': unfinished if jobs_list else 0}

def progress(data_root: Path | None=None) -> dict[str, Any]:
    jobs_list = load_job_files(data_root)
    base = layout.ensure(data_root)
    counts = {'rgb': 0, 'mask': 0, 'graph': 0, 'scene': 0, 'slides': len(jobs_list)}
    failed = []
    for job in jobs_list:
        if rgb_complete(job):
            counts['rgb'] += 1
        if stage_done(base, 'mask', job):
            counts['mask'] += 1
        if stage_done(base, 'graph', job):
            counts['graph'] += 1
        if stage_done(base, 'scene', job):
            counts['scene'] += 1
        for stage in ('rgb', 'mask', 'graph', 'scene'):
            marker = layout.claim_dir(base, stage, job['slide_id']) / 'owner.json'
            if not marker.is_file():
                continue
            try:
                text = marker.read_text(encoding='utf-8').strip()
                payload = json.loads(text) if text else {}
            except Exception:
                continue
            if payload.get('status') == 'fail':
                failed.append({'slide_id': job['slide_id'], 'stage': stage, 'error': payload.get('error')})
    counts['failed'] = failed
    return counts

def wait_complete(data_root: Path | None=None, *, timeout_s: float=48 * 3600, poll_s: float=60.0) -> dict[str, Any]:
    started = time.time()
    last = {}
    while time.time() - started < timeout_s:
        last = progress(data_root)
        atomic_json(layout.ensure(data_root) / 'logs' / 'progress.json', last)
        if last['scene'] == last['slides'] and last['slides'] > 0:
            last['status'] = 'COMPLETE'
            return last
        time.sleep(poll_s)
    last['status'] = 'TIMEOUT'
    return last
