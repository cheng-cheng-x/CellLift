from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import socket
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
from ..io_utils import atomic_json, read_json
from ..protocol import MODEL_INPUT_SHARDS, TASKS
from .config import FUSION_EXPERTS, IMAGE_ARMS, RESIDUAL_ARMS, all_fusion_jobs, all_train_jobs, arm_dir, claim_dir, fuse_dir, is_conditional_geometry, probe_dir, protocol_meta, resolved_arm_dir, result_root, runtime_dir
from ..model_input import layout

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

def try_claim(job_id: str, *, stale_s: float=12 * 3600, fail_cooldown_s: float=120.0) -> bool:
    directory = claim_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'owner.json'
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding='utf-8'))
        except Exception:
            payload = {}
        status = payload.get('status')
        if status == 'PASS':
            return False
        age = time.time() - float(payload.get('time') or 0)
        same_host = payload.get('host') == socket.gethostname()
        dead = same_host and payload.get('pid') is not None and (not _pid_alive(int(payload['pid'])))
        if status == 'running' and age < stale_s and (not dead):
            return False
        if status == 'fail' and age < fail_cooldown_s:
            return False
        marker.unlink(missing_ok=True)
    encoded = (json.dumps({**_owner(), 'job_id': job_id, 'status': 'running'}, indent=2, sort_keys=True) + '\n').encode('utf-8')
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

def finish_claim(job_id: str, status: str, detail: dict[str, Any] | None=None) -> None:
    atomic_json(claim_dir(job_id) / 'owner.json', {**_owner(), 'job_id': job_id, 'status': status, **(detail or {})})

def job_catalog() -> list[dict[str, Any]]:
    jobs = [{'id': f'dino:shard:{index:02d}', 'kind': 'dino_shard', 'shard': index} for index in range(MODEL_INPUT_SHARDS)]
    jobs.append({'id': 'dino:merge', 'kind': 'dino_merge', 'needs': [f'dino:shard:{index:02d}' for index in range(MODEL_INPUT_SHARDS)]})
    for task in TASKS:
        jobs.append({'id': f'probe:{task}', 'kind': 'probe', 'task': task})
    for task, arm in all_train_jobs():
        needs = []
        if arm in IMAGE_ARMS:
            needs.append('dino:merge')
        if arm in RESIDUAL_ARMS:
            needs.append(f'probe:{task}')
        jobs.append({'id': f'train:{task}:{arm}', 'kind': 'train', 'task': task, 'arm': arm, 'needs': needs})
    for task, expert in all_fusion_jobs():
        jobs.append({'id': f'fuse:{task}:{expert}', 'kind': 'fuse', 'task': task, 'expert': expert, 'needs': [f'train:{task}:B', f'train:{task}:{expert}']})
    jobs.append({'id': 'summarize', 'kind': 'summarize', 'needs': [f'fuse:{task}:{expert}' for task, expert in all_fusion_jobs()]})
    return jobs

def write_jobs() -> dict[str, Any]:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    jobs = job_catalog()
    payload = {**protocol_meta(), 'jobs': jobs}
    atomic_json(runtime_dir() / 'jobs.json', payload)
    return {'jobs': len(jobs), 'path': str(runtime_dir() / 'jobs.json')}

def _done(job: dict[str, Any]) -> bool:
    kind = job['kind']
    if kind == 'dino_shard':
        base = layout.root()
        marker = base / 'logs' / 'status' / f"global_dino_shard_{int(job['shard']):03d}.json"
        return marker.is_file()
    if kind == 'dino_merge':
        return (layout.root() / 'features' / 'tile_dino_global.parquet').is_file()
    if kind == 'probe':
        return (probe_dir(job['task']) / 'manifest.json').is_file()
    if kind == 'train':
        directory = resolved_arm_dir(job['task'], job['arm']) if is_conditional_geometry() and job['arm'] == 'B' else arm_dir(job['task'], job['arm'])
        return (directory / 'metrics.json').is_file()
    if kind == 'fuse':
        return (fuse_dir(job['task'], job['expert']) / 'metrics.json').is_file()
    if kind == 'summarize':
        return (result_root() / 'tables' / 'summary.json').is_file()
    return False

def _ready(job: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> bool:
    if _done(job):
        return False
    for dep in job.get('needs') or []:
        if not _done(catalog[dep]):
            return False
    return True

def catalog_map() -> dict[str, dict[str, Any]]:
    path = runtime_dir() / 'jobs.json'
    jobs = read_json(path)['jobs'] if path.is_file() else job_catalog()
    return {item['id']: item for item in jobs}

def next_job(*, skip_kinds: tuple[str, ...]=()) -> dict[str, Any] | None:
    catalog = catalog_map()
    priority = {'dino_shard': 0, 'dino_merge': 1, 'probe': 2, 'train': 3, 'fuse': 4, 'summarize': 5}
    wave1 = {'B', 'G2', 'G3', 'H2', 'H3'}
    skip = set(skip_kinds)
    candidates = [job for job in catalog.values() if job['kind'] not in skip and _ready(job, catalog)]
    candidates.sort(key=lambda job: (priority.get(job['kind'], 9), 0 if job.get('arm') in wave1 or job['kind'] in {'probe', 'dino_shard', 'dino_merge'} else 1, job['id']))
    for job in candidates:
        if try_claim(job['id']):
            return job
    return None

def execute(job: dict[str, Any], *, device: str='cuda') -> dict[str, Any]:
    import importlib
    kind = job['kind']
    if kind == 'dino_shard':
        from . import dino_global as dino_mod
        importlib.reload(dino_mod)
        return dino_mod.extract_shard(int(job['shard']), device=device)
    if kind == 'dino_merge':
        from . import dino_global as dino_mod
        importlib.reload(dino_mod)
        return dino_mod.merge_global_dino()
    if kind == 'probe':
        from . import probe as probe_mod
        importlib.reload(probe_mod)
        return probe_mod.fit_probe(job['task'], device=device)
    if kind == 'train':
        if is_conditional_geometry() and job['arm'] == 'B':
            return {'status': 'reuse_set_encoding', 'path': str(resolved_arm_dir(job['task'], 'B'))}
        from . import arms as arms_mod
        from . import bags as bags_mod
        from . import geometry as geometry_mod
        from . import train as train_mod
        importlib.reload(geometry_mod)
        importlib.reload(bags_mod)
        importlib.reload(arms_mod)
        importlib.reload(train_mod)
        return train_mod.train_arm(job['task'], job['arm'], device=device)
    if kind == 'fuse':
        from . import fuse as fuse_mod
        importlib.reload(fuse_mod)
        return fuse_mod.fuse_one(job['task'], job['expert'])
    if kind == 'summarize':
        from . import fuse as fuse_mod
        importlib.reload(fuse_mod)
        tables = fuse_mod.collect_tables()
        path = fuse_mod.render_results(None, tables)
        return {'tables': str(path), 'n_complete': len(tables.get('complete') or [])}
    raise ValueError(kind)

def worker_loop(*, device: str='cuda', idle_s: float=30.0) -> dict[str, Any]:
    write_jobs()
    finished = []
    skip = ('fuse', 'summarize') if str(device).startswith('cuda') else ()
    while True:
        job = next_job(skip_kinds=skip)
        if job is None:
            catalog = catalog_map()
            if all((_done(item) for item in catalog.values())):
                break
            time.sleep(idle_s)
            continue
        try:
            print(f"start {job['id']}", flush=True)
            payload = execute(job, device=device)
            finish_claim(job['id'], 'PASS', {'detail': {k: payload[k] for k in list(payload)[:8]}})
            finished.append(job['id'])
            print(f"done {job['id']}", flush=True)
        except Exception as exc:
            finish_claim(job['id'], 'fail', {'error': str(exc)})
            print(f"fail {job['id']}: {exc}", flush=True)
            time.sleep(max(float(idle_s), 60.0))
            continue
    return {'finished': finished, 'host': socket.gethostname()}

def progress() -> dict[str, Any]:
    catalog = catalog_map()
    counts = {'done': 0, 'pending': 0, 'by_kind': {}}
    for job in catalog.values():
        kind = job['kind']
        counts['by_kind'].setdefault(kind, {'done': 0, 'total': 0})
        counts['by_kind'][kind]['total'] += 1
        if _done(job):
            counts['done'] += 1
            counts['by_kind'][kind]['done'] += 1
        else:
            counts['pending'] += 1
    counts['total'] = len(catalog)
    return counts
