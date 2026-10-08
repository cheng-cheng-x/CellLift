from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Sequence
import argparse
from celllift.runtime import json
import os
import shutil
import socket
import subprocess
import threading
import time
ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run.py'
CONFIG_DIR = ROOT / 'configs'
PYTHON = Path(_resource_path('artifact_0008'))
GLOBAL_RESULT = Path(_resource_path('artifact_0054'))
DATASET_ORDER = ('tcga_crc_msi', 'sicapv2', 'bracs')
ARMS = ('D', 'DS', 'R', 'RS')
MEANPOOL_SEEDS = (17, 42, 73, 101, 137)
DEEPSETS_SEEDS = (42,)
FOLDS = {'tcga_crc_msi': 5, 'sicapv2': 4, 'bracs': 5}

@dataclass(frozen=True)
class Job:
    key: str
    command: tuple[str, ...]
    output: Path
    capability: str = 'any'
    expected_status: str = 'PASS'

def _config(dataset: str) -> Path:
    return CONFIG_DIR / f'{dataset}.yaml'

def _paths(dataset: str):
    from celllift.runtime import yaml
    cfg = yaml.safe_load(_config(dataset).read_text(encoding='utf-8'))
    return (Path(cfg['paths']['data_root']), Path(cfg['paths']['result_root']))

def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'PASS'
    except Exception:
        return False

def _job(dataset: str, parts: Sequence[str], output: Path, capability: str='any', expected_status: str='PASS') -> Job:
    key = '__'.join((dataset, *map(str, parts)))
    return Job(key, tuple(map(str, (PYTHON, RUN, *parts, '--config', _config(dataset)))), output, capability, expected_status)

def discover_jobs() -> list[Job]:
    jobs = []
    for dataset in DATASET_ORDER:
        data_root, result_root = _paths(dataset)
        if not _passed(data_root / '03_modal_features/manifest.json'):
            continue
        for fold in range(FOLDS[dataset]):
            residual = data_root / f'03_modal_features/residual/fold_{fold:02d}/all_residuals.json'
            shuffle = data_root / f'04_shuffle_maps/fold_{fold:02d}/mapping.json'
            if not _passed(residual):
                jobs.append(_job(dataset, ('fit-residual', '--fold', fold, '--device', 'cuda'), residual))
            if not _passed(shuffle):
                jobs.append(_job(dataset, ('build-shuffle', '--fold', fold), shuffle))
            if not (_passed(residual) and _passed(shuffle)):
                continue
            meanpool_all = data_root / f'05_meanpool_cache/fold_{fold:02d}/manifest.json'
            if not _passed(meanpool_all):
                jobs.append(_job(dataset, ('build-meanpool-all', '--fold', fold), meanpool_all))
            for arm in ARMS:
                arm_meanpool = data_root / f'05_meanpool_cache/fold_{fold:02d}/{arm}/manifest.json'
                if _passed(arm_meanpool):
                    for seed in MEANPOOL_SEEDS:
                        output = result_root / f'06_experts/meanpool/{arm}/fold_{fold:02d}/seed_{seed}/manifest.json'
                        if not _passed(output):
                            jobs.append(_job(dataset, ('train-expert', '--fold', fold, '--arm', arm, '--encoder', 'meanpool', '--seed', seed, '--device', 'cuda'), output))
                for seed in DEEPSETS_SEEDS:
                    output = result_root / f'06_experts/deepsets/{arm}/fold_{fold:02d}/seed_{seed}/manifest.json'
                    if not _passed(output):
                        jobs.append(_job(dataset, ('train-expert', '--fold', fold, '--arm', arm, '--encoder', 'deepsets', '--seed', seed, '--device', 'cuda'), output, 'a800'))
        for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
            expected = [result_root / f'06_experts/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}/manifest.json' for fold in range(FOLDS[dataset]) for arm in ARMS for seed in seeds]
            summary = result_root / f'08_metrics/{encoder}/summary.json'
            if all((_passed(path) for path in expected)) and (not _passed(summary)):
                jobs.append(_job(dataset, ('evaluate-oof', '--encoder', encoder), summary))
    rank = {'fit-residual': 0, 'build-shuffle': 1, 'build-meanpool-all': 2, 'meanpool': 3, 'deepsets': 4, 'evaluate-oof': 5}

    def priority(job: Job):
        key = job.key
        stage = next((name for name in rank if name in key), 'evaluate-oof')
        return (rank[stage], DATASET_ORDER.index(key.split('__', 1)[0]), key)
    return sorted(jobs, key=priority)

def _all_oof_complete() -> bool:
    for dataset in DATASET_ORDER:
        _, result_root = _paths(dataset)
        for encoder in ('meanpool', 'deepsets'):
            if not _passed(result_root / f'08_metrics/{encoder}/summary.json'):
                return False
    return True

def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _claim(job: Job, owner: str, stale_seconds: int=600) -> Path | None:
    root = GLOBAL_RESULT / 'runtime/job_queue/claims'
    path = root / job.key
    root.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        heartbeat = path / 'heartbeat.json'
        age = time.time() - heartbeat.stat().st_mtime if heartbeat.exists() else float('inf')
        if age <= stale_seconds:
            return None
        try:
            shutil.rmtree(path)
            path.mkdir()
        except (FileExistsError, FileNotFoundError, OSError):
            return None
    try:
        _atomic_json(path / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'job': job.key, 'time': time.time()})
        _atomic_json(path / 'heartbeat.json', {'owner': owner, 'time': time.time()})
    except FileNotFoundError:
        return None
    return path

def _ready(job: Job) -> bool:
    if not job.output.is_file():
        return False
    try:
        return json.loads(job.output.read_text(encoding='utf-8')).get('status') == job.expected_status
    except Exception:
        return False

def _attempts(job: Job) -> int:
    path = GLOBAL_RESULT / f'runtime/job_queue/attempts/{job.key}.json'
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding='utf-8')).get('attempts', 0))

def _run(job: Job, claim: Path, owner: str) -> bool:
    attempts_path = GLOBAL_RESULT / f'runtime/job_queue/attempts/{job.key}.json'
    attempts = _attempts(job) + 1
    _atomic_json(attempts_path, {'job': job.key, 'attempts': attempts, 'owner': owner, 'time': time.time()})
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            _atomic_json(claim / 'heartbeat.json', {'owner': owner, 'time': time.time()})
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    log = GLOBAL_RESULT / f'logs/job_queue/{job.key}.attempt_{attempts}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = f"{ROOT.parent}:{env.get('PYTHONPATH', '')}"
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(name, '8')
    started = time.time()
    try:
        with log.open('ab', buffering=0) as stream:
            process = subprocess.run(job.command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        success = process.returncode == 0 and _ready(job)
        record = {'status': 'PASS' if success else 'FAILED', 'job': job.key, 'owner': owner, 'attempt': attempts, 'returncode': process.returncode, 'started': started, 'finished': time.time(), 'output': str(job.output), 'log': str(log)}
        destination = GLOBAL_RESULT / f"runtime/job_queue/{('done' if success else 'failed')}/{job.key}.json"
        _atomic_json(destination, record)
        return success
    finally:
        stop.set()
        thread.join(timeout=2)
        shutil.rmtree(claim, ignore_errors=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capability', choices=('a800', 'auxiliary_ad5736'), required=True)
    parser.add_argument('--worker-id', default='')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    owner = args.worker_id or f'{socket.gethostname()}:{os.getpid()}'
    state = GLOBAL_RESULT / f"runtime/job_queue/workers/{owner.replace(':', '_')}.json"
    while True:
        if _all_oof_complete():
            marker = GLOBAL_RESULT / 'runtime/oof_complete.json'
            _atomic_json(marker, {'status': 'PASS', 'time': time.time(), 'worker': owner})
            frozen = GLOBAL_RESULT / 'protocol_frozen.json'
            if not frozen.is_file() or json.loads(frozen.read_text(encoding='utf-8')).get('status') != 'FROZEN':
                jobs = [Job('global__freeze-protocol', tuple(map(str, (PYTHON, RUN, 'freeze-protocol', '--config-dir', CONFIG_DIR, '--global-result-root', GLOBAL_RESULT))), frozen, 'any', 'FROZEN')]
            else:
                from celllift.matched_geometry_controls.official_worker import run_loop
                return run_loop(args.capability, owner, args.poll_seconds)
        else:
            jobs = discover_jobs()
        jobs = [job for job in jobs if job.capability == 'any' or args.capability == 'a800']
        progressed = False
        for job in jobs:
            if _attempts(job) >= 2 and (not _ready(job)):
                continue
            claim = _claim(job, owner)
            if claim is None:
                continue
            _atomic_json(state, {'status': 'RUNNING', 'time': time.time(), 'worker': owner, 'job': job.key})
            _run(job, claim, owner)
            progressed = True
            break
        if not progressed:
            _atomic_json(state, {'status': 'WAITING', 'time': time.time(), 'worker': owner, 'ready_jobs': len(jobs)})
            time.sleep(args.poll_seconds)
if __name__ == '__main__':
    raise SystemExit(main())
