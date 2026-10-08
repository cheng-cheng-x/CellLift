from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import sys
import time
import traceback
from celllift.runtime import ResourcePath as Path
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from celllift.benchmark_prediction.fuse import fuse_from_job, pred_path
from celllift.benchmark_prediction.io import job_dir, reused
from celllift.benchmark_prediction.jobs import job_matrix, baseline_job_matrix
from celllift.benchmark_prediction.paths import PYTHON, runtime_root
from celllift.benchmark_prediction.splits import build_all
from celllift.benchmark_prediction.train_baseline import train_baseline
from celllift.benchmark_prediction.train_feature import train_feature
from celllift.benchmark_prediction.train_geometry import train_geometry, train_residual
from celllift.benchmark_prediction.train_route import train_expert, train_pretrain, train_route

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('build-splits', 'job-list', 'worker', 'cpu-worker', 'evaluate', 'status', 'baseline-worker', 'baseline-status'))
    value.add_argument('--device', default='cuda')
    value.add_argument('--job-id')
    value.add_argument('--dataset', action='append', default=None)
    value.add_argument('--once', action='store_true')
    return value

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def claim_job(name: str, stale_seconds: int=8 * 3600) -> bool:
    claim_root = runtime_root() / 'claims'
    claim_root.mkdir(parents=True, exist_ok=True)
    path = claim_root / f'{name}.claim'
    payload = {'job': name, 'host': os.uname().nodename, 'pid': os.getpid(), 'time': time.time()}
    while True:
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                path.unlink(missing_ok=True)
                continue
            expired = time.time() - float(held.get('time', 0)) > stale_seconds
            same_host = held.get('host') == payload['host']
            if not expired:
                if same_host and (not _pid_alive(int(held.get('pid', 0)))):
                    path.unlink(missing_ok=True)
                    continue
                return False
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(payload))
        return True

def release_claim(name: str) -> None:
    (runtime_root() / 'claims' / f'{name}.claim').unlink(missing_ok=True)

def _pass(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') in {'PASS', 'REUSED'}
    except (OSError, ValueError):
        return False

def mean_selected_epochs(job: dict) -> int | None:
    arm = job.get('arm_tag') or job['arm']
    family = 'pretrain' if job['family'] == 'pretrain' else job['family']
    epochs = []
    for fold in range(4):
        path = job_dir({**job, 'family': family, 'arm': arm, 'arm_tag': job.get('arm_tag'), 'unit': f'fold_{fold:02d}', 'fold': fold, 'retrain': False}) / 'job.json'
        if not _pass(path):
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        epochs.append(int(payload.get('selected_epoch', payload.get('epochs', 1))) + 1)
    return max(1, int(round(sum(epochs) / len(epochs))))

def job_ready(job: dict) -> tuple[bool, str]:
    destination = job_dir(job)
    if reused(destination):
        return (False, 'done')
    arm = job.get('arm_tag') or job['arm']
    if job['family'] in {'geometry', 'expert'} and arm in {'GR', 'G2S', 'ER'}:
        residual = job_dir({**job, 'family': 'residual', 'arm': 'probe', 'arm_tag': None}) / 'job.json'
        if not _pass(residual):
            return (False, 'wait_residual')
    if job['family'] == 'route' and job.get('route') in {'a', 'd'}:
        pre_arm = job['arm'] if job['dataset'] != 'bracs' else job['arm']
        pre = job_dir({**job, 'family': 'pretrain', 'arm': pre_arm, 'arm_tag': None}) / 'job.json'
        if not _pass(pre):
            return (False, 'wait_pretrain')
    if job.get('retrain') and job['dataset'] == 'sicapv2' and (job['family'] not in {'residual', 'pretrain'}):
        locked = mean_selected_epochs({**job, 'family': job['family']})
        if locked is None:
            return (False, 'wait_folds')
        job['fixed_epochs'] = locked
    if job.get('retrain') and job['family'] == 'pretrain':
        locked = mean_selected_epochs(job)
        if locked is None:
            return (False, 'wait_folds')
    if job['family'] == 'fusion':
        b = pred_path(job['dataset'], job['partner_family'], job['partner_arm'], job['unit'], job['seed'], int(job.get('fold') or 0))
        e = pred_path(job['dataset'], job['expert_family'], job['expert_arm'], job['unit'], job['seed'], int(job.get('fold') or 0))
        if not b.is_file() or not e.is_file():
            return (False, 'wait_predictions')
    return (True, 'ok')

def dispatch(job: dict, device: str) -> dict:
    family = job['family']
    if family == 'baseline':
        return train_baseline(job, device)
    if family == 'geometry':
        return train_geometry(job, device)
    if family == 'residual':
        return train_residual(job, device)
    if family == 'expert':
        return train_expert(job, device)
    if family == 'route':
        return train_route(job, device)
    if family == 'pretrain':
        return train_pretrain(job, device)
    if family == 'feature':
        return train_feature(job, device)
    if family == 'fusion':
        return fuse_from_job(job, device)
    raise ValueError(family)

def run_one(job: dict, device: str) -> dict:
    name = job['job_id']
    if reused(job_dir(job)):
        return {'job': name, 'status': 'SKIP'}
    ready, why = job_ready(job)
    if not ready:
        return {'job': name, 'status': 'NOT_READY', 'reason': why}
    if not claim_job(name):
        return {'job': name, 'status': 'CLAIMED_ELSEWHERE'}
    log_root = runtime_root() / 'failed'
    (log_root / f'{name}.json').unlink(missing_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        manifest = dispatch(job, device)
        status = manifest.get('status', 'PASS')
        return {'job': name, 'status': status, 'manifest_status': status}
    except Exception as error:
        payload = {'status': 'FAILED', 'job': name, 'error': repr(error), 'traceback': traceback.format_exc()}
        (log_root / f'{name}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return {'job': name, 'status': 'FAILED', 'error': repr(error)}
    finally:
        release_claim(name)

def worker_loop(device: str, cpu: bool, once: bool, job_id: str | None=None) -> int:
    print(json.dumps({'event': 'worker_start', 'host': os.uname().nodename, 'pid': os.getpid(), 'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cpu': cpu}), flush=True)
    jobs = job_matrix()
    if job_id:
        selected = [job for job in jobs if job['job_id'] == job_id]
        if not selected:
            print(json.dumps({'status': 'FAILED', 'error': f'unknown job {job_id}'}))
            return 1
        print(json.dumps(run_one(selected[0], device)), flush=True)
        return 0
    while True:
        launched = False
        for job in jobs:
            if cpu and job['family'] != 'fusion':
                continue
            if not cpu and job['family'] == 'fusion':
                continue
            ready, _why = job_ready(job)
            if not ready:
                continue
            if reused(job_dir(job)):
                continue
            result = run_one(job, device)
            print(json.dumps(result), flush=True)
            if result.get('status') in {'CLAIMED_ELSEWHERE', 'NOT_READY', 'SKIP', 'BLOCKED'}:
                continue
            launched = True
            break
        if once:
            if not launched:
                print(json.dumps({'status': 'IDLE', 'cpu': cpu, 'device': device}), flush=True)
            return 0
        if not launched:
            print(json.dumps({'status': 'IDLE', 'cpu': cpu, 'device': device}), flush=True)
            return 0
        jobs = job_matrix()

def status_summary() -> dict:
    jobs = job_matrix()
    counts = {'PASS': 0, 'PENDING': 0, 'FAILED': 0, 'WAIT': 0}
    rows = []
    failed_root = runtime_root() / 'failed'
    for job in jobs:
        dest = job_dir(job)
        if reused(dest):
            counts['PASS'] += 1
            state = 'PASS'
        elif (failed_root / f"{job['job_id']}.json").is_file():
            counts['FAILED'] += 1
            state = 'FAILED'
        else:
            ready, why = job_ready(job)
            state = 'PENDING' if ready else f'WAIT:{why}'
            counts['PENDING' if ready else 'WAIT'] += 1
        rows.append({'job_id': job['job_id'], 'state': state, 'priority': job.get('priority')})
    return {'counts': counts, 'n': len(jobs), 'rows': rows}

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    runtime_root().mkdir(parents=True, exist_ok=True)
    if args.stage != 'build-splits':
        from celllift.benchmark_prediction.paths import split_root
        if not (split_root() / 'summary.json').is_file():
            build_all()
    if args.stage == 'build-splits':
        print(json.dumps({key: {k: v for k, v in value.items() if k != 'units'} for key, value in build_all().items()}, indent=2))
        return 0
    if args.stage == 'job-list':
        jobs = job_matrix()
        path = runtime_root() / 'jobs.json'
        path.write_text(json.dumps([{k: v for k, v in job.items() if k not in {'official_fit', 'official_val', 'official_predict', 'official_test'}} for job in jobs], indent=2), encoding='utf-8')
        print(json.dumps({'n': len(jobs), 'path': str(path)}))
        return 0
    if args.stage == 'status':
        print(json.dumps(status_summary()['counts']))
        return 0
    if args.stage == 'baseline-status':
        jobs = baseline_job_matrix()
        print(json.dumps({'n': len(jobs), 'ids': [job['job_id'] for job in jobs]}))
        return 0
    if args.stage == 'evaluate':
        from celllift.benchmark_prediction.evaluate import evaluate_all
        print(json.dumps(evaluate_all(args.device, datasets=args.dataset), indent=2, default=str))
        return 0
    if args.stage == 'baseline-worker':
        jobs = baseline_job_matrix()
        if args.job_id:
            jobs = [job for job in jobs if job['job_id'] == args.job_id]
        for job in jobs:
            print(json.dumps(run_one(job, args.device)), flush=True)
        return 0
    return worker_loop(args.device, cpu=args.stage == 'cpu-worker', once=args.once, job_id=args.job_id)
if __name__ == '__main__':
    raise SystemExit(main())
