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
from celllift.geometry_experts.baseline_fold import infer_dataset
from celllift.geometry_experts.cache import cache_shard, check_cache, merge_shards
from celllift.geometry_experts.dataset import DATASETS, EXPERT_ARMS, batch_result_root, load_config
from celllift.geometry_experts.diagnose import diagnose_dataset
from celllift.geometry_experts.evaluate import ARMS, evaluate_dataset
from celllift.geometry_experts.fuse import fuse_dataset
from celllift.geometry_experts.io_utils import atomic_json, read_json
from celllift.geometry_experts.train_expert import job_matrix, train_inner

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('build-cache', 'merge-cache', 'cache-check', 'infer-b', 'train', 'fuse', 'evaluate', 'diagnose', 'worker', 'job-list', 'self-test'))
    value.add_argument('--config', type=Path)
    value.add_argument('--dataset', choices=DATASETS)
    value.add_argument('--arm', choices=EXPERT_ARMS)
    value.add_argument('--fold', type=int)
    value.add_argument('--inner', type=int)
    value.add_argument('--seed', type=int, default=42)
    value.add_argument('--shard', type=int, default=0)
    value.add_argument('--shard-id', type=int, default=0)
    value.add_argument('--num-shards', type=int, default=1)
    value.add_argument('--device', default='cuda')
    value.add_argument('--jobs', type=Path)
    value.add_argument('--max-epochs', type=int, default=60)
    value.add_argument('--patience', type=int, default=10)
    value.add_argument('--arms', default=','.join(ARMS))
    value.add_argument('--log-root', type=Path)
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

def claim_job(claim_root: Path, name: str, *, stale_seconds: int=6 * 3600) -> bool:
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
            same_host = held.get('host') == payload['host']
            expired = time.time() - float(held.get('time', 0)) > stale_seconds
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

def run_worker(args) -> dict:
    from celllift.geometry_experts.baseline import load_baseline
    from celllift.geometry_experts.data import SceneCache
    cfg = load_config(args.config)
    payload = read_json(args.jobs)
    jobs = payload['jobs'] if isinstance(payload, dict) else payload
    selected = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]
    log_root = args.log_root or batch_result_root(cfg) / 'runtime'
    log_root.mkdir(parents=True, exist_ok=True)
    failed_root = log_root / 'failed'
    failed_root.mkdir(parents=True, exist_ok=True)
    claim_root = log_root / 'claims'
    datasets = sorted({str(job['dataset']) for job in jobs})
    if len(datasets) != 1:
        raise ValueError(f'worker must serve one dataset, got {datasets}')
    cache = SceneCache(cfg, datasets[0])
    cache.preload()
    baseline = load_baseline(datasets[0], cfg)
    results = []
    for position, job in enumerate(selected):
        name = '__'.join((str(job[key]) for key in ('dataset', 'arm', 'outer', 'inner', 'seed')))
        marker = log_root / f'done__{name}.json'
        if marker.is_file():
            results.append({'job': name, 'status': 'SKIP'})
            continue
        if not claim_job(claim_root, name):
            results.append({'job': name, 'status': 'CLAIMED_ELSEWHERE'})
            continue
        try:
            manifest = train_inner(cfg, job['dataset'], job['arm'], int(job['outer']), int(job['inner']), int(job['seed']), args.device, cache=cache, baseline=baseline, max_epochs=args.max_epochs, patience=args.patience)
            atomic_json(marker, {'status': 'PASS', 'job': name, 'manifest': manifest.get('status')})
            results.append({'job': name, 'status': 'PASS', 'metric': manifest.get('best_nll')})
            (claim_root / f'{name}.claim').unlink(missing_ok=True)
        except Exception as error:
            atomic_json(failed_root / f'{name}.json', {'status': 'FAILED', 'job': name, 'error': repr(error), 'traceback': traceback.format_exc()})
            results.append({'job': name, 'status': 'FAILED', 'error': repr(error)})
            (claim_root / f'{name}.claim').unlink(missing_ok=True)
        print(json.dumps({'progress': f'{position + 1}/{len(selected)}', **results[-1]}), flush=True)
    summary = {'status': 'PASS', 'device': args.device, 'requested': len(selected), 'passed': sum((1 for row in results if row['status'] == 'PASS')), 'skipped': sum((1 for row in results if row['status'] in {'SKIP', 'CLAIMED_ELSEWHERE'})), 'failed': sum((1 for row in results if row['status'] == 'FAILED')), 'results': results}
    atomic_json(log_root / f"worker_{args.device.replace(':', '-')}_{args.shard_id}.json", summary)
    return summary

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.stage == 'self-test':
        from celllift.geometry_experts.tests import test_invariants
        failures = 0
        for name, function in sorted(vars(test_invariants).items()):
            if name.startswith('test_') and callable(function):
                try:
                    function()
                    print(f'PASS {name}')
                except Exception as error:
                    failures += 1
                    print(f'FAIL {name}: {error!r}')
        print(json.dumps({'status': 'FAIL' if failures else 'PASS', 'failures': failures}))
        return 1 if failures else 0
    cfg = load_config(args.config)
    if args.stage == 'build-cache':
        result = cache_shard(cfg, args.dataset, args.shard)
    elif args.stage == 'merge-cache':
        result = merge_shards(cfg, args.dataset)
    elif args.stage == 'cache-check':
        result = check_cache(cfg, args.dataset)
    elif args.stage == 'job-list':
        jobs = job_matrix(args.dataset, args.seed)
        destination = batch_result_root(cfg) / 'runtime' / f'jobs_{args.dataset}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, {'jobs': jobs})
        result = {'status': 'PASS', 'jobs': len(jobs), 'path': str(destination)}
    elif args.stage == 'infer-b':
        if args.dataset is None:
            raise ValueError('infer-b requires --dataset')
        result = infer_dataset(cfg, args.dataset, args.device, args.fold)
    elif args.stage == 'train':
        if None in (args.dataset, args.arm, args.fold, args.inner):
            raise ValueError('train requires --dataset --arm --fold --inner')
        result = train_inner(cfg, args.dataset, args.arm, args.fold, args.inner, args.seed, args.device, max_epochs=args.max_epochs, patience=args.patience)
    elif args.stage == 'fuse':
        result = fuse_dataset(cfg, args.dataset, args.seed, args.device if args.device != 'cuda' else 'cpu')
    elif args.stage == 'worker':
        result = run_worker(args)
    elif args.stage == 'diagnose':
        result = diagnose_dataset(cfg, args.dataset, args.seed)
    else:
        arms = [arm for arm in args.arms.split(',') if arm]
        result = evaluate_dataset(cfg, args.dataset, args.seed, arms)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
