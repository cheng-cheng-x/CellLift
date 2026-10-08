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
from celllift.morphology_interaction.dataset import DATASETS, ROUTES, batch_result_root, load_config
from celllift.morphology_interaction.io_utils import atomic_json, read_json

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('build-cache', 'merge-cache', 'cache-check', 'pretrain', 'validate-c', 'prefetch-c', 'train', 'evaluate', 'diagnose', 'worker', 'job-list', 'self-test'))
    value.add_argument('--config', type=Path)
    value.add_argument('--dataset', choices=DATASETS)
    value.add_argument('--route', choices=ROUTES, default='a')
    value.add_argument('--arm')
    value.add_argument('--fold', type=int)
    value.add_argument('--seed', type=int, default=42)
    value.add_argument('--shard', type=int, default=0)
    value.add_argument('--shard-id', type=int, default=0)
    value.add_argument('--num-shards', type=int, default=1)
    value.add_argument('--device', default='cuda')
    value.add_argument('--jobs', type=Path)
    value.add_argument('--max-epochs', type=int, default=60)
    value.add_argument('--patience', type=int, default=10)
    value.add_argument('--target', choices=('2d', '3d'), default='3d')
    value.add_argument('--log-root', type=Path)
    value.add_argument('--workers', type=int, default=4)
    value.add_argument('--fit-fold', type=int, dest='fit_fold')
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

def _pretrain_checkpoint(cfg, route: str, arm: str, outer: int):
    from celllift.morphology_interaction.dataset import ARM_NODE_3D
    target = '3d' if ARM_NODE_3D.get(str(arm)) else '2d'
    if route == 'a':
        from celllift.morphology_interaction.route_a.pretrain import pretrain_dir
        root = pretrain_dir(cfg, target)
    elif route == 'd':
        from celllift.morphology_interaction.route_d.pretrain import pretrain_dir
        root = pretrain_dir(cfg, target)
    else:
        return (None, None)
    for path in (root / f'fold_{int(outer):02d}' / 'best.pt', root / 'best.pt'):
        if path.is_file():
            return (path, target)
    return (None, target)

def _train_one(cfg, job, device, max_epochs, patience, cache=None):
    route = job['route']
    extra = {}
    if route == 'a':
        from celllift.morphology_interaction.route_a.train import train_outer
    elif route == 'b':
        from celllift.morphology_interaction.route_b.train import train_outer
    elif route == 'c':
        from celllift.morphology_interaction.route_c.train import train_outer
    else:
        from celllift.morphology_interaction.route_d.train import train_outer
    if route in {'a', 'd'}:
        path, target = _pretrain_checkpoint(cfg, route, job['arm'], int(job['outer']))
        if path is not None:
            extra['init_path'] = str(path)
            extra['pretrain_pool'] = f'fit_isolated_{target}'
    if job.get('official_fit'):
        extra['official_fit'] = job['official_fit']
        extra['official_val'] = job.get('official_val') or []
        extra['official_predict'] = job.get('official_predict') or job.get('official_val') or []
    if job.get('fixed_epochs') is not None:
        extra['fixed_epochs'] = int(job['fixed_epochs'])
    return train_outer(cfg, job['dataset'], job['arm'], int(job['outer']), int(job['seed']), device, max_epochs=max_epochs, patience=patience, cache=cache, **extra)

def _run_queue(args, jobs, run_one, prefix: str) -> dict:
    cfg = load_config(args.config)
    selected = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]
    log_root = args.log_root or batch_result_root(cfg, args.route) / 'runtime'
    log_root.mkdir(parents=True, exist_ok=True)
    failed_root = log_root / 'failed'
    failed_root.mkdir(parents=True, exist_ok=True)
    claim_root = log_root / 'claims'
    results = []
    for position, job in enumerate(selected):
        name = prefix + '__' + '__'.join((str(job[key]) for key in job if key != 'stage'))
        marker = log_root / f'done__{name}.json'
        if marker.is_file():
            results.append({'job': name, 'status': 'SKIP'})
            continue
        if not claim_job(claim_root, name):
            results.append({'job': name, 'status': 'CLAIMED_ELSEWHERE'})
            continue
        try:
            manifest = run_one(cfg, job)
            atomic_json(marker, {'status': 'PASS', 'job': name, 'manifest': manifest.get('status')})
            results.append({'job': name, 'status': 'PASS'})
            (claim_root / f'{name}.claim').unlink(missing_ok=True)
        except Exception as error:
            atomic_json(failed_root / f'{name}.json', {'status': 'FAILED', 'job': name, 'error': repr(error), 'traceback': traceback.format_exc()})
            results.append({'job': name, 'status': 'FAILED', 'error': repr(error)})
            (claim_root / f'{name}.claim').unlink(missing_ok=True)
        print(json.dumps({'progress': f'{position + 1}/{len(selected)}', **results[-1]}), flush=True)
    summary = {'status': 'PASS', 'device': args.device, 'requested': len(selected), 'passed': sum((1 for row in results if row['status'] == 'PASS')), 'skipped': sum((1 for row in results if row['status'] in {'SKIP', 'CLAIMED_ELSEWHERE'})), 'failed': sum((1 for row in results if row['status'] == 'FAILED')), 'results': results}
    atomic_json(log_root / f"worker_{prefix}_{args.device.replace(':', '-')}_{args.shard_id}.json", summary)
    return summary

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.stage == 'self-test':
        from celllift.morphology_interaction.tests import test_invariants
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
    from celllift.morphology_interaction.cache import cache_shard, check_cache, merge_shards
    from celllift.morphology_interaction.diagnose import diagnose_dataset
    from celllift.morphology_interaction.evaluate import evaluate_dataset
    cfg = load_config(args.config)
    if args.stage == 'build-cache':
        result = cache_shard(cfg, args.dataset, args.shard, args.device)
    elif args.stage == 'merge-cache':
        result = merge_shards(cfg, args.dataset)
    elif args.stage == 'cache-check':
        result = check_cache(cfg, args.dataset)
    elif args.stage == 'pretrain':
        if args.route == 'a':
            from celllift.morphology_interaction.route_a.pretrain import run_pretrain
        else:
            from celllift.morphology_interaction.route_d.pretrain import run_pretrain
        result = run_pretrain(cfg, args.target, args.device, dataset=args.dataset or 'sicapv2', fold=args.fit_fold)
    elif args.stage == 'validate-c':
        from celllift.morphology_interaction.route_c.validate import run_validate
        result = run_validate(cfg, args.dataset or 'sicapv2')
    elif args.stage == 'prefetch-c':
        from celllift.morphology_interaction.dataset import route_arms
        from celllift.morphology_interaction.route_c.fields import prefetch_missing_fields
        dataset = args.dataset or 'sicapv2'
        arms = (args.arm,) if args.arm else route_arms('c')
        result = {'status': 'PASS', 'dataset': dataset, 'arms': []}
        for arm in arms:
            result['arms'].append(prefetch_missing_fields(cfg, dataset, arm, workers=args.workers))
    elif args.stage == 'job-list':
        from celllift.morphology_interaction.route_a.train import job_matrix as jobs_a
        from celllift.morphology_interaction.route_b.train import job_matrix as jobs_b
        from celllift.morphology_interaction.route_c.train import job_matrix as jobs_c
        from celllift.morphology_interaction.route_d.train import job_matrix as jobs_d
        makers = {'a': jobs_a, 'b': jobs_b, 'c': jobs_c, 'd': jobs_d}
        jobs = makers[args.route](args.dataset, args.seed)
        runtime = batch_result_root(cfg, args.route) / 'runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        atomic_json(runtime / f'jobs_{args.dataset}.json', {'jobs': jobs})
        result = {'status': 'PASS', 'jobs': len(jobs), 'route': args.route}
    elif args.stage == 'train':
        if None in (args.dataset, args.arm, args.fold):
            raise ValueError('train requires --dataset --arm --fold')
        result = _train_one(cfg, {'dataset': args.dataset, 'route': args.route, 'arm': args.arm, 'outer': args.fold, 'seed': args.seed}, args.device, args.max_epochs, args.patience)
    elif args.stage == 'worker':
        payload = read_json(args.jobs)
        jobs = payload['jobs'] if isinstance(payload, dict) else payload
        cache = None
        if jobs:
            from celllift.morphology_interaction.data import SceneCache
            cache = SceneCache(cfg, jobs[0]['dataset'])
            cache.preload(load_spatial_maps=args.route in {'a', 'c', 'd'})

        def run_one(_cfg, job):
            return _train_one(cfg, job, args.device, args.max_epochs, args.patience, cache=cache)
        result = _run_queue(args, jobs, run_one, f'route{args.route}')
    elif args.stage == 'diagnose':
        result = diagnose_dataset(cfg, args.dataset, args.route, args.seed)
    else:
        result = evaluate_dataset(cfg, args.dataset, args.seed, args.route)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
