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
from celllift.scene_residual.cache import cache_shard, check_cache, merge_shards
from celllift.scene_residual.dataset import DATASETS, load_config
from celllift.scene_residual.evaluate import ARMS, JOB_ARMS, evaluate_dataset
from celllift.scene_residual.io_utils import atomic_json, read_json
from celllift.scene_residual.permute import build_permutation
from celllift.scene_residual.train import job_matrix, train_fold

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('build-cache', 'merge-cache', 'cache-check', 'permute', 'train', 'beta-scale', 'local-readout', 'evaluate', 'explain', 'worker', 'job-list', 'status', 'self-test'))
    value.add_argument('--config', type=Path)
    value.add_argument('--dataset', choices=DATASETS)
    value.add_argument('--arm', choices=JOB_ARMS)
    value.add_argument('--fold', type=int)
    value.add_argument('--seed', type=int, default=42)
    value.add_argument('--shard', type=int, default=0)
    value.add_argument('--shard-id', type=int, default=0)
    value.add_argument('--num-shards', type=int, default=1)
    value.add_argument('--device', default='cuda')
    value.add_argument('--jobs', type=Path, help='job list json for `worker`')
    value.add_argument('--max-epochs', type=int, default=60)
    value.add_argument('--patience', type=int, default=10)
    value.add_argument('--max-steps', type=int)
    value.add_argument('--train-tiles', type=int, default=32)
    value.add_argument('--arms', default=','.join(ARMS))
    value.add_argument('--log-root', type=Path)
    return value

def _job_train(cfg, job: dict, device: str, overrides: dict, cache=None, baseline=None) -> dict:
    return train_fold(cfg, job['dataset'], job['arm'], int(job['fold']), int(job['seed']), device, cache=cache, baseline=baseline, **overrides)

def _load_worker_state(cfg, jobs: list[dict]):
    from celllift.scene_residual.data import SceneCache
    from celllift.scene_residual.train import build_baseline
    datasets = sorted({str(job['dataset']) for job in jobs})
    if len(datasets) != 1:
        raise ValueError(f'a worker must serve exactly one dataset, got {datasets}')
    dataset = datasets[0]
    cache = SceneCache(cfg, dataset)
    cache.preload()
    baseline, tile_baseline = build_baseline(cfg, dataset)
    return (dataset, cache, (baseline, tile_baseline))

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
            if same_host and _pid_alive(int(held.get('pid', 0))) and (not expired):
                return False
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(json.dumps(payload))
        return True

def run_worker(args) -> dict:
    cfg = load_config(args.config)
    payload = read_json(args.jobs)
    jobs = payload['jobs'] if isinstance(payload, dict) else payload
    selected = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]
    log_root = args.log_root or Path(cfg['paths']['result_root']) / 'runtime'
    log_root.mkdir(parents=True, exist_ok=True)
    failed_root = log_root / 'failed'
    failed_root.mkdir(parents=True, exist_ok=True)
    claim_root = log_root / 'claims'
    overrides = {'max_epochs': args.max_epochs, 'patience': args.patience, 'max_steps': args.max_steps, 'train_tiles': args.train_tiles}
    _, cache, baseline = _load_worker_state(cfg, jobs)
    results = []
    for position, job in enumerate(selected):
        name = '__'.join((str(job[key]) for key in ('dataset', 'arm', 'fold', 'seed')))
        marker = log_root / f'done__{name}.json'
        if marker.is_file():
            results.append({'job': name, 'status': 'SKIP'})
            continue
        if not claim_job(claim_root, name):
            results.append({'job': name, 'status': 'CLAIMED_ELSEWHERE'})
            continue
        try:
            if str(job['arm']) == 'G-beta':
                from celllift.scene_residual.beta_scale import scale_fold
                manifest = scale_fold(cfg, job['dataset'], job['arm'], int(job['fold']), int(job['seed']), args.device, cache=cache, baseline=baseline)
            elif str(job['arm']) in {'G-Local', 'XY-Local'}:
                from celllift.scene_residual.local_readout import train_fold as train_local
                manifest = train_local(cfg, job['dataset'], job['arm'], int(job['fold']), int(job['seed']), args.device, cache=cache, baseline=baseline, max_epochs=overrides['max_epochs'], patience=overrides['patience'], max_steps=overrides['max_steps'])
            else:
                manifest = _job_train(cfg, job, args.device, overrides, cache=cache, baseline=baseline)
            atomic_json(marker, {'status': 'PASS', 'job': name, 'manifest': manifest.get('status')})
            results.append({'job': name, 'status': 'PASS', 'metric': manifest.get('best_metric'), 'selected_beta': manifest.get('selected_beta')})
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
        from celllift.scene_residual.tests import test_invariants
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
    if args.stage == 'worker':
        if args.config is None or args.jobs is None:
            raise ValueError('worker requires --config and --jobs')
        print(json.dumps(run_worker(args), indent=2, ensure_ascii=False, default=str))
        return 0
    if args.config is None and args.stage != 'job-list':
        raise ValueError('stage requires --config')
    if args.stage == 'job-list':
        if args.dataset is None:
            raise ValueError('job-list requires --dataset')
        cfg = load_config(args.config) if args.config else None
        arms = [arm for arm in args.arms.split(',') if arm]
        jobs = [{'dataset': args.dataset, 'arm': arm, 'fold': fold, 'seed': args.seed} for arm in arms for fold in range(_folds(args.dataset))]
        print(json.dumps({'jobs': jobs}, indent=2))
        return 0
    cfg = load_config(args.config)
    if args.stage == 'build-cache':
        if args.dataset is None:
            raise ValueError('build-cache requires --dataset')
        result = cache_shard(cfg, args.dataset, args.shard)
    elif args.stage == 'merge-cache':
        if args.dataset is None:
            raise ValueError('merge-cache requires --dataset')
        result = merge_shards(cfg, args.dataset)
    elif args.stage == 'cache-check':
        if args.dataset is None:
            raise ValueError('cache-check requires --dataset')
        result = check_cache(cfg, args.dataset, args.shard)
    elif args.stage == 'status':
        result = status_report(cfg)
    elif args.stage == 'permute':
        if args.dataset is None:
            raise ValueError('permute requires --dataset')
        result = build_permutation(cfg, args.dataset)
    elif args.stage == 'train':
        if None in (args.dataset, args.arm, args.fold):
            raise ValueError('train requires --dataset/--arm/--fold')
        result = _job_train(cfg, {'dataset': args.dataset, 'arm': args.arm, 'fold': args.fold, 'seed': args.seed}, args.device, {'max_epochs': args.max_epochs, 'patience': args.patience, 'max_steps': args.max_steps, 'train_tiles': args.train_tiles})
    elif args.stage == 'beta-scale':
        if None in (args.dataset, args.fold):
            raise ValueError('beta-scale requires --dataset/--fold')
        from celllift.scene_residual.beta_scale import ARM as BETA_ARM, scale_fold
        result = scale_fold(cfg, args.dataset, args.arm or BETA_ARM, args.fold, args.seed, args.device)
    elif args.stage == 'local-readout':
        if None in (args.dataset, args.fold):
            raise ValueError('local-readout requires --dataset/--fold')
        from celllift.scene_residual.local_readout import train_fold as train_local
        result = train_local(cfg, args.dataset, args.arm or 'G-Local', args.fold, args.seed, args.device, max_epochs=args.max_epochs, patience=args.patience, max_steps=args.max_steps)
    elif args.stage == 'explain':
        if args.dataset is None:
            raise ValueError('explain requires --dataset')
        from celllift.scene_residual.local_readout import explain_dataset
        result = explain_dataset(cfg, args.dataset, args.seed, args.arm or 'G-Local')
    else:
        if args.dataset is None:
            raise ValueError('evaluate requires --dataset')
        arms = [arm for arm in args.arms.split(',') if arm]
        result = evaluate_dataset(cfg, args.dataset, args.seed, arms)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0

def _folds(dataset: str) -> int:
    from celllift.scene_residual.dataset import DATASET_SPEC
    return int(DATASET_SPEC[dataset]['folds'])

def status_report(cfg) -> dict:
    import statistics
    from celllift.scene_residual.evaluate import fold_prediction_path
    root = Path(cfg['paths']['result_root'])
    runtime = root / 'runtime'
    datasets = (str(cfg['dataset']),) if cfg.get('dataset') else ('sicapv2', 'bracs', 'tcga_crc_msi')
    report: dict = {'datasets': {}, 'failed': 0, 'in_flight': []}
    for dataset in datasets:
        jobs_path = runtime / f'jobs_{dataset}.json'
        if not jobs_path.is_file():
            continue
        jobs = read_json(jobs_path)['jobs']
        arms = sorted({job['arm'] for job in jobs})
        done, epochs, metrics = ({}, [], [])
        for job in jobs:
            job_root = root / dataset / 'predictions' / job['arm'] / f"seed_{job['seed']:02d}" / f"fold_{job['fold']:02d}"
            if (job_root / 'job.json').is_file():
                manifest = read_json(job_root / 'job.json')
                done[job['arm']] = done.get(job['arm'], 0) + 1
                epochs.append(int(manifest.get('epochs', 0)))
                metrics.append(float(manifest.get('best_metric', float('nan'))))
            elif (job_root / 'best.pt').is_file():
                report['in_flight'].append(f"{dataset}__{job['arm']}__fold{job['fold']}")
        report['datasets'][dataset] = {'planned': len(jobs), 'done': sum(done.values()), 'per_arm': {arm: f"{done.get(arm, 0)}/{sum((1 for j in jobs if j['arm'] == arm))}" for arm in arms}, 'epochs_median': statistics.median(epochs) if epochs else None, 'selection_median': round(statistics.median(metrics), 4) if metrics else None}
    failed_root = runtime / 'failed'
    if failed_root.is_dir():
        report['failed'] = len(list(failed_root.glob('*.json')))
    return report
if __name__ == '__main__':
    raise SystemExit(main())
