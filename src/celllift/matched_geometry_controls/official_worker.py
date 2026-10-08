from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from celllift.runtime import json
import time
from .worker import ARMS, CONFIG_DIR, DATASET_ORDER, DEEPSETS_SEEDS, GLOBAL_RESULT, Job, MEANPOOL_SEEDS, PYTHON, RUN, _attempts, _atomic_json, _claim, _paths, _ready, _run
SHARDS = {'tcga_crc_msi': 4, 'sicapv2': 2, 'bracs': 2}

def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'PASS'
    except Exception:
        return False

def _command(dataset: str, stage: str, *parts) -> tuple[str, ...]:
    return tuple(map(str, (PYTHON, RUN, stage, '--config', CONFIG_DIR / f'{dataset}.yaml', *parts)))

def _job(dataset: str, name: str, output: Path, stage: str, *parts, capability: str='any'):
    return Job(f'official__{dataset}__{name}', _command(dataset, stage, *parts), output, capability)

def discover_official_jobs() -> list[Job]:
    jobs = []
    for dataset in DATASET_ORDER:
        data_root, result_root = _paths(dataset)
        shards = SHARDS[dataset]
        input_root = data_root / '09_official_test/01_projection_scene_inputs'
        scene_root = data_root / '09_official_test/02_projection_scene_selected_scene'
        modal_root = data_root / '09_official_test/03_modal_features'
        shard_inputs = [input_root / f'manifest_shard_{shard:03d}.json' for shard in range(shards)]
        for shard, output in enumerate(shard_inputs):
            if not _passed(output):
                jobs.append(_job(dataset, f'cache__{shard}', output, 'cache-inputs', '--shard-id', shard, '--num-shards', shards, '--device', 'cuda', '--official-test'))
        input_manifest = input_root / 'manifest.json'
        if all((_passed(path) for path in shard_inputs)) and (not _passed(input_manifest)):
            jobs.append(_job(dataset, 'finalize-inputs', input_manifest, 'finalize-inputs', '--official-test'))
        infer_manifests = [scene_root / f'shard_{shard:03d}/manifest.json' for shard in range(shards)]
        if _passed(input_manifest):
            for shard, output in enumerate(infer_manifests):
                if not _passed(output):
                    jobs.append(_job(dataset, f'infer__{shard}', output, 'infer', '--shard-id', shard, '--num-shards', shards, '--device', 'cuda', '--official-test'))
        modal_shards = [modal_root / f'index_shard_{shard:03d}.json' for shard in range(shards)]
        for shard, (infer, output) in enumerate(zip(infer_manifests, modal_shards)):
            if _passed(input_manifest) and _passed(infer) and (not _passed(output)):
                jobs.append(_job(dataset, f'modal__{shard}', output, 'build-modal', '--shard-id', shard, '--num-shards', shards, '--official-test'))
        modal_manifest = modal_root / 'manifest.json'
        if all((_passed(path) for path in modal_shards)) and (not _passed(modal_manifest)):
            jobs.append(_job(dataset, 'finalize-modal', modal_manifest, 'finalize-modal', '--official-test'))
        if not _passed(modal_manifest):
            continue
        scopes = list(range(5)) if dataset == 'bracs' else [None]
        for fold in scopes:
            scope = f'fold_{fold:02d}' if fold is not None else 'final'
            fold_args = () if fold is None else ('--fold', fold)
            residual = data_root / f'09_official_test/04_residual/{scope}/manifest.json'
            shuffle = data_root / f'09_official_test/04_shuffle/{scope}/mapping.json'
            if not _passed(residual):
                jobs.append(_job(dataset, f'residual__{scope}', residual, 'official-fit-residual', *fold_args, '--device', 'cuda'))
            if not _passed(shuffle):
                jobs.append(_job(dataset, f'shuffle__{scope}', shuffle, 'official-build-shuffle', *fold_args))
            if not (_passed(residual) and _passed(shuffle)):
                continue
            meanpool = data_root / f'09_official_test/05_meanpool_cache/{scope}/manifest.json'
            if not _passed(meanpool):
                jobs.append(_job(dataset, f'meanpool-cache__{scope}', meanpool, 'official-build-meanpool', *fold_args))
            for arm in ARMS:
                arm_cache = data_root / f'09_official_test/05_meanpool_cache/{scope}/{arm}/manifest.json'
                for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
                    if encoder == 'meanpool' and (not _passed(arm_cache)):
                        continue
                    for seed in seeds:
                        output = result_root / f'09_official_test/geometry/{encoder}/{arm}/{scope}/seed_{seed}/manifest.json'
                        if not _passed(output):
                            capability = 'a800' if encoder == 'deepsets' else 'any'
                            jobs.append(_job(dataset, f'geometry__{encoder}__{arm}__{scope}__{seed}', output, 'official-train-geometry', *fold_args, '--arm', arm, '--encoder', encoder, '--seed', seed, '--device', 'cuda', capability=capability))
        if dataset == 'bracs':
            for fold in range(5):
                for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
                    for seed in seeds:
                        output = result_root / f'09_official_test/b0/bracs/{encoder}/fold_{fold:02d}/seed_{seed}/manifest.json'
                        if not _passed(output):
                            jobs.append(_job(dataset, f'b0__{encoder}__fold_{fold:02d}__{seed}', output, 'official-train-b0', '--fold', fold, '--encoder', encoder, '--seed', seed, '--device', 'cuda', capability='a800'))
        for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
            geometry_outputs = []
            for fold in scopes:
                scope = f'fold_{fold:02d}' if fold is not None else 'final'
                geometry_outputs.extend((result_root / f'09_official_test/geometry/{encoder}/{arm}/{scope}/seed_{seed}/manifest.json' for arm in ARMS for seed in seeds))
            b0_outputs = [] if dataset != 'bracs' else [result_root / f'09_official_test/b0/bracs/{encoder}/fold_{fold:02d}/seed_{seed}/manifest.json' for fold in range(5) for seed in seeds]
            fused = result_root / f'09_official_test/predictions/{encoder}.json'
            if all((_passed(path) for path in geometry_outputs + b0_outputs)) and (not _passed(fused)):
                jobs.append(_job(dataset, f'fusion__{encoder}', fused, 'official-fuse', '--encoder', encoder))
        summary = result_root / '09_official_test/summary.json'
        predictions = [result_root / f'09_official_test/predictions/{encoder}.json' for encoder in ('meanpool', 'deepsets')]
        if all((_passed(path) for path in predictions)) and (not _passed(summary)):
            jobs.append(_job(dataset, 'evaluate', summary, 'official-evaluate'))
    rank_names = ('cache__', 'finalize-inputs', 'infer__', 'modal__', 'finalize-modal', 'residual__', 'shuffle__', 'meanpool-cache__', 'b0__', 'geometry__meanpool', 'geometry__deepsets', 'fusion__', 'evaluate')

    def priority(job):
        return (next((i for i, name in enumerate(rank_names) if name in job.key), len(rank_names)), DATASET_ORDER.index(job.key.split('__')[1]), job.key)
    return sorted(jobs, key=priority)

def complete() -> bool:
    return all((_passed(_paths(dataset)[1] / '09_official_test/summary.json') for dataset in DATASET_ORDER))

def run_loop(capability: str, owner: str, poll_seconds: int=30) -> int:
    state = GLOBAL_RESULT / f"runtime/job_queue/workers/{owner.replace(':', '_')}.json"
    while True:
        if complete():
            _atomic_json(GLOBAL_RESULT / 'runtime/official_test_complete.json', {'status': 'PASS', 'time': time.time(), 'worker': owner})
            _atomic_json(state, {'status': 'DONE', 'phase': 'official_test', 'time': time.time(), 'worker': owner})
            return 0
        jobs = [job for job in discover_official_jobs() if job.capability == 'any' or capability == 'a800']
        progressed = False
        for job in jobs:
            if _attempts(job) >= 2 and (not _ready(job)):
                continue
            claim = _claim(job, owner)
            if claim is None:
                continue
            _atomic_json(state, {'status': 'RUNNING', 'phase': 'official_test', 'time': time.time(), 'worker': owner, 'job': job.key})
            _run(job, claim, owner)
            progressed = True
            break
        if not progressed:
            _atomic_json(state, {'status': 'WAITING', 'phase': 'official_test', 'time': time.time(), 'worker': owner, 'ready_jobs': len(jobs)})
            time.sleep(poll_seconds)
