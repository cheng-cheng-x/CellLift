from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Sequence
from celllift.runtime import yaml
SEEDS = (17, 42, 73, 101, 137)
FOLDS = range(5)
EXPERTS = ('E0_RGB_MASK2D', 'E1_MASK2D', 'E2_MASK_DIRECT3D', 'E3_MASK_SHUF_DIRECT3D', 'E4_MASK_RESIDUAL3D', 'E5_MASK_SHUF_RESIDUAL3D')

@dataclass(frozen=True)
class Job:
    job_id: str
    command: tuple[str, ...]
    complete: Path
    dependencies: tuple[Path, ...]
    uses_gpu: bool = True

def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'PASS'
    except (OSError, ValueError):
        return False

def _gpu_has_headroom(gpu_index: int, *, expected_peak_mib: int=8000, reserve_fraction: float=0.2) -> bool:
    try:
        query = subprocess.check_output(('nvidia-smi', f'--id={int(gpu_index)}', '--query-gpu=uuid,compute_mode,memory.total,memory.used', '--format=csv,noheader,nounits'), text=True, timeout=15).strip()
        uuid, compute_mode, total, used = [value.strip() for value in query.split(',')]
        applications = subprocess.check_output(('nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'), text=True, timeout=15)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    application_rows = []
    for line in applications.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) == 2:
            application_rows.append((fields[0], int(fields[1])))
    occupied = any((app_uuid == uuid for app_uuid, _ in application_rows))
    for app_uuid, pid in application_rows:
        if app_uuid != uuid:
            continue
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode('utf-8', errors='replace')
        except OSError:
            continue
        if 'public_patch_tasks/breast_roi_baseline' in command:
            return False
    if 'exclusive' in compute_mode.lower() and occupied:
        return False
    total_mib, used_mib = (int(total), int(used))
    required = int(expected_peak_mib + reserve_fraction * total_mib)
    return total_mib - used_mib >= required

def _matrix(code_root: Path, config: Path, cfg: dict) -> list[Job]:
    run = code_root / 'run.py'
    python = Path(cfg['runtime']['python'])
    data = Path(cfg['paths']['data_root'])
    result = Path(cfg['paths']['result_root'])
    anchors = data / '04_direct3d' / 'anchor_cache_manifest.json'
    jobs: list[Job] = []
    for fold in FOLDS:
        destination = data / '05_probe_residuals' / 'development' / f'fold_{fold:02d}'
        jobs.append(Job(f'probe_fold{fold}', (str(python), str(run), 'fit_probe', '--config', str(config), '--phase', 'development', '--fold', str(fold), '--device', 'cuda'), destination / 'manifest.json', (anchors,), True))
    probe_manifests = tuple((data / '05_probe_residuals' / 'development' / f'fold_{fold:02d}' / 'manifest.json' for fold in FOLDS))
    jobs.append(Job('residual_manifest', (str(python), str(run), 'build_residual3d', '--config', str(config), '--phase', 'development'), data / '05_probe_residuals' / 'development' / 'residual_manifest.json', probe_manifests, False))
    for task, encoder in (('t7', 'meanpool'), ('t7', 'deepsets'), ('t3', 'meanpool'), ('t3', 'deepsets')):
        for expert in EXPERTS:
            for fold in FOLDS:
                dependencies = [anchors]
                if 'RESIDUAL' in expert:
                    dependencies.append(data / '05_probe_residuals' / 'development' / f'fold_{fold:02d}' / 'probe_predictions.parquet')
                for seed in SEEDS:
                    output = result / 'experts' / 'development' / task / encoder / expert / f'fold_{fold:02d}' / f'seed_{seed}'
                    job_id = f'expert_{task}_{encoder}_{expert}_f{fold}_s{seed}'
                    jobs.append(Job(job_id, (str(python), str(run), 'train_experts', '--config', str(config), '--phase', 'development', '--task', task, '--encoder', encoder, '--expert', expert, '--fold', str(fold), '--seed', str(seed), '--device', 'cuda'), output / 'manifest.json', tuple(dependencies), True))
        expert_manifests = tuple((result / 'experts' / 'development' / task / encoder / expert / f'fold_{fold:02d}' / f'seed_{seed}' / 'manifest.json' for expert in EXPERTS for fold in FOLDS for seed in SEEDS))
        metrics = result / 'metrics' / 'development' / task / encoder
        jobs.append(Job(f'fusion_{task}_{encoder}', (str(python), str(run), 'fit_fusion', '--config', str(config), '--phase', 'development', '--task', task, '--encoder', encoder, '--device', 'cpu'), metrics / 'manifest.json', expert_manifests, False))
        jobs.append(Job(f'report_{task}_{encoder}', (str(python), str(run), 'report', '--config', str(config), '--phase', 'development', '--task', task, '--encoder', encoder, '--device', 'cpu'), metrics / 'report.md', (metrics / 'manifest.json',), False))
    return jobs

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    code_root = Path(__file__).resolve().parents[1]
    result = Path(cfg['paths']['result_root'])
    queue = result / 'runtime' / 'development_queue'
    queue.mkdir(parents=True, exist_ok=True)
    jobs = _matrix(code_root, config, cfg)
    recorder = Path(__file__).with_name('run_recorded_job.py')
    worker = f'{socket.gethostname()}-gpu{args.gpu_index}-pid{os.getpid()}'
    matrix_path = queue / 'matrix.json'
    if not matrix_path.is_file():
        temporary = matrix_path.with_name(f'.{matrix_path.name}.tmp.{os.getpid()}')
        temporary.write_text(json.dumps({'status': 'REGISTERED', 'job_count': len(jobs), 'jobs': [{'job_id': job.job_id, 'command': job.command, 'complete': str(job.complete), 'dependencies': [str(path) for path in job.dependencies], 'uses_gpu': job.uses_gpu} for job in jobs]}, indent=2) + '\n', encoding='utf-8')
        try:
            os.link(temporary, matrix_path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    while True:
        incomplete = [job for job in jobs if not _passed(job.complete) and (not (job.complete.suffix == '.md' and job.complete.is_file()))]
        if not incomplete:
            return 0
        progressed = False
        for job in incomplete:
            if not all((_passed(path) if path.suffix == '.json' else path.is_file() for path in job.dependencies)):
                continue
            if job.uses_gpu and (not _gpu_has_headroom(args.gpu_index)):
                continue
            claim = queue / f'{job.job_id}.claim'
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            (claim / 'owner.json').write_text(json.dumps({'worker': worker, 'time': time.time()}) + '\n', encoding='utf-8')
            is_expert = 'train_experts' in job.command
            tile_budget = int(cfg['training']['tile_budget'])
            object_budget = int(cfg['training']['object_budget'])
            result_code = 1
            for attempt in range(4):
                runtime = result / 'runtime' / 'development' / f'{job.job_id}.attempt{attempt}.json'
                log = result / 'logs' / 'development' / f'{job.job_id}.attempt{attempt}.log'
                child_parts = list(job.command)
                if job.uses_gpu:
                    device_position = child_parts.index('--device') + 1
                    child_parts[device_position] = f'cuda:{args.gpu_index}'
                child = (*child_parts, *(('--tile-budget', str(tile_budget), '--object-budget', str(object_budget)) if is_expert else ()))
                command: Sequence[str] = (sys.executable, str(recorder), '--manifest', str(runtime), '--log', str(log), '--config', str(config), *(('--gpu-index', str(args.gpu_index)) if job.uses_gpu else ()), '--', *child)
                result_code = subprocess.run(command, check=False).returncode
                if result_code == 0:
                    break
                message = log.read_text(encoding='utf-8', errors='replace').lower()
                oom = 'out of memory' in message or 'cuda error: out of memory' in message
                if not is_expert or not oom or attempt == 3:
                    return result_code
                tile_budget = max(1, tile_budget // 2)
                object_budget = max(1, object_budget // 2)
            if result_code != 0:
                return result_code
            progressed = True
            break
        if not progressed:
            time.sleep(max(5, args.poll_seconds))
if __name__ == '__main__':
    raise SystemExit(main())
