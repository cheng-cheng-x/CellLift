from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import csv
import fcntl
from celllift.runtime import json
import os
import socket
import subprocess
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
from celllift.segmentation.common import load_config, read_json, worker_config, write_json
SCRIPT_ROOT = Path(__file__).resolve().parent

def gpu_state(index: int) -> tuple[dict[str, Any], set[str]]:
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
    records: dict[int, dict[str, Any]] = {}
    for fields in csv.reader(output.splitlines()):
        gpu_index, name, uuid, utilization, used, free, total = [field.strip() for field in fields]
        records[int(gpu_index)] = {'index': int(gpu_index), 'name': name, 'uuid': uuid, 'utilization_percent': float(utilization), 'used_memory_mb': float(used), 'free_memory_mb': float(free), 'total_memory_mb': float(total)}
    process_output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
    active = {line.strip() for line in process_output.splitlines() if line.strip()}
    if index not in records:
        raise RuntimeError(f'GPU index {index} is unavailable')
    return (records[index], active)

def wait_for_idle_gpu(cfg: dict[str, Any], worker_id: int, gpu_index: int, status_path: Path) -> dict[str, Any]:
    while True:
        first, active_first = gpu_state(gpu_index)
        write_json(status_path, {'status': 'probing_gpu', 'worker_id': worker_id, 'gpu_index': gpu_index, 'first_sample': first, 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()})
        time.sleep(int(cfg['runtime']['gpu_probe_seconds']))
        second, active_second = gpu_state(gpu_index)
        idle = first['uuid'] not in active_first and second['uuid'] not in active_second and (first['utilization_percent'] <= float(cfg['runtime']['gpu_max_utilization_percent'])) and (second['utilization_percent'] <= float(cfg['runtime']['gpu_max_utilization_percent'])) and (first['used_memory_mb'] <= float(cfg['runtime']['gpu_max_used_memory_mb'])) and (second['used_memory_mb'] <= float(cfg['runtime']['gpu_max_used_memory_mb']))
        if idle:
            return second
        write_json(status_path, {'status': 'waiting_for_idle_gpu', 'worker_id': worker_id, 'gpu_index': gpu_index, 'first_sample': first, 'second_sample': second, 'active_compute_gpu_uuids': sorted(active_first | active_second), 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()})
        if Path(cfg['result_root'], 'runtime/STOP').is_file():
            raise RuntimeError('global STOP sentinel detected')
        time.sleep(30)

def run(config_path: Path, worker_id: int) -> dict[str, Any]:
    cfg = load_config(config_path)
    worker = worker_config(cfg, worker_id)
    root = Path(cfg['result_root'])
    gpu_index = int(worker['gpu_index'])
    status_path = root / 'runtime/agents' / f'worker_{worker_id:02d}.json'
    lock_path = Path(f"{_public_resource('artifact_0066')}{cfg['experiment_id']}_gpu_{gpu_index}.lock")
    lock_handle = lock_path.open('w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f'local GPU lock already held: {lock_path}') from error
    log_root = root / 'logs/workers'
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(int(cfg['runtime']['worker_attempts'])):
            if (root / 'runtime/STOP').is_file():
                result = {'status': 'stopped', 'worker_id': worker_id, 'gpu_index': gpu_index, 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()}
                write_json(status_path, result)
                return result
            gpu = wait_for_idle_gpu(cfg, worker_id, gpu_index, status_path)
            environment = os.environ.copy()
            environment.update({'CUDA_VISIBLE_DEVICES': str(gpu_index), 'CELLPOSE_LOCAL_MODELS_PATH': str(cfg['runtime']['model_cache']), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'OPENCV_FOR_THREADS_NUM': '1', 'MPLCONFIGDIR': str(root / 'runtime/matplotlib')})
            log_path = log_root / f'worker_{worker_id:02d}_attempt_{attempt}.log'
            with log_path.open('a', encoding='utf-8') as log:
                process = subprocess.Popen([str(cfg['runtime']['python']), str(SCRIPT_ROOT / 'worker.py'), '--config', str(config_path), '--worker-id', str(worker_id)], stdout=log, stderr=subprocess.STDOUT, env=environment)
                while process.poll() is None:
                    worker_heartbeat = read_json(root / 'runtime/workers' / f'worker_{worker_id:02d}.json', {})
                    write_json(status_path, {'status': 'running', 'worker_id': worker_id, 'gpu_index': gpu_index, 'gpu_uuid': gpu['uuid'], 'gpu_name': gpu['name'], 'worker_pid': process.pid, 'attempt': attempt, 'worker_heartbeat': worker_heartbeat, 'log_path': str(log_path), 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()})
                    time.sleep(int(cfg['runtime']['heartbeat_seconds']))
            summary = read_json(root / '04_status/workers' / f'worker_{worker_id:02d}_summary.json', {})
            if process.returncode == 0 and summary.get('status') == 'PASS':
                result = {'status': 'complete', 'worker_id': worker_id, 'gpu_index': gpu_index, 'gpu_uuid': gpu['uuid'], 'gpu_name': gpu['name'], 'attempt': attempt, 'worker_summary': summary, 'log_path': str(log_path), 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()}
                write_json(status_path, result)
                return result
            write_json(status_path, {'status': 'retrying', 'worker_id': worker_id, 'gpu_index': gpu_index, 'attempt': attempt, 'returncode': process.returncode, 'worker_summary': summary, 'log_path': str(log_path), 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()})
            time.sleep(10)
        result = {'status': 'failed', 'worker_id': worker_id, 'gpu_index': gpu_index, 'reason': 'worker_failed_after_all_attempts', 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()}
        write_json(status_path, result)
        return result
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--worker-id', required=True, type=int)
    args = parser.parse_args()
    result = run(args.config, args.worker_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'complete' else 2
if __name__ == '__main__':
    raise SystemExit(main())
