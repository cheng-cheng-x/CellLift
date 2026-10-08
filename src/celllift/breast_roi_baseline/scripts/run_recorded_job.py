from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import hashlib
from celllift.runtime import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path

def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def _nvidia(index: int, query: str) -> str | None:
    result = subprocess.run(['nvidia-smi', '-i', str(index), f'--query-gpu={query}', '--format=csv,noheader,nounits'], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _git_state() -> dict[str, object]:
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], check=False, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], check=False, capture_output=True, text=True)
    return {'head': head.stdout.strip() if head.returncode == 0 else None, 'dirty': bool(status.stdout.strip()) if status.returncode == 0 else None}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--gpu-index', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    started = time.monotonic()
    gpu_uuid = _nvidia(args.gpu_index, 'uuid') if args.gpu_index is not None else None
    gpu_name = _nvidia(args.gpu_index, 'name') if args.gpu_index is not None else None
    peak_memory = 0
    with args.log.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if args.gpu_index is not None:
                value = _nvidia(args.gpu_index, 'memory.used')
                if value:
                    peak_memory = max(peak_memory, int(float(value.splitlines()[0])))
            time.sleep(2)
        return_code = int(process.returncode)
    payload: dict[str, object] = {'status': 'PASS' if return_code == 0 else 'FAILED', 'return_code': return_code, 'command': command, 'host': platform.node(), 'pid': process.pid, 'gpu_index': args.gpu_index, 'gpu_uuid': gpu_uuid, 'gpu_name': gpu_name, 'peak_gpu_memory_mib': peak_memory if args.gpu_index is not None else None, 'started_utc': start.isoformat(), 'ended_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started, 'config': str(args.config) if args.config else None, 'config_sha256': _sha256(args.config) if args.config else None, 'git': _git_state(), 'log': str(args.log), 'log_sha256': _sha256(args.log)}
    _atomic_json(args.manifest, payload)
    return return_code
if __name__ == '__main__':
    raise SystemExit(main())
