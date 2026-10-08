from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from datetime import datetime, timezone
import hashlib
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
import socket
import subprocess
from typing import Any

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def atomic_torch_save(path: str | Path, value: Any) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)

def atomic_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)

def runtime_identity() -> dict[str, Any]:
    value: dict[str, Any] = {'host': socket.gethostname(), 'started_at': datetime.now(timezone.utc).isoformat()}
    try:
        value['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        value['git_commit'] = None
    try:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=10).splitlines()
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        value['gpu'] = next((line for line in lines if line.split(',')[0].strip() == visible), None)
    except (OSError, subprocess.SubprocessError):
        value['gpu'] = None
    return value
