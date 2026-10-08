from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping
import hashlib
from celllift.runtime import json
import os
import re
import tempfile

def safe_component(value: str, max_prefix: int=48) -> str:
    text = str(value)
    prefix = re.sub('[^A-Za-z0-9._-]+', '-', text).strip('-._') or 'item'
    return f'{prefix[:max_prefix]}-{hashlib.sha256(text.encode()).hexdigest()[:12]}'

def sha256_file(path: str | Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f'.{destination.name}.tmp.', dir=destination.parent)
    os.close(handle)
    return Path(raw)

def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    temporary = _temporary(destination)
    try:
        with temporary.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + '\n'
    atomic_bytes(path, encoded.encode('utf-8'))

def atomic_npz(path: str | Path, **arrays: Any) -> None:
    import numpy as np
    destination = Path(path)
    temporary = _temporary(destination)
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_parquet(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    materialized = list(rows)
    destination = Path(path)
    temporary = _temporary(destination)
    try:
        pq.write_table(pa.Table.from_pylist(materialized), temporary, compression='zstd', row_group_size=65536)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(materialized)

def atomic_torch(path: str | Path, payload: Any) -> None:
    import torch
    destination = Path(path)
    temporary = _temporary(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_parquet(path: str | Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def load_npz(path: str | Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path) as value:
        return {key: value[key].copy() for key in value.files}
