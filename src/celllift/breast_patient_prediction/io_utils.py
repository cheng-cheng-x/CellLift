from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
import tempfile
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    path.write_text(text, encoding='utf-8')
    return sha256_bytes(text.encode('utf-8'))

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))

def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return sha256_bytes(text.encode('utf-8'))

def stable_shard(key: str, count: int) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest(), 'big') % int(count)

def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f'.{destination.name}.tmp.', dir=str(destination.parent))
    os.close(handle)
    return Path(raw)

def atomic_bytes(path: Path, payload: bytes) -> None:
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

def atomic_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + '\n'
    atomic_bytes(path, encoded.encode('utf-8'))

def atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)

def atomic_parquet(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
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

def atomic_npz(path: Path, compressed: bool=True, **arrays: Any) -> None:
    import numpy as np
    destination = Path(path)
    temporary = _temporary(destination)
    writer = np.savez_compressed if compressed else np.savez
    try:
        with temporary.open('wb') as stream:
            writer(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
