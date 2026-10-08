from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping, Sequence
import csv
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

def sha256_tree(root: str | Path, suffixes: Sequence[str]=('.py',)) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    paths = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
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

def atomic_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    destination = Path(path)
    temporary = _temporary(destination)
    count = 0
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction='raise')
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count

def atomic_torch(path: str | Path, payload: Any) -> None:
    import torch
    destination = Path(path)
    temporary = _temporary(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

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
