from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import hashlib
from celllift.runtime import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def skip_apple(path: Path) -> bool:
    return path.name.startswith('._') or '/._' in path.as_posix()

def sha256_file(path: str | Path, chunk: int=1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)

def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)

def write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table({'_empty': pa.array([], type=pa.int8())})
    else:
        table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(path.suffix + '.tmp')
    pq.write_table(table, tmp)
    tmp.replace(path)

def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path, partitioning=None).to_pylist()

def unique_values(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})

def image_mean_uint8(rgb: np.ndarray) -> float:
    return float(rgb.mean())
