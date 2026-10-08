from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import gzip
import hashlib
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import numpy as np
from PIL import Image

def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_json(path: str | Path, default: Any=None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + '\n', encoding='utf-8')
    os.replace(temporary, output)

def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def write_tsv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str] | None=None) -> None:
    output = Path(path)
    materialized = list(rows)
    if fields is None:
        fields = []
        for row in materialized:
            for key in row:
                if key not in fields:
                    fields.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, output)

def write_gzip_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=6) as handle:
        json.dump(payload, handle, separators=(',', ':'), sort_keys=True, allow_nan=True)
    os.replace(temporary, output)

def read_gzip_json(path: str | Path) -> Any:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def fingerprint(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()

def stable_slot(identifier: str, count: int) -> int:
    if count < 1:
        raise ValueError('slot count must be positive')
    digest = hashlib.sha256(identifier.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % count

def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    with temporary.open('wb') as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)

def validate_rgb_png(path: str | Path, expected_width: int=1024, expected_height: int=1024) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == 'PNG' and image.mode == 'RGB' and (image.size == (expected_width, expected_height))
    except (FileNotFoundError, OSError):
        return False

def validate_label_map(path: str | Path, expected_width: int=1024, expected_height: int=1024) -> bool:
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        return array.shape == (expected_height, expected_width) and array.dtype.kind in {'i', 'u'} and (int(array.min(initial=0)) >= 0)
    except (FileNotFoundError, OSError, ValueError):
        return False

def algorithm_fingerprint(cfg: dict[str, Any]) -> str:
    return fingerprint({'experiment_id': cfg['experiment_id'], 'physical': cfg['physical'], 'runtime_cellpose_version': cfg['runtime']['cellpose_version'], 'nucleus': cfg['nucleus'], 'cell': cfg['cell'], 'pairing': cfg['pairing']})

def assigned_worker(cfg: dict[str, Any], identifier: str) -> tuple[int, int]:
    slot = stable_slot(identifier, int(cfg['runtime']['virtual_shards']))
    matches = [int(worker['worker_id']) for worker in cfg['workers'] if slot in [int(value) for value in worker['virtual_slots']]]
    if len(matches) != 1:
        raise RuntimeError(f'virtual slot {slot} has {len(matches)} assigned workers')
    return (matches[0], slot)

def worker_config(cfg: dict[str, Any], worker_id: int) -> dict[str, Any]:
    matches = [worker for worker in cfg['workers'] if int(worker['worker_id']) == worker_id]
    if len(matches) != 1:
        raise RuntimeError(f'unknown or duplicate worker id: {worker_id}')
    return matches[0]
