from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import hashlib
from celllift.runtime import json
import os
import re
import tempfile
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping, Sequence
TARGET_MPP_UM_PER_PX = 0.46
TILE_SIZE_PX = 1024
EXPECTED_ROI_COUNT = 4227
EXPECTED_ROI_SPLIT_COUNTS = {'train': 3163, 'val': 494, 'test': 570}
EXPECTED_WSI_SPLIT_COUNTS = {'train': 234, 'val': 47, 'test': 69}
T7_LABELS = ('N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC')
T3_LABELS = ('BT', 'AT', 'MT')

class BracsDataError(RuntimeError):
    pass

def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_path = tempfile.mkstemp(prefix=f'.{destination.name}.tmp.', dir=str(destination.parent))
    os.close(handle)
    return Path(raw_path)

def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')
    atomic_write_bytes(path, encoded)

def atomic_write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> int:
    destination = Path(path)
    temporary = _temporary_path(destination)
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

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: str | Path, chunk_size: int=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (chunk := stream.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()

def safe_component(value: str, *, max_prefix: int=48) -> str:
    text = str(value).strip()
    prefix = re.sub('[^A-Za-z0-9._-]+', '-', text).strip('-._') or 'item'
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
    return f'{prefix[:max_prefix]}-{digest}'

def stable_tile_id(roi_id: str, row: int, column: int) -> str:
    if row < 0 or column < 0:
        raise ValueError('tile row and column must be non-negative')
    roi_digest = hashlib.sha256(str(roi_id).encode('utf-8')).hexdigest()[:20]
    return f'bracs-{roi_digest}-r{row:04d}-c{column:04d}'
