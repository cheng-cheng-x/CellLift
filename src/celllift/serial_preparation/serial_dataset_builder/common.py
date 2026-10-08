from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import gzip
import hashlib
from celllift.runtime import json
import os
import tempfile
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

def read_json(path: str | Path, default: Any=None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=str(output.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def iter_tsv(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as handle:
        yield from csv.DictReader(handle, delimiter='\t')

def atomic_write_tsv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None=None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if fields is None:
        field_list: list[str] = []
        for row in materialized:
            for key in row:
                if key not in field_list:
                    field_list.append(key)
    else:
        field_list = list(fields)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=str(output.parent))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=field_list, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(materialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, output)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

def read_gzip_json(path: str | Path) -> Any:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def atomic_write_gzip_json(path: str | Path, payload: Any, *, compresslevel: int=6) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    try:
        with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=compresslevel) as handle:
            json.dump(payload, handle, separators=(',', ':'), sort_keys=True, allow_nan=False)
        os.replace(tmp, output)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

def sha256_file(path: str | Path, block_size: int=4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(block_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def fingerprint(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return sha256_text(normalized)

def stable_hash_u64(value: str) -> int:
    return int(hashlib.sha256(value.encode('utf-8')).hexdigest()[:16], 16)

def stable_bucket(value: str, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError('bucket_count must be positive')
    return stable_hash_u64(value) % bucket_count

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'pass'}

def parse_int(value: Any, default: int=0) -> int:
    if value in {None, '', 'nan', 'NaN'}:
        return default
    return int(float(str(value)))

def parse_float(value: Any, default: float=0.0) -> float:
    if value in {None, '', 'nan', 'NaN'}:
        return default
    return float(value)

def roi_layer_id(optimized_track_id: str, section_id: str | int) -> str:
    return f'{optimized_track_id}__section_{int(section_id):03d}'

def edge_id(track_id: str, left_section: str | int, right_section: str | int) -> str:
    return f'{track_id}__{int(left_section):03d}_{int(right_section):03d}'

def nucleus_entity_id(layer_index: int, nucleus_id: int) -> int:
    return int(layer_index) << 32 | int(nucleus_id)

def cell_entity_id(layer_index: int, cell_id: int) -> int:
    return int(layer_index) << 32 | 1 << 31 | int(cell_id)
