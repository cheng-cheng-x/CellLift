from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping
from .common import atomic_write_json, atomic_write_tsv

def atomic_write_parquet(path: str | Path, rows: list[dict[str, Any]], *, schema: Any=None, row_group_size: int=131072) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, tmp, compression='zstd', compression_level=3, use_dictionary=True, row_group_size=row_group_size, write_statistics=True)
    os.replace(tmp, output)

def write_table(path: str | Path, rows: list[dict[str, Any]], *, schema: Any=None, prefer_parquet: bool=True) -> str:
    output = Path(path)
    if prefer_parquet and output.suffix == '.parquet':
        try:
            atomic_write_parquet(output, rows, schema=schema)
            return 'parquet'
        except ModuleNotFoundError:
            fallback = output.with_suffix('.tsv')
            atomic_write_tsv(fallback, rows)
            return 'tsv_fallback_missing_pyarrow'
    atomic_write_tsv(output, rows)
    return 'tsv'

def compact_partitioned_rows(input_paths: Iterable[str | Path], output_path: str | Path, *, schema: Any=None, row_group_size: int=131072) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths = list(input_paths)
    tables = []
    total_rows = 0
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        if path.suffix == '.parquet':
            table = pq.read_table(path)
        else:
            with path.open('r', newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            table = pa.Table.from_pylist(rows, schema=schema)
        tables.append(table)
        total_rows += table.num_rows
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if tables:
        table = pa.concat_tables(tables, promote_options='default')
    else:
        table = pa.Table.from_pylist([], schema=schema)
    tmp = output.with_name(f'.{output.name}.{os.getpid()}.tmp')
    pq.write_table(table, tmp, compression='zstd', compression_level=3, use_dictionary=True, row_group_size=row_group_size)
    os.replace(tmp, output)
    return {'status': 'PASS', 'input_files': len(paths), 'rows': total_rows, 'output_path': str(output)}
