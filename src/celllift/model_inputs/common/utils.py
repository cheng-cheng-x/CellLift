from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
import re
import zipfile
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

def sha256_file(path: str | Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def stable_shard(key: str, count: int) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest(), 'big') % count

def shard_hex(key: str) -> str:
    return hashlib.blake2b(key.encode('utf-8'), digest_size=1).hexdigest()

def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, destination)

def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_config(path: str | Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TypeError('configuration root must be a mapping')
    return payload

def atomic_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, temporary, compression='zstd', row_group_size=131072)
    os.replace(temporary, destination)

def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def require_upstream_pass(raw_root: str | Path) -> None:
    root = Path(raw_root) / '00_manifest'
    for name in ('verification.status.json', 'semantic.status.json'):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        if payload.get('status') != 'PASS':
            raise RuntimeError(f'upstream gate is not PASS: {path}')

def prepare_output_tree(data_root: str | Path) -> Path:
    root = Path(data_root)
    for child in ('00_manifest', '01_standardized_rgb', '02_cellpose_dual', '03_graph_cache', '04_labels_splits', '05_qc', 'logs'):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root
_NS = {'m': _public_resource('artifact_0059')}

def _xlsx_column(cell_ref: str) -> int:
    letters = re.match('[A-Z]+', cell_ref).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1

def read_xlsx_rows(path: str | Path, sheet_index: int=0) -> list[list[Any]]:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            for item in root.findall('m:si', _NS):
                shared.append(''.join((node.text or '' for node in item.iterfind('.//m:t', _NS))))
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheets = workbook.find('m:sheets', _NS)
        sheet = list(sheets)[sheet_index]
        relation_id = sheet.attrib[_public_resource('artifact_0060')]
        rel_root = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = None
        for rel in rel_root:
            if rel.attrib.get('Id') == relation_id:
                target = rel.attrib['Target']
                break
        if target is None:
            raise RuntimeError(f'worksheet relation not found in {path}')
        target = target.lstrip('/')
        if not target.startswith('xl/'):
            target = 'xl/' + target
        xml = ET.fromstring(archive.read(target))
        result: list[list[Any]] = []
        for row in xml.findall('.//m:sheetData/m:row', _NS):
            values: dict[int, Any] = {}
            for cell in row.findall('m:c', _NS):
                index = _xlsx_column(cell.attrib['r'])
                kind = cell.attrib.get('t')
                value_node = cell.find('m:v', _NS)
                if kind == 'inlineStr':
                    value = ''.join((node.text or '' for node in cell.iterfind('.//m:t', _NS)))
                elif value_node is None:
                    value = None
                elif kind == 's':
                    value = shared[int(value_node.text)]
                elif kind == 'b':
                    value = bool(int(value_node.text))
                else:
                    raw = value_node.text or ''
                    try:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                    except ValueError:
                        value = raw
                values[index] = value
            width = max(values, default=-1) + 1
            result.append([values.get(index) for index in range(width)])
        return result

def xlsx_dicts(path: str | Path) -> list[dict[str, Any]]:
    rows = read_xlsx_rows(path)
    if not rows:
        return []
    headers = [str(value).strip() if value is not None else f'column_{index}' for index, value in enumerate(rows[0])]
    return [{headers[index]: row[index] if index < len(row) else None for index in range(len(headers))} for row in rows[1:] if any((value not in (None, '') for value in row))]

def normalize_patch_id(value: Any) -> str:
    name = Path(str(value).strip()).name
    return Path(name).stem

def one_hot_label(row: dict[str, Any], names: tuple[str, ...]) -> str:
    selected = [name for name in names if int(row.get(name) or 0) == 1]
    if len(selected) != 1:
        raise RuntimeError(f'expected exactly one label among {names}: {row}')
    return selected[0]
