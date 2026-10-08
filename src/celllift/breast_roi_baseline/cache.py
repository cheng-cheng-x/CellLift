from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping
import numpy as np
from celllift.breast_roi_baseline.features import complete_geometry_from_ellipsoids

def sha256_file(path: str | Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(chunk_size), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, destination)

def _read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _rotation(table: Mapping[str, list[Any]], selected: np.ndarray) -> np.ndarray:
    values = np.column_stack([np.asarray(table[f'rotation_{row}{column}'], np.float32)[selected] for row in range(3) for column in range(3)])
    return values.reshape(-1, 3, 3)

def _fixed_list(values: np.ndarray, width: int):
    import pyarrow as pa
    matrix = np.asarray(values, np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f'expected [N,{width}], got {matrix.shape}')
    return pa.FixedSizeListArray.from_arrays(pa.array(matrix.reshape(-1), type=pa.float32()), width)

def geometry_shard_to_anchor_table(geometry_path: str | Path, graph_metadata: Mapping[str, Mapping[str, Any]]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = Path(geometry_path)
    table = pq.read_table(source, partitioning=None).to_pydict()
    count = len(table['graph_id'])
    if count == 0 or count % 2:
        raise RuntimeError(f'geometry shard is empty or unpaired: {source}')
    object_type = np.asarray(table['object_type'], object)
    nucleus_index = np.flatnonzero(object_type == 'nucleus')
    cell_index = np.flatnonzero(object_type == 'cell')
    if len(nucleus_index) != len(cell_index) or not np.array_equal(cell_index, nucleus_index + 1):
        raise RuntimeError('geometry rows are not alternating nucleus/cell pairs')
    graph = np.asarray(table['graph_id'], object)
    anchor = np.asarray(table['anchor_id'], np.int64)
    if not np.array_equal(graph[nucleus_index], graph[cell_index]) or not np.array_equal(anchor[nucleus_index], anchor[cell_index]):
        raise RuntimeError('nucleus/cell geometry keys differ')

    def axes(index: np.ndarray) -> np.ndarray:
        return np.column_stack([np.asarray(table[name], np.float32)[index] for name in ('axis_a_um', 'axis_b_um', 'axis_c_um')])
    raw, valid_ncr = complete_geometry_from_ellipsoids(axes(nucleus_index), _rotation(table, nucleus_index), axes(cell_index), _rotation(table, cell_index))
    rays = np.asarray([table['nucleus_rays_um'][index] for index in nucleus_index], np.float32)
    if rays.shape != (len(nucleus_index), 36):
        raise RuntimeError('geometry nucleus rays do not have width 36')
    metadata = []
    for value in graph[nucleus_index]:
        key = str(value)
        if key not in graph_metadata:
            raise RuntimeError(f'geometry graph is absent from tile manifest: {key}')
        metadata.append(graph_metadata[key])
    columns = {'graph_id': pa.array([str(value) for value in graph[nucleus_index]]), 'anchor_id': pa.array(anchor[nucleus_index]), 'roi_id': pa.array([str(row['roi_id']) for row in metadata]), 'tile_id': pa.array([str(row['tile_id']) for row in metadata]), 'wsi_id': pa.array([str(row['wsi_id']) for row in metadata]), 'split_new': pa.array([str(row['split_new']) for row in metadata]), 'label_7': pa.array(np.asarray([int(row['label_7']) for row in metadata], np.int8)), 'label_3': pa.array(np.asarray([int(row['label_3']) for row in metadata], np.int8)), 'validation_fold': pa.array([None if row.get('validation_fold') is None else int(row['validation_fold']) for row in metadata], type=pa.int8()), 'final_validation_fold': pa.array([None if row.get('final_validation_fold') is None else int(row['final_validation_fold']) for row in metadata], type=pa.int8()), 'rays36': _fixed_list(rays, 36), 'raw_geometry9': _fixed_list(raw, 9), 'valid_ncr3d': pa.array(valid_ncr)}
    return pa.table(columns)

def build_anchor_cache(*, geometry_manifest: str | Path, tile_manifest: str | Path, output_dir: str | Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    destination = Path(output_dir)
    manifest_path = destination / 'anchor_cache_manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS' and all((Path(item['path']).is_file() and sha256_file(item['path']) == item['sha256'] for item in previous.get('shards', []))):
            return previous
        raise RuntimeError('existing anchor cache manifest is not reusable')
    geometry = json.loads(Path(geometry_manifest).read_text(encoding='utf-8'))
    if geometry.get('status') != 'PASS':
        raise RuntimeError('dual geometry manifest is not PASS')
    tile_rows = _read_rows(Path(tile_manifest))
    graph_metadata = {str(row['graph_id']): row for row in tile_rows}
    if len(graph_metadata) != len(tile_rows):
        raise RuntimeError('duplicate graph_id in BRACS tile manifest')
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    anchors = 0
    for index, item in enumerate(geometry['shards']):
        source = Path(item['path'])
        if sha256_file(source) != item['sha256']:
            raise RuntimeError(f'geometry shard checksum mismatch: {source}')
        target = destination / f'anchors_{index:04d}.parquet'
        temporary = target.with_name(f'.{target.name}.tmp.{os.getpid()}')
        table = geometry_shard_to_anchor_table(source, graph_metadata)
        pq.write_table(table, temporary, compression='zstd', row_group_size=65536)
        os.replace(temporary, target)
        anchors += len(table)
        outputs.append({'path': str(target), 'rows': len(table), 'sha256': sha256_file(target)})
    if anchors != int(geometry['anchor_count']):
        raise RuntimeError('anchor cache coverage differs from dual geometry')
    payload = {'status': 'PASS', 'anchors': anchors, 'graphs': len(graph_metadata), 'token_source': 'compatibility_score_direct_geometry_and_observed_nucleus_rays36', 'shards': outputs, 'geometry_manifest': str(geometry_manifest), 'geometry_manifest_sha256': sha256_file(geometry_manifest), 'tile_manifest': str(tile_manifest), 'tile_manifest_sha256': sha256_file(tile_manifest)}
    atomic_json(manifest_path, payload)
    return payload
