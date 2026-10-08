from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.features.geometry import finite_slab_projection_polygons, morphology_2d_from_polygons, morphology_3d, radial_polygons
from celllift.set_encoding.features.ncr import compute_log_ncr
from celllift.set_encoding.features.tokens import TOKEN_COLUMNS, build_object_tokens

def _sha256(path: Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd', row_group_size=65536)
    os.replace(temporary, path)

def _ellipsoid_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([[row['center_x_um'], row['center_y_um'], row['center_z_um']] for row in rows], dtype=np.float64)
    axes = np.asarray([[row['axis_a_um'], row['axis_b_um'], row['axis_c_um']] for row in rows], dtype=np.float64)
    rotations = np.asarray([[[row[f'rotation_{i}{j}'] for j in range(3)] for i in range(3)] for row in rows], dtype=np.float64)
    return (centers, axes, rotations)

def _pair_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nucleus: dict[tuple[str, int], dict[str, Any]] = {}
    cell: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row['graph_id']), int(row['anchor_id']))
        target = nucleus if row['object_type'] == 'nucleus' else cell if row['object_type'] == 'cell' else None
        if target is None:
            raise ValueError(f"unexpected object_type {row['object_type']!r}")
        if key in target:
            raise ValueError(f'duplicate geometry key {key}')
        target[key] = row
    if nucleus.keys() != cell.keys():
        raise RuntimeError('nucleus/cell anchor keys do not match')
    keys = sorted(nucleus)
    return ([nucleus[key] for key in keys], [cell[key] for key in keys])

def _token_rows(source: list[dict[str, Any]], tokens: np.ndarray, *, object_type: str, area: np.ndarray, volume: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(source):
        item: dict[str, Any] = {'dataset': row['dataset'], 'graph_id': row['graph_id'], 'anchor_id': int(row['anchor_id']), 'object_type': object_type, 'split': row['split'], 'area_2d_um2': float(area[index]), 'volume_3d_um3': float(volume[index])}
        item.update({name: float(tokens[index, column]) for column, name in enumerate(TOKEN_COLUMNS)})
        output.append(item)
    return output

def build_one_shard(source: Path, *, nucleus_output: Path, cell_output: Path, ncr_output: Path, patch_width_um: float, patch_height_um: float, section_thickness_um: float) -> dict[str, Any]:
    import pyarrow.parquet as pq
    rows = pq.read_table(source, partitioning=None).to_pylist()
    nucleus_rows, cell_rows = _pair_rows(rows)
    count = len(nucleus_rows)
    if not count:
        raise RuntimeError(f'empty geometry shard: {source}')
    if any((len(row.get('nucleus_rays_um') or ()) != 36 for row in nucleus_rows)):
        raise RuntimeError('geometry cache lacks 36 observed nucleus rays')
    anchor_xy = np.asarray([[row['anchor_x_um'], row['anchor_y_um']] for row in nucleus_rows], dtype=np.float64)
    rays = np.asarray([row['nucleus_rays_um'] for row in nucleus_rows], dtype=np.float64)
    nucleus_2d = morphology_2d_from_polygons(radial_polygons(anchor_xy, rays))
    nucleus_centers, nucleus_axes, nucleus_rotations = _ellipsoid_arrays(nucleus_rows)
    cell_centers, cell_axes, cell_rotations = _ellipsoid_arrays(cell_rows)
    slab = (-section_thickness_um / 2.0, section_thickness_um / 2.0)
    cell_slab_centers = cell_centers.copy()
    cell_slab_centers[:, 2] -= section_thickness_um / 2.0
    cell_polygons, cell_intersects = finite_slab_projection_polygons(cell_slab_centers, cell_axes, cell_rotations, slab_bounds_um=slab, directions=64)
    cell_2d = morphology_2d_from_polygons(cell_polygons)
    if np.any(cell_2d.valid != cell_intersects):
        cell_2d = type(cell_2d)(**{**cell_2d.__dict__, 'valid': cell_2d.valid & cell_intersects})
    nucleus_3d = morphology_3d(nucleus_axes, nucleus_rotations)
    cell_3d = morphology_3d(cell_axes, cell_rotations)
    nucleus_positions = anchor_xy / np.asarray([patch_width_um, patch_height_um])
    cell_positions = cell_centers[:, :2] / np.asarray([patch_width_um, patch_height_um])
    nucleus_border = np.asarray([bool(row['anchor_border_flag']) for row in nucleus_rows], np.float32)
    cell_border = (np.any(cell_polygons[..., 0] <= 0, axis=1) | np.any(cell_polygons[..., 0] >= patch_width_um, axis=1) | np.any(cell_polygons[..., 1] <= 0, axis=1) | np.any(cell_polygons[..., 1] >= patch_height_um, axis=1)).astype(np.float32)
    cell_border[~cell_intersects] = 0.0
    nucleus_tokens = build_object_tokens(nucleus_2d, nucleus_positions, nucleus_border, nucleus_3d, geometry_mode='3d')
    cell_tokens = build_object_tokens(cell_2d, cell_positions, cell_border, cell_3d, geometry_mode='3d')
    _atomic_parquet(nucleus_output, _token_rows(nucleus_rows, nucleus_tokens, object_type='nucleus', area=nucleus_2d.area, volume=nucleus_3d.volume))
    _atomic_parquet(cell_output, _token_rows(cell_rows, cell_tokens, object_type='cell', area=cell_2d.area, volume=cell_3d.volume))
    audit_2d = compute_log_ncr(nucleus_2d.area, cell_2d.area)
    audit_3d = compute_log_ncr(nucleus_3d.volume, cell_3d.volume)
    ncr_rows = []
    for index, row in enumerate(nucleus_rows):
        ncr_rows.append({'dataset': row['dataset'], 'graph_id': row['graph_id'], 'anchor_id': int(row['anchor_id']), 'split': row['split'], 'nucleus_area_2d_um2': float(nucleus_2d.area[index]), 'cell_area_2d_um2': float(cell_2d.area[index]), 'cytoplasm_area_2d_um2': float(audit_2d.cytoplasm_measure[index]), 'raw_log_ncr_2d': float(audit_2d.raw_log_ncr[index]), 'valid_ncr_2d': bool(audit_2d.valid[index]), 'ncr_reason_2d': str(audit_2d.reason[index]), 'nucleus_volume_3d_um3': float(nucleus_3d.volume[index]), 'cell_volume_3d_um3': float(cell_3d.volume[index]), 'cytoplasm_volume_3d_um3': float(audit_3d.cytoplasm_measure[index]), 'raw_log_ncr_3d': float(audit_3d.raw_log_ncr[index]), 'valid_ncr_3d': bool(audit_3d.valid[index]), 'ncr_reason_3d': str(audit_3d.reason[index])})
    _atomic_parquet(ncr_output, ncr_rows)
    return {'anchors': count, 'valid_ncr_2d': int(audit_2d.valid.sum()), 'valid_ncr_3d': int(audit_3d.valid.sum())}

def _build_or_reuse_shard(task: tuple[int, str, str, float, float, float]) -> dict[str, Any]:
    index, source_text, data_root_text, patch_width_um, patch_height_um, section_thickness_um = task
    source = Path(source_text)
    data_root = Path(data_root_text)
    outputs = (data_root / '02_nucleus_tokens' / f'nucleus_tokens_{index:04d}.parquet', data_root / '03_cell_tokens' / f'cell_tokens_{index:04d}.parquet', data_root / '04_ncr_features' / f'ncr_{index:04d}.parquet')
    shard_manifest = data_root / '06_qc' / 'feature_shards' / f'shard_{index:04d}.json'
    source_sha256 = _sha256(source)
    if shard_manifest.is_file() and all((path.is_file() for path in outputs)):
        previous = json.loads(shard_manifest.read_text(encoding='utf-8'))
        expected_hashes = previous.get('output_sha256', [])
        if previous.get('status') == 'PASS' and previous.get('source_sha256') == source_sha256 and (len(expected_hashes) == len(outputs)) and all((_sha256(path) == digest for path, digest in zip(outputs, expected_hashes))):
            return previous
    result = build_one_shard(source, nucleus_output=outputs[0], cell_output=outputs[1], ncr_output=outputs[2], patch_width_um=patch_width_um, patch_height_um=patch_height_um, section_thickness_um=section_thickness_um)
    record = {'status': 'PASS', 'source': str(source), 'source_sha256': source_sha256, 'outputs': [str(path) for path in outputs], 'output_sha256': [_sha256(path) for path in outputs], **result}
    _atomic_json(shard_manifest, record)
    return record

def build_feature_cache(geometry_dir: str | Path, data_root: str | Path, *, patch_width_um: float, patch_height_um: float, section_thickness_um: float=5.0, workers: int=1) -> dict[str, Any]:
    geometry_dir = Path(geometry_dir)
    data_root = Path(data_root)
    sources = sorted(geometry_dir.glob('dual_geometry_*.parquet'))
    if not sources:
        raise FileNotFoundError(f'no geometry shards in {geometry_dir}')
    if workers < 1:
        raise ValueError('workers must be positive')
    totals = {'anchors': 0, 'valid_ncr_2d': 0, 'valid_ncr_3d': 0}
    tasks = [(index, str(source), str(data_root), float(patch_width_um), float(patch_height_um), float(section_thickness_um)) for index, source in enumerate(sources)]
    if workers == 1:
        output_records = [_build_or_reuse_shard(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            output_records = list(executor.map(_build_or_reuse_shard, tasks))
    for record in output_records:
        for key in totals:
            totals[key] += int(record[key])
    manifest = {'status': 'PASS', 'token_dim': len(TOKEN_COLUMNS), 'token_columns': list(TOKEN_COLUMNS), 'section_thickness_um': section_thickness_um, 'workers': workers, **totals, 'shards': output_records}
    _atomic_json(data_root / '06_qc' / 'feature_cache_manifest.json', manifest)
    return manifest

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--patch-width-um', type=float, required=True)
    parser.add_argument('--patch-height-um', type=float, required=True)
    parser.add_argument('--section-thickness-um', type=float, default=5.0)
    parser.add_argument('--workers', type=int, default=1)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    result = build_feature_cache(args.geometry_dir, args.data_root, patch_width_um=args.patch_width_um, patch_height_um=args.patch_height_um, section_thickness_um=args.section_thickness_um, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
