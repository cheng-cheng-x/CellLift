from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import importlib.util
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
import re
import sys
from typing import Any, Iterator, Mapping
import numpy as np
from celllift.runtime import torch
from .io_utils import atomic_json, sha256
from .protocol import PROTOCOL_ID, require_test_gate
ROOT = Path(__file__).resolve().parent
conditional_geometry_ROOT = ROOT.parent / 'conditional_geometry'
TOKEN_3D_COLUMNS = ('log_volume', 'log_a_b', 'log_b_c', 'long_axis_z_squared')

def _rows(path: Path, **kwargs) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None, **kwargs).to_pylist()

def _suffix(path: Path) -> int:
    match = re.search('(\\d+)$', path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))

def _shards(root: Path, folder: str) -> dict[int, Path]:
    values = {_suffix(path): path for path in sorted((root / folder).glob('*.parquet'))}
    if not values:
        raise FileNotFoundError(root / folder)
    return values

def _vector(array, width: int) -> np.ndarray:
    combined = array.combine_chunks()
    offsets = np.asarray(combined.offsets.to_numpy(zero_copy_only=False), np.int64)
    if len(offsets) != len(combined) + 1 or np.any(np.diff(offsets) != width):
        raise RuntimeError(f'fixed list width changed from {width}')
    return np.asarray(combined.values.to_numpy(zero_copy_only=False), np.float32).reshape(-1, width)

def _read_rays(path: Path, test_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['graph_id', 'anchor_id', 'object_type', 'nucleus_rays_um'], filters=[('object_type', '=', 'nucleus'), ('graph_id', 'in', test_ids)], partitioning=None)
    graph = np.asarray(table['graph_id'].to_pylist(), object)
    anchor = np.asarray(table['anchor_id'].to_numpy(zero_copy_only=False))
    rays = _vector(table['nucleus_rays_um'], 36)
    order = np.lexsort((anchor, graph))
    return (graph[order], anchor[order], rays[order])

def _load_probe(checkpoint: Path, device: torch.device):
    if str(conditional_geometry_ROOT) not in sys.path:
        sys.path.insert(0, str(conditional_geometry_ROOT))
    module_path = conditional_geometry_ROOT / 'models' / 'mask_probe.py'
    specification = importlib.util.spec_from_file_location('conditional_geometry_mask_probe_official', module_path)
    if specification is None or specification.loader is None:
        raise ImportError(f'cannot load frozen mask probe definition: {module_path}')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    MaskOnly3DProbe = module.MaskOnly3DProbe
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MaskOnly3DProbe(256, 0.1).to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    return model

def _test_context(geometry: Mapping[int, Path], test_ids: list[str], graph_index: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(len(graph_index), np.int64)
    total = np.zeros((len(graph_index), 36), np.float64)
    total2 = np.zeros_like(total)
    for path in geometry.values():
        graph, _, rays = _read_rays(path, test_ids)
        indices = np.asarray([graph_index[str(value)] for value in graph], np.int64)
        np.add.at(count, indices, 1)
        np.add.at(total, indices, rays)
        np.add.at(total2, indices, np.square(rays))
    if np.any(count <= 0):
        raise RuntimeError('one or more official TEST graphs have no mask rays')
    mean = total / count[:, None]
    std = np.sqrt(np.maximum(total2 / count[:, None] - np.square(mean), 0.0))
    raw = np.concatenate((mean, std, np.log1p(count)[:, None]), axis=1).astype(np.float32)
    return (raw, count)

def _fixed_list(values: np.ndarray, width: int):
    import pyarrow as pa
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)

def build_official_fold_tokens(cfg: Mapping[str, Any], fold: int, *, device: str='cuda') -> dict[str, Any]:
    gate = require_test_gate(cfg['paths']['result_root'])
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested for official residual cache but unavailable')
    data_root = Path(cfg['paths']['data_root'])
    destination = data_root / '04_residual3d_tokens' / 'official_test' / f'fold_{fold:02d}'
    completed = destination / 'manifest.json'
    if completed.is_file():
        value = json.loads(completed.read_text(encoding='utf-8'))
        if value.get('status') == 'PASS' and value.get('protocol_id') == PROTOCOL_ID:
            gate_path = Path(cfg['paths']['result_root']) / 'selection_frozen.json'
            if value.get('official_test_gate_sha256') != sha256(gate_path):
                raise RuntimeError(f'official token gate checksum mismatch: {completed}')
            for file in value.get('files', []):
                path = Path(file['path'])
                if not path.is_file() or sha256(path) != file.get('sha256'):
                    raise RuntimeError(f'official token shard checksum mismatch: {path}')
            for name in ('probe', 'statistics', 'residual_manifest'):
                path = Path(value[name])
                if not path.is_file() or sha256(path) != value.get(f'{name}_sha256'):
                    raise RuntimeError(f'official token {name} checksum mismatch: {path}')
            return {**value, 'reused': True}
    model_root = Path(cfg['paths']['model_input_root'])
    graph_rows = _rows(model_root / '03_graph_cache' / 'graph_index.parquet', columns=['graph_id', 'split'], filters=[('split', 'in', ['TEST', 'test', 'Test'])])
    if any((str(row['split']).lower() != 'test' for row in graph_rows)):
        raise RuntimeError('non-TEST row crossed official token predicate')
    graph_ids = sorted((str(row['graph_id']) for row in graph_rows))
    graph_index = {graph: index for index, graph in enumerate(graph_ids)}
    set_encoding = Path(cfg['paths']['set_encoding_data_root'])
    nucleus, cell, ncr, geometry = (_shards(set_encoding, '02_nucleus_tokens'), _shards(set_encoding, '03_cell_tokens'), _shards(set_encoding, '04_ncr_features'), _shards(set_encoding, '01_dual_geometry'))
    if not nucleus.keys() == cell.keys() == ncr.keys() == geometry.keys():
        raise RuntimeError('official TEST token shard suffixes differ')
    mask_conditioning_data = Path(cfg['paths']['mask_conditioning_data_root'])
    statistics_path = mask_conditioning_data / 'prepared' / f'fold_{fold:02d}' / 'statistics.json'
    residual_manifest_path = mask_conditioning_data / 'residuals' / f'fold_{fold:02d}' / 'manifest.json'
    statistics = json.loads(statistics_path.read_text(encoding='utf-8'))
    residual_manifest = json.loads(residual_manifest_path.read_text(encoding='utf-8'))
    if residual_manifest.get('status') != 'PASS':
        raise RuntimeError(f'fold residual manifest is not PASS: {residual_manifest_path}')
    public_root = Path(cfg['paths']['result_root']).parents[1]
    probe_path = public_root / 'set_encoder_mask_conditioning_mask_input' / str(cfg['dataset']) / 'probe' / f'fold_{fold:02d}' / 'final.pt'
    probe = _load_probe(probe_path, target)
    context_raw, graph_count = _test_context(geometry, graph_ids, graph_index)
    context = ((context_raw - np.asarray(statistics['context_mean'], np.float32)) / np.asarray(statistics['context_std'], np.float32)).astype(np.float32)
    context_tensor = torch.as_tensor(context, device=target)
    ray_mean, ray_std = (np.asarray(statistics['ray_mean'], np.float32), np.asarray(statistics['ray_std'], np.float32))
    target_mean, target_std = (np.asarray(statistics['target_mean'], np.float32), np.asarray(statistics['target_std'], np.float32))
    residual_mean = np.asarray(residual_manifest['residual_mean_training_only'], np.float32)
    residual_std = np.asarray(residual_manifest['residual_std_training_only'], np.float32)
    ncr_median = float(statistics['ncr3d_median'])
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination.mkdir(parents=True, exist_ok=True)
    files, total_rows = ([], 0)
    with torch.no_grad():
        for shard in sorted(nucleus):
            token_columns = ['graph_id', 'anchor_id', *TOKEN_3D_COLUMNS]
            graph_filter = [('graph_id', 'in', graph_ids)]
            nt = pq.read_table(nucleus[shard], columns=token_columns, filters=graph_filter, partitioning=None)
            ct = pq.read_table(cell[shard], columns=token_columns, filters=graph_filter, partitioning=None)
            rt = pq.read_table(ncr[shard], columns=['graph_id', 'anchor_id', 'raw_log_ncr_3d', 'valid_ncr_3d'], filters=graph_filter, partitioning=None)
            ng = np.asarray(nt['graph_id'].to_pylist(), object)
            na = np.asarray(nt['anchor_id'].to_numpy(zero_copy_only=False))
            if not len(ng):
                continue
            cg = np.asarray(ct['graph_id'].to_pylist(), object)
            ca = np.asarray(ct['anchor_id'].to_numpy(zero_copy_only=False))
            rg = np.asarray(rt['graph_id'].to_pylist(), object)
            ra = np.asarray(rt['anchor_id'].to_numpy(zero_copy_only=False))
            gg, ga, rays = _read_rays(geometry[shard], graph_ids)
            if not (np.array_equal(ng, cg) and np.array_equal(ng, rg) and np.array_equal(ng, gg) and np.array_equal(na, ca) and np.array_equal(na, ra) and np.array_equal(na, ga)):
                raise RuntimeError(f'official paired anchor order mismatch in shard {shard}')
            nucleus3d = np.column_stack([np.asarray(nt[name].to_numpy(), np.float32) for name in TOKEN_3D_COLUMNS])
            cell3d = np.column_stack([np.asarray(ct[name].to_numpy(), np.float32) for name in TOKEN_3D_COLUMNS])
            valid = np.asarray(rt['valid_ncr_3d'].to_numpy(zero_copy_only=False), bool)
            raw_ncr = np.asarray(rt['raw_log_ncr_3d'].to_numpy(zero_copy_only=False), np.float32)
            target_raw = np.concatenate((nucleus3d, cell3d, np.where(valid, raw_ncr, ncr_median)[:, None]), axis=1)
            target_normalized = ((target_raw - target_mean) / target_std).astype(np.float32)
            mask = ((rays - ray_mean) / ray_std).astype(np.float32)
            graph_indices = np.asarray([graph_index[str(value)] for value in ng], np.int64)
            predictions = []
            for start in range(0, len(mask), 16384):
                stop = min(len(mask), start + 16384)
                predictions.append(probe(torch.as_tensor(mask[start:stop], device=target), context_tensor[graph_indices[start:stop]]).float().cpu().numpy())
            prediction = np.concatenate(predictions)
            residual = ((target_normalized - prediction - residual_mean) / residual_std).astype(np.float32)
            target_normalized[~valid, 8] = 0.0
            residual[~valid, 8] = 0.0
            raw_n = np.concatenate((target_normalized[:, :4], target_normalized[:, 8:9]), axis=1)
            raw_c = np.concatenate((target_normalized[:, 4:8], target_normalized[:, 8:9]), axis=1)
            residual_n = np.concatenate((residual[:, :4], residual[:, 8:9]), axis=1)
            residual_c = np.concatenate((residual[:, 4:8], residual[:, 8:9]), axis=1)
            if not all((np.isfinite(value).all() for value in (mask, raw_n, raw_c, residual_n, residual_c))):
                raise RuntimeError('non-finite official modal token')
            table = pa.table({'graph_id': pa.array(ng), 'anchor_id': pa.array(na), 'mask2d': _fixed_list(mask, 36), 'raw_nucleus3d': _fixed_list(raw_n, 5), 'raw_cell3d': _fixed_list(raw_c, 5), 'residual_nucleus3d': _fixed_list(residual_n, 5), 'residual_cell3d': _fixed_list(residual_c, 5), 'valid_ncr3d': pa.array(valid)})
            final = destination / f'tokens_{shard:04d}.parquet'
            temporary = final.with_name(f'.{final.name}.tmp.{os.getpid()}')
            pq.write_table(table, temporary, compression='zstd', row_group_size=100000)
            os.replace(temporary, final)
            files.append({'shard': shard, 'rows': len(table), 'path': str(final), 'sha256': sha256(final)})
            total_rows += len(table)
    expected_rows = int(sum(graph_count))
    if total_rows != expected_rows:
        raise RuntimeError(f'official anchor coverage mismatch: {total_rows} != {expected_rows}')
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': fold, 'official_test_gate_sha256': sha256(Path(cfg['paths']['result_root']) / 'selection_frozen.json'), 'graphs': len(graph_ids), 'anchors': total_rows, 'files': files, 'probe': str(probe_path), 'probe_sha256': sha256(probe_path), 'statistics': str(statistics_path), 'statistics_sha256': sha256(statistics_path), 'residual_manifest': str(residual_manifest_path), 'residual_manifest_sha256': sha256(residual_manifest_path), 'labels_read': False}
    atomic_json(completed, payload)
    return payload
