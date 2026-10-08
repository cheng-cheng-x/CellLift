from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import gzip
from celllift.runtime import json
import os
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .utils import atomic_parquet, read_parquet_rows, stable_shard
_STAINS = np.asarray([[0.65, 0.07, 0.27], [0.7, 0.99, 0.57], [0.29, 0.11, 0.78]], dtype=np.float32)
_STAINS_INVERSE_TRANSPOSE = np.linalg.inv(_STAINS).T

def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}.tmp.{os.getpid()}.npy')
    with temporary.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)

def _atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as handle:
        json.dump(value, handle, separators=(',', ':'), sort_keys=True)
    os.replace(temporary, path)

def _sorted_linear_percentiles(values: np.ndarray, quantiles: tuple[float, float]) -> tuple[float, float]:
    ordered = np.sort(values, axis=None)
    outputs: list[float] = []
    for quantile in quantiles:
        position = quantile * (len(ordered) - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        fraction = position - lower
        outputs.append(float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction))
    return (outputs[0], outputs[1])

def _scale_channel(channel: np.ndarray, *, sorted_percentile: bool=False) -> np.ndarray:
    finite = channel[np.isfinite(channel)]
    if not len(finite):
        raise RuntimeError('stain channel contains no finite values')
    low, high = _sorted_linear_percentiles(finite, (0.01, 0.99)) if sorted_percentile else np.percentile(finite, (1.0, 99.0))
    if high <= low:
        return np.zeros(channel.shape, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)

def stain_inputs(rgb: np.ndarray, *, sorted_percentile: bool=False) -> tuple[np.ndarray, np.ndarray]:
    optical_density = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    concentrations = optical_density @ _STAINS_INVERSE_TRANSPOSE
    hematoxylin = _scale_channel(concentrations[..., 0], sorted_percentile=sorted_percentile)
    eosin = _scale_channel(concentrations[..., 1], sorted_percentile=sorted_percentile)
    cell_input = np.zeros((*hematoxylin.shape, 3), dtype=np.float32)
    cell_input[..., 0] = eosin
    cell_input[..., 1] = hematoxylin
    return (hematoxylin, cell_input)

def canonical_relabel(mask: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask, dtype=np.int64)
    maximum = int(labels.max(initial=0))
    if maximum == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=maximum + 1)
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=maximum + 1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=maximum + 1)
    present = np.flatnonzero(counts[1:]) + 1
    order = sorted(((float(sum_y[item] / counts[item]), float(sum_x[item] / counts[item]), int(item)) for item in present))
    lookup = np.zeros(maximum + 1, dtype=np.int32)
    for new_id, (_, _, old_id) in enumerate(order, start=1):
        lookup[old_id] = new_id
    return lookup[labels]

def instance_payload(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    labels = np.asarray(mask, dtype=np.int64)
    maximum = int(labels.max(initial=0))
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=maximum + 1)
    yy, xx = np.indices(labels.shape, dtype=np.int64)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=maximum + 1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=maximum + 1)
    min_x = np.full(maximum + 1, width, dtype=np.int64)
    min_y = np.full(maximum + 1, height, dtype=np.int64)
    max_x = np.full(maximum + 1, -1, dtype=np.int64)
    max_y = np.full(maximum + 1, -1, dtype=np.int64)
    foreground = flat > 0
    foreground_labels = flat[foreground]
    np.minimum.at(min_x, foreground_labels, xx.ravel()[foreground])
    np.minimum.at(min_y, foreground_labels, yy.ravel()[foreground])
    np.maximum.at(max_x, foreground_labels, xx.ravel()[foreground])
    np.maximum.at(max_y, foreground_labels, yy.ravel()[foreground])
    instances: list[dict[str, Any]] = []
    for instance_id in range(1, maximum + 1):
        if not counts[instance_id]:
            continue
        x0, x1 = (int(min_x[instance_id]), int(max_x[instance_id]) + 1)
        y0, y1 = (int(min_y[instance_id]), int(max_y[instance_id]) + 1)
        instances.append({'instance_id': instance_id, 'area_px': int(counts[instance_id]), 'centroid_xy': [float(sum_x[instance_id] / counts[instance_id]), float(sum_y[instance_id] / counts[instance_id])], 'bbox_xyxy': [x0, y0, x1, y1], 'border_flag': bool(x0 == 0 or y0 == 0 or x1 == width or (y1 == height))})
    return {'shape_yx': [height, width], 'instances': instances}

def pair_instances(nucleus_mask: np.ndarray, cell_mask: np.ndarray) -> dict[str, Any]:
    nucleus_to_cell: dict[int, int | None] = {}
    for nucleus_id in range(1, int(nucleus_mask.max()) + 1):
        overlapping = cell_mask[nucleus_mask == nucleus_id]
        overlapping = overlapping[overlapping > 0]
        if not len(overlapping):
            nucleus_to_cell[nucleus_id] = None
        else:
            counts = np.bincount(overlapping)
            nucleus_to_cell[nucleus_id] = int(np.argmax(counts))
    cell_nuclei: dict[int, list[int]] = {cell_id: [] for cell_id in range(1, int(cell_mask.max()) + 1)}
    for nucleus_id, cell_id in nucleus_to_cell.items():
        if cell_id:
            cell_nuclei.setdefault(cell_id, []).append(nucleus_id)
    nucleus_relations: list[dict[str, Any]] = []
    accepted = 0
    for nucleus_id, cell_id in nucleus_to_cell.items():
        assigned = len(cell_nuclei.get(int(cell_id or 0), [])) if cell_id else 0
        status = 'accepted_one_to_one' if cell_id and assigned == 1 else 'unmatched_nucleus' if not cell_id else 'cell_has_multiple_nuclei'
        accepted += int(status == 'accepted_one_to_one')
        nucleus_relations.append({'nucleus_id': nucleus_id, 'cell_id': cell_id, 'pair_status': status, 'cell_assigned_nucleus_count': assigned})
    cell_relations = []
    for cell_id, nuclei in sorted(cell_nuclei.items()):
        status = 'accepted_one_to_one' if len(nuclei) == 1 else 'nucleus_free' if not nuclei else 'multiple_nuclei'
        cell_relations.append({'cell_id': cell_id, 'assigned_nucleus_ids': nuclei, 'assigned_nucleus_count': len(nuclei), 'cell_status': status, 'cell_valid_for_matching': len(nuclei) == 1})
    nucleus_count = int(nucleus_mask.max())
    return {'nucleus_relations': nucleus_relations, 'cell_relations': cell_relations, 'summary': {'nucleus_count': nucleus_count, 'cell_count': int(cell_mask.max()), 'accepted_pair_count': accepted, 'nucleus_matched_fraction': accepted / nucleus_count if nucleus_count else 0.0}}

def _as_list(value: Any, count: int) -> list[np.ndarray]:
    if isinstance(value, list):
        return [np.asarray(item) for item in value]
    array = np.asarray(value)
    return [array] if count == 1 and array.ndim == 2 else [np.asarray(item) for item in array]

def _vectorized_get_masks_torch(pt: Any, inds: tuple[np.ndarray, ...], shape0: tuple[int, ...], rpad: int=20, max_size_fraction: float=0.4) -> np.ndarray:
    import fastremap
    import torch
    from cellpose.dynamics import max_pool_nd
    ndim = len(shape0)
    device = pt.device
    pt += rpad
    pt = torch.clamp(pt, min=0)
    for axis in range(len(pt)):
        pt[axis] = torch.clamp(pt[axis], max=shape0[axis] + rpad - 1)
    shape = tuple(np.asarray(shape0) + 2 * rpad)
    coo = torch.sparse_coo_tensor(pt, torch.ones(pt.shape[1], device=device, dtype=torch.int), shape)
    h1 = coo.to_dense()
    del coo
    hmax1 = max_pool_nd(h1.unsqueeze(0), kernel_size=5).squeeze()
    seeds1 = torch.nonzero((h1 - hmax1 > -1e-06) * (h1 > 10))
    del hmax1
    if len(seeds1) == 0:
        return np.zeros(shape0, dtype='uint16')
    npts = h1[tuple(seeds1.T)]
    seeds1 = seeds1[npts.argsort()]
    n_seeds = len(seeds1)
    offsets = torch.arange(-5, 6, device=device)
    if ndim == 2:
        yy = seeds1[:, 0, None, None] + offsets[None, :, None]
        xx = seeds1[:, 1, None, None] + offsets[None, None, :]
        h_slc = h1[yy, xx]
        seed_masks = torch.zeros((n_seeds, 11, 11), device=device)
        seed_masks[:, 5, 5] = 1
    else:
        zz = seeds1[:, 0, None, None, None] + offsets[None, :, None, None]
        yy = seeds1[:, 1, None, None, None] + offsets[None, None, :, None]
        xx = seeds1[:, 2, None, None, None] + offsets[None, None, None, :]
        h_slc = h1[zz, yy, xx]
        seed_masks = torch.zeros((n_seeds, 11, 11, 11), device=device)
        seed_masks[:, 5, 5, 5] = 1
    del h1
    for _ in range(5):
        seed_masks = max_pool_nd(seed_masks, kernel_size=3)
        seed_masks *= h_slc > 2
    del h_slc
    nonzero = torch.nonzero(seed_masks)
    del seed_masks
    seed_index = nonzero[:, 0]
    coordinates = nonzero[:, 1:] + seeds1[seed_index] - 5
    dtype = torch.int32 if n_seeds < 2 ** 16 else torch.int64
    flat_index = coordinates[:, 0]
    for axis in range(1, ndim):
        flat_index = flat_index * shape[axis] + coordinates[:, axis]
    labels = (seed_index + 1).to(dtype)
    flat_mask = torch.zeros(int(np.prod(shape)), dtype=dtype, device=device)
    flat_mask.scatter_reduce_(0, flat_index, labels, reduce='amax', include_self=True)
    dense = flat_mask.reshape(shape)
    mapped = dense[tuple(pt)].cpu().numpy()
    output_dtype = 'uint16' if n_seeds < 2 ** 16 else 'uint32'
    output = np.zeros(shape0, dtype=output_dtype)
    output[inds] = mapped
    unique, counts = fastremap.unique(output, return_counts=True)
    too_large = unique[counts > np.prod(shape0) * max_size_fraction]
    if len(too_large) > 0 and (len(too_large) > 1 or too_large[0] != 0):
        output = fastremap.mask(output, too_large)
    fastremap.renumber(output, in_place=True)
    return output.reshape(tuple(shape0))

def enable_vectorized_cellpose_masks() -> None:
    from cellpose import dynamics
    dynamics.get_masks_torch = _vectorized_get_masks_torch

def _load_models(*, nucleus_only: bool=False, require_cuda: bool=False, vectorized_masks: bool=False) -> tuple[Any, Any | None]:
    from cellpose import models
    if vectorized_masks:
        enable_vectorized_cellpose_masks()
    nucleus = models.Cellpose(gpu=True, model_type='nuclei')
    if require_cuda and (not str(getattr(nucleus, 'device', '')).startswith('cuda')):
        raise RuntimeError('Cellpose nuclei model did not acquire CUDA; refusing silent CPU fallback')
    cell = None if nucleus_only else models.Cellpose(gpu=True, model_type='cyto3')
    if require_cuda and cell is not None and (not str(getattr(cell, 'device', '')).startswith('cuda')):
        raise RuntimeError('Cellpose cyto model did not acquire CUDA; refusing silent CPU fallback')
    return (nucleus, cell)

def _eval(model: Any, images: list[np.ndarray], *, diameter: float, channels: list[int], batch_size: int) -> list[np.ndarray]:
    result = model.eval(images, diameter=diameter, channels=channels, flow_threshold=0.8, cellprob_threshold=-1.0, normalize=False, batch_size=batch_size)
    return _as_list(result[0], len(images))

def _complete_existing(row: dict[str, Any], *, nucleus_only: bool=False) -> dict[str, Any] | None:
    required = ('nucleus_mask_path', 'nucleus_instances_path') if nucleus_only else ('nucleus_mask_path', 'cell_mask_path', 'nucleus_instances_path', 'cell_instances_path', 'pairs_path')
    paths = [Path(row[key]) for key in required]
    if not all((path.is_file() for path in paths)):
        return None
    nucleus = np.load(paths[0], mmap_mode='r')
    expected_shape = (int(row['output_height_px']), int(row['output_width_px']))
    if nucleus.shape != expected_shape:
        raise RuntimeError(f"existing segmentation shape mismatch: {row['patch_id']}")
    if nucleus_only:
        with gzip.open(Path(row['nucleus_instances_path']), 'rt', encoding='utf-8') as handle:
            nuclei = json.load(handle)
        return {'nucleus_count': len(nuclei['instances']), 'cell_count': 0, 'accepted_pair_count': 0, 'nucleus_matched_fraction': None}
    cell = np.load(Path(row['cell_mask_path']), mmap_mode='r')
    if nucleus.shape != cell.shape:
        raise RuntimeError(f"existing nucleus/cell shape mismatch: {row['patch_id']}")
    with gzip.open(Path(row['pairs_path']), 'rt', encoding='utf-8') as handle:
        pairs = json.load(handle)
    return dict(pairs['summary'])

def _write_segmentation_result(row: dict[str, Any], nucleus_mask: np.ndarray, cell_mask: np.ndarray | None, *, nucleus_only: bool) -> dict[str, Any]:
    try:
        nucleus_mask = canonical_relabel(nucleus_mask)
        if not nucleus_only:
            if cell_mask is None:
                raise RuntimeError('cell mask is missing in dual-input mode')
            cell_mask = canonical_relabel(cell_mask)
        if nucleus_mask.shape != (int(row['output_height_px']), int(row['output_width_px'])):
            raise RuntimeError(f'unexpected nucleus mask shape: {nucleus_mask.shape}')
        nuclei = instance_payload(nucleus_mask)
        _atomic_npy(Path(row['nucleus_mask_path']), nucleus_mask)
        _atomic_gzip_json(Path(row['nucleus_instances_path']), nuclei)
        if nucleus_only:
            summary = {'nucleus_count': len(nuclei['instances']), 'cell_count': 0, 'accepted_pair_count': 0, 'nucleus_matched_fraction': None}
            status = {'patch_id': row['patch_id'], 'status': 'complete', 'resumed': False, 'input_mode': 'nucleus_only', **summary}
            if '_trace_started' in row:
                print(json.dumps({'trace': 'write_complete', 'patch_id': row['patch_id'], 'seconds_total': time.perf_counter() - row['_trace_started']}), flush=True)
            return status
        cells = instance_payload(cell_mask)
        pairs = pair_instances(nucleus_mask, cell_mask)
        _atomic_npy(Path(row['cell_mask_path']), cell_mask)
        _atomic_gzip_json(Path(row['cell_instances_path']), cells)
        _atomic_gzip_json(Path(row['pairs_path']), pairs)
        return {'patch_id': row['patch_id'], 'status': 'complete', 'resumed': False, **pairs['summary']}
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        detail = traceback.format_exc() if '_trace_started' in row else None
        print(json.dumps({'trace': 'patch_failed', 'patch_id': row['patch_id'], 'failure_reason': failure, 'traceback': detail}), flush=True)
        return {'patch_id': row['patch_id'], 'status': 'failed', 'failure_reason': failure}

def run_segment(cfg: dict[str, Any], dataset: str, shard_id: int, num_shards: int, *, pilot: bool=False) -> dict[str, Any]:
    from PIL import Image
    data_root = Path(cfg['paths']['data_root'])
    manifest = data_root / '00_manifest' / ('pilot_manifest.parquet' if pilot else 'patch_manifest.parquet')
    rows = [row for row in read_parquet_rows(manifest) if stable_shard(str(row['patch_id']), num_shards) == shard_id]
    nucleus_only = cfg.get('input', {}).get('mode') == 'nucleus_only'
    sorted_percentile = bool(cfg.get('cellpose', {}).get('sorted_percentile', False))
    batch_size = int(cfg['cellpose']['pilot_batch_size' if pilot else 'batch_size'])
    chunk_size = int(cfg['cellpose'].get('image_chunk_size', batch_size))
    nucleus_model, cell_model = _load_models(nucleus_only=nucleus_only, require_cuda=bool(cfg.get('cellpose', {}).get('require_cuda', False)), vectorized_masks=bool(cfg.get('cellpose', {}).get('vectorized_get_masks', False)))
    trace_remaining = int(cfg.get('runtime', {}).get('trace_first_patches', 0))
    postprocess_workers = max(1, int(cfg.get('runtime', {}).get('postprocess_workers', 1)))
    max_pending_writes = max(postprocess_workers, int(cfg.get('runtime', {}).get('max_pending_writes', postprocess_workers * 2)))
    executor = ThreadPoolExecutor(max_workers=postprocess_workers) if postprocess_workers > 1 else None
    pending_writes: deque[Future[dict[str, Any]]] = deque()
    statuses: list[dict[str, Any]] = []
    for offset in range(0, len(rows), chunk_size):
        batch_rows = rows[offset:offset + chunk_size]
        pending: list[dict[str, Any]] = []
        nucleus_inputs: list[np.ndarray] = []
        cell_inputs: list[np.ndarray] = []
        for row in batch_rows:
            try:
                resumed = _complete_existing(row, nucleus_only=nucleus_only)
                if resumed is not None:
                    statuses.append({'patch_id': row['patch_id'], 'status': 'complete', 'resumed': True, 'input_mode': 'nucleus_only' if nucleus_only else 'dual', **resumed})
                    continue
                with Image.open(row['rgb_path']) as image:
                    rgb = np.asarray(image.convert('RGB'))
                trace_this = trace_remaining > 0
                trace_started = time.perf_counter()
                if trace_this:
                    print(json.dumps({'trace': 'start', 'patch_id': row['patch_id']}), flush=True)
                nucleus_input, cell_input = stain_inputs(rgb, sorted_percentile=sorted_percentile)
                if trace_this:
                    print(json.dumps({'trace': 'stain_complete', 'patch_id': row['patch_id'], 'seconds': time.perf_counter() - trace_started}), flush=True)
                    row['_trace_started'] = trace_started
                    trace_remaining -= 1
                pending.append(row)
                nucleus_inputs.append(nucleus_input)
                if not nucleus_only:
                    cell_inputs.append(cell_input)
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
                detail = traceback.format_exc() if '_trace_started' in row else None
                print(json.dumps({'trace': 'patch_failed', 'patch_id': row['patch_id'], 'failure_reason': failure, 'traceback': detail}), flush=True)
                statuses.append({'patch_id': row['patch_id'], 'status': 'failed', 'failure_reason': failure})
        if not pending:
            continue
        try:
            nucleus_masks = _eval(nucleus_model, nucleus_inputs, diameter=17.0, channels=[0, 0], batch_size=batch_size)
            for traced_row in pending:
                if '_trace_started' in traced_row:
                    print(json.dumps({'trace': 'eval_complete', 'patch_id': traced_row['patch_id'], 'seconds_total': time.perf_counter() - traced_row['_trace_started']}), flush=True)
            cell_masks = None if nucleus_only else _eval(cell_model, cell_inputs, diameter=20.0, channels=[1, 2], batch_size=batch_size)
        except Exception as exc:
            for row in pending:
                failure = f'CellposeError: {exc}'
                print(json.dumps({'trace': 'patch_failed', 'patch_id': row['patch_id'], 'failure_reason': failure}), flush=True)
                statuses.append({'patch_id': row['patch_id'], 'status': 'failed', 'failure_reason': failure})
            continue
        paired_masks = zip(pending, nucleus_masks, [None] * len(pending) if nucleus_only else cell_masks)
        for row, nucleus_mask, cell_mask in paired_masks:
            if executor is None:
                statuses.append(_write_segmentation_result(row, nucleus_mask, cell_mask, nucleus_only=nucleus_only))
            else:
                pending_writes.append(executor.submit(_write_segmentation_result, row, nucleus_mask, cell_mask, nucleus_only=nucleus_only))
                if len(pending_writes) >= max_pending_writes:
                    statuses.append(pending_writes.popleft().result())
    while pending_writes:
        statuses.append(pending_writes.popleft().result())
    if executor is not None:
        executor.shutdown(wait=True)
    tag = 'pilot' if pilot else 'full'
    output = data_root / '00_manifest' / f'segment_{tag}_shard_{shard_id:03d}_of_{num_shards:03d}.parquet'
    atomic_parquet(output, statuses)
    failed = [row for row in statuses if row['status'] != 'complete']
    if failed:
        raise RuntimeError(f'segmentation failed for {len(failed)}/{len(statuses)} patches; see {output}')
    return {'status': 'PASS', 'processed': len(statuses), 'output': str(output)}
