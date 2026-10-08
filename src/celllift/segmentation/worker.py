from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import gzip
import hashlib
import io
from celllift.runtime import json
import math
import os
import platform
import socket
import time
import traceback
from celllift.runtime import ResourcePath as Path
from typing import Any
import cv2
import numpy as np
from celllift.runtime import torch
from cellpose import models
from PIL import Image, ImageDraw
from scipy import ndimage
from celllift.segmentation.common import algorithm_fingerprint, atomic_save_npy, load_config, read_gzip_json, read_json, read_tsv, sha256_file, stable_slot, validate_label_map, worker_config, write_gzip_json, write_json

def output_paths(root: Path, identifier: str, worker_id: int) -> dict[str, Path]:
    shard = f'worker_{worker_id:02d}'
    return {'nucleus_mask': root / '01_nuclei/masks' / shard / f'{identifier}.npy', 'nucleus_instances': root / '01_nuclei/instances' / shard / f'{identifier}.json.gz', 'cell_mask': root / '02_cells/masks' / shard / f'{identifier}.npy', 'cell_instances': root / '02_cells/instances' / shard / f'{identifier}.json.gz', 'pairs': root / '03_pairs/records' / shard / f'{identifier}.json.gz', 'status': root / '04_status/roi' / shard / f'{identifier}.json', 'overlay': root / '05_qc/snapshot_overlays' / f'{identifier}.png'}

def relabel(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.squeeze(np.asarray(mask))
    if array.shape != shape:
        raise RuntimeError(f'unexpected Cellpose mask shape {array.shape}; expected {shape}')
    array = array.astype(np.int64, copy=False)
    labels = np.unique(array)
    labels = labels[labels > 0]
    if not labels.size:
        return np.zeros(shape, dtype=np.int32)
    lookup = np.zeros(int(labels[-1]) + 1, dtype=np.int32)
    lookup[labels] = np.arange(1, labels.size + 1, dtype=np.int32)
    return lookup[array]

def extract_eval_masks(output: Any, shape: tuple[int, int]) -> np.ndarray:
    masks = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(masks, list):
        raw = masks[0]
    else:
        array = np.asarray(masks)
        raw = array if array.ndim == 2 else array[0]
    return relabel(raw, shape)

def preprocess_he(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pixels = rgb.astype(np.float32)
    optical_density = -np.log((pixels + 1.0) / 256.0)
    stain_matrix = np.asarray([[0.65, 0.072], [0.704, 0.99], [0.286, 0.105]], dtype=np.float32)
    concentrations = optical_density.reshape(-1, 3) @ np.linalg.pinv(stain_matrix).T
    concentrations = concentrations.reshape(*rgb.shape[:2], 2)
    hematoxylin = concentrations[:, :, 0]
    eosin = concentrations[:, :, 1]
    tissue = (optical_density.sum(axis=2) > 0.16) & (pixels.mean(axis=2) < 242.0)
    tissue_h = hematoxylin[tissue]
    if tissue_h.size >= 100:
        nucleus_low = float(np.percentile(tissue_h, 1))
        nucleus_high = float(np.percentile(tissue_h, 99))
    else:
        finite_h = hematoxylin[np.isfinite(hematoxylin)]
        nucleus_low = float(np.percentile(finite_h, 1)) if finite_h.size else 0.0
        nucleus_high = float(np.percentile(finite_h, 99)) if finite_h.size else 1.0
    nucleus_image = np.clip((hematoxylin - nucleus_low) / max(nucleus_high - nucleus_low, 1e-06) * 255.0, 0, 255).astype(np.uint8)
    normalized = []
    channel_ranges = []
    for values in (hematoxylin, eosin):
        finite = values[np.isfinite(values)]
        low = float(np.percentile(finite, 1)) if finite.size else 0.0
        high = float(np.percentile(finite, 99)) if finite.size else 1.0
        normalized.append(np.clip((values - low) / max(high - low, 1e-06) * 255.0, 0, 255).astype(np.uint8))
        channel_ranges.append((low, high))
    cell_h, cell_e = normalized
    cell_image = np.stack([cell_e, cell_h, np.zeros_like(cell_h)], axis=2)
    return (nucleus_image, cell_image, {'tissue_fraction': float(tissue.mean()), 'nucleus_h_low': nucleus_low, 'nucleus_h_high': nucleus_high, 'cell_h_low': channel_ranges[0][0], 'cell_h_high': channel_ranges[0][1], 'cell_e_low': channel_ranges[1][0], 'cell_e_high': channel_ranges[1][1]})

def infer_model(model: Any, image: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    output = model.eval([image], channels=[int(value) for value in settings['channels']], diameter=float(settings['diameter_px']), invert=bool(settings['invert']), batch_size=int(settings['batch_size']), flow_threshold=float(settings['flow_threshold']), cellprob_threshold=float(settings['cellprob_threshold']), min_size=int(settings['min_size_px']))
    return extract_eval_masks(output, image.shape[:2])

def describe_model_device(model: Any) -> dict[str, Any]:
    report: dict[str, Any] = {'cellpose_device': str(getattr(model, 'device', 'unknown')), 'cellpose_gpu': bool(getattr(model, 'gpu', False)), 'net_first_param_device': 'unknown'}
    cp_model = getattr(model, 'cp', None)
    net = getattr(cp_model, 'net', None)
    try:
        report['net_first_param_device'] = str(next(net.parameters()).device)
    except Exception as error:
        report['net_first_param_device'] = f'unavailable:{type(error).__name__}'
    return report

def assert_model_on_cuda(kind: str, model: Any) -> dict[str, Any]:
    report = describe_model_device(model)
    if not report['cellpose_gpu']:
        raise RuntimeError(f'Cellpose {kind} model reports gpu=False: {report}')
    if not report['cellpose_device'].startswith('cuda'):
        raise RuntimeError(f'Cellpose {kind} model is not on CUDA: {report}')
    if not report['net_first_param_device'].startswith('cuda'):
        raise RuntimeError(f'Cellpose {kind} net is not on CUDA: {report}')
    return report

def contours_for_local_mask(local: np.ndarray, x0: int, y0: int) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in contour] for contour in contours if len(contour) >= 3]

def instances_from_labels(labels: np.ndarray) -> list[dict[str, Any]]:
    height, width = labels.shape
    objects = ndimage.find_objects(labels)
    output: list[dict[str, Any]] = []
    for instance_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        y0, y1 = (int(y_slice.start), int(y_slice.stop))
        x0, x1 = (int(x_slice.start), int(x_slice.stop))
        local = labels[y0:y1, x0:x1] == instance_id
        ys, xs = np.nonzero(local)
        if not len(xs):
            continue
        output.append({'instance_id': instance_id, 'area_px': int(len(xs)), 'bbox_xyxy': [x0, y0, x1, y1], 'centroid_xy': [float(xs.mean() + x0), float(ys.mean() + y0)], 'border_flag': bool(x0 == 0 or y0 == 0 or x1 == width or (y1 == height)), 'contours_xy': contours_for_local_mask(local, x0, y0)})
    return output

def pair_labels(nuclei: np.ndarray, cells: np.ndarray, min_containment: float) -> dict[str, Any]:
    nucleus_count = int(nuclei.max(initial=0))
    cell_count = int(cells.max(initial=0))
    if nucleus_count and cell_count:
        index = nuclei.reshape(-1).astype(np.int64) * (cell_count + 1) + cells.reshape(-1).astype(np.int64)
        overlap = np.bincount(index, minlength=(nucleus_count + 1) * (cell_count + 1)).reshape(nucleus_count + 1, cell_count + 1)
        nucleus_areas = overlap.sum(axis=1)[1:]
        cell_areas = overlap.sum(axis=0)[1:]
        foreground_overlap = overlap[1:, 1:]
        assigned_cells = foreground_overlap.argmax(axis=1) + 1
        assigned_overlap = foreground_overlap.max(axis=1)
        assigned_cells[assigned_overlap == 0] = 0
        containment = assigned_overlap / np.maximum(nucleus_areas, 1)
        nuclei_per_cell = np.bincount(assigned_cells, minlength=cell_count + 1)
    else:
        nucleus_areas = np.bincount(nuclei.reshape(-1).astype(np.int64), minlength=nucleus_count + 1)[1:]
        cell_areas = np.bincount(cells.reshape(-1).astype(np.int64), minlength=cell_count + 1)[1:]
        assigned_cells = np.zeros(nucleus_count, dtype=np.int64)
        assigned_overlap = np.zeros(nucleus_count, dtype=np.int64)
        containment = np.zeros(nucleus_count, dtype=np.float64)
        nuclei_per_cell = np.zeros(cell_count + 1, dtype=np.int64)
    nucleus_relations = []
    accepted_pair_count = 0
    for index in range(nucleus_count):
        nucleus_id = index + 1
        cell_id = int(assigned_cells[index])
        cell_assignment_count = int(nuclei_per_cell[cell_id]) if cell_id > 0 else 0
        if cell_id == 0:
            pair_status = 'unmatched_nucleus'
        elif cell_assignment_count > 1:
            pair_status = 'ambiguous_multi_nucleus_cell'
        elif float(containment[index]) < min_containment:
            pair_status = 'one_to_one_low_containment'
        else:
            pair_status = 'accepted_one_to_one'
            accepted_pair_count += 1
        nucleus_relations.append({'nucleus_id': nucleus_id, 'cell_id': cell_id, 'overlap_px': int(assigned_overlap[index]), 'nucleus_area_px': int(nucleus_areas[index]), 'cell_area_px': int(cell_areas[cell_id - 1]) if cell_id > 0 else 0, 'nucleus_containment_fraction': float(containment[index]), 'cell_assigned_nucleus_count': cell_assignment_count, 'pair_status': pair_status})
    assigned_lists: list[list[int]] = [[] for _ in range(cell_count + 1)]
    for nucleus_id, cell_id in enumerate(assigned_cells, start=1):
        if int(cell_id) > 0:
            assigned_lists[int(cell_id)].append(nucleus_id)
    cell_relations = []
    for cell_id in range(1, cell_count + 1):
        assigned = assigned_lists[cell_id]
        if not assigned:
            status = 'nucleus_free_cell'
        elif len(assigned) == 1:
            status = 'single_nucleus_cell'
        else:
            status = 'multi_nucleus_cell'
        cell_relations.append({'cell_id': cell_id, 'cell_area_px': int(cell_areas[cell_id - 1]), 'assigned_nucleus_ids': assigned, 'assigned_nucleus_count': len(assigned), 'cell_status': status})
    matched = int((assigned_cells > 0).sum())
    one_cells = int((nuclei_per_cell[1:] == 1).sum())
    multi_cells = int((nuclei_per_cell[1:] > 1).sum())
    cells_with_nucleus = one_cells + multi_cells
    one_area_ratios = [relation['cell_area_px'] / max(relation['nucleus_area_px'], 1) for relation in nucleus_relations if relation['cell_id'] > 0 and relation['cell_assigned_nucleus_count'] == 1]
    return {'summary': {'nucleus_count': nucleus_count, 'cell_count': cell_count, 'matched_nucleus_count': matched, 'unmatched_nucleus_count': nucleus_count - matched, 'nucleus_matched_fraction': float(matched / nucleus_count) if nucleus_count else math.nan, 'cells_with_nucleus_count': cells_with_nucleus, 'cell_without_nucleus_count': cell_count - cells_with_nucleus, 'cells_with_nucleus_fraction': float(cells_with_nucleus / cell_count) if cell_count else math.nan, 'one_nucleus_cell_count': one_cells, 'cells_exactly_one_nucleus_fraction': float(one_cells / cell_count) if cell_count else math.nan, 'multi_nucleus_cell_count': multi_cells, 'cells_multi_nucleus_fraction': float(multi_cells / cell_count) if cell_count else math.nan, 'accepted_pair_count': accepted_pair_count, 'accepted_pair_fraction_of_nuclei': float(accepted_pair_count / nucleus_count) if nucleus_count else math.nan, 'nucleus_containment_median': float(np.median(containment)) if containment.size else math.nan, 'one_to_one_cell_nucleus_area_ratio_median': float(np.median(one_area_ratios)) if one_area_ratios else math.nan}, 'nucleus_relations': nucleus_relations, 'cell_relations': cell_relations}

def label_boundary(labels: np.ndarray) -> np.ndarray:
    positive = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[1:, :] |= positive[1:, :] & (labels[1:, :] != labels[:-1, :])
    boundary[:-1, :] |= positive[:-1, :] & (labels[:-1, :] != labels[1:, :])
    boundary[:, 1:] |= positive[:, 1:] & (labels[:, 1:] != labels[:, :-1])
    boundary[:, :-1] |= positive[:, :-1] & (labels[:, :-1] != labels[:, 1:])
    boundary[0, :] |= positive[0, :]
    boundary[-1, :] |= positive[-1, :]
    boundary[:, 0] |= positive[:, 0]
    boundary[:, -1] |= positive[:, -1]
    return boundary

def save_overlay(rgb: np.ndarray, nuclei: np.ndarray, cells: np.ndarray, pair_summary: dict[str, Any], output: Path) -> None:
    canvas = rgb.copy()
    canvas[label_boundary(cells)] = np.asarray([0, 230, 255], np.uint8)
    canvas[label_boundary(nuclei)] = np.asarray([255, 215, 0], np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1024, 30), fill=(0, 0, 0))
    draw.text((6, 7), f"n={pair_summary['nucleus_count']} c={pair_summary['cell_count']} match={pair_summary['nucleus_matched_fraction']:.3f} accepted={pair_summary['accepted_pair_fraction_of_nuclei']:.3f}", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp.{os.getpid()}')
    image.save(temporary, format='PNG', compress_level=3)
    os.replace(temporary, output)

def valid_existing(paths: dict[str, Path], source_sha: str, algorithm_id: str, width: int, height: int) -> dict[str, Any] | None:
    status = read_json(paths['status'], {})
    if status.get('status') != 'complete':
        return None
    if status.get('source_png_sha256') != source_sha or status.get('algorithm_fingerprint') != algorithm_id:
        return None
    for key in ('nucleus_mask', 'cell_mask'):
        if not validate_label_map(paths[key], width, height):
            return None
        if sha256_file(paths[key]) != status.get(f'{key}_sha256'):
            return None
    for key in ('nucleus_instances', 'cell_instances', 'pairs'):
        if not paths[key].is_file():
            return None
        if sha256_file(paths[key]) != status.get(f'{key}_sha256'):
            return None
    return status

def process_one(row: dict[str, str], cfg: dict[str, Any], worker_id: int, models_by_kind: dict[str, Any], qc_ids: set[str]) -> dict[str, Any]:
    root = Path(cfg['result_root'])
    identifier = row['roi_layer_id']
    width = int(cfg['physical']['width_px'])
    height = int(cfg['physical']['height_px'])
    source = Path(row['png_path'])
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != row['png_sha256']:
        raise RuntimeError(f'source PNG SHA256 mismatch: {identifier}')
    with Image.open(io.BytesIO(source_bytes)) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise RuntimeError(f'source PNG format mismatch: {identifier}')
        if image.size != (width, height):
            raise RuntimeError(f'source PNG size mismatch: {identifier}')
        rgb = np.asarray(image).copy()
    algorithm_id = algorithm_fingerprint(cfg)
    paths = output_paths(root, identifier, worker_id)
    existing = valid_existing(paths, source_sha, algorithm_id, width, height)
    if existing is not None:
        return {**existing, '_reused': True}
    started = time.time()
    nucleus_input, cell_input, preprocessing = preprocess_he(rgb)
    nucleus_labels = infer_model(models_by_kind['nucleus'], nucleus_input, cfg['nucleus'])
    cell_labels = infer_model(models_by_kind['cell'], cell_input, cfg['cell'])
    nucleus_instances = instances_from_labels(nucleus_labels)
    cell_instances = instances_from_labels(cell_labels)
    pairing = pair_labels(nucleus_labels, cell_labels, float(cfg['pairing']['direct_pair_min_nucleus_containment']))
    atomic_save_npy(paths['nucleus_mask'], nucleus_labels)
    atomic_save_npy(paths['cell_mask'], cell_labels)
    coordinate_space = 'registered_native_1024_at_0.46_um_per_px'
    write_gzip_json(paths['nucleus_instances'], {'roi_layer_id': identifier, 'kind': 'nucleus', 'model': cfg['nucleus'], 'coordinate_space': coordinate_space, 'instances': nucleus_instances})
    write_gzip_json(paths['cell_instances'], {'roi_layer_id': identifier, 'kind': 'cell', 'model': cfg['cell'], 'coordinate_space': coordinate_space, 'instances': cell_instances})
    write_gzip_json(paths['pairs'], {'roi_layer_id': identifier, 'coordinate_space': coordinate_space, 'pairing_policy': cfg['pairing'], **pairing})
    if identifier in qc_ids:
        save_overlay(rgb, nucleus_labels, cell_labels, pairing['summary'], paths['overlay'])
    status = {'status': 'complete', 'experiment_id': cfg['experiment_id'], 'roi_layer_id': identifier, 'component_id': row['component_id'], 'track_id': row['track_id'], 'section_id': row['section_id'], 'source_png_path': str(source), 'source_png_sha256': source_sha, 'worker_id': worker_id, 'virtual_slot': stable_slot(identifier, int(cfg['runtime']['virtual_shards'])), 'hostname': socket.gethostname(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''), 'cellpose_version': str(getattr(__import__('cellpose'), 'version', 'unknown')), 'torch_version': torch.__version__, 'actual_model_devices': {kind: describe_model_device(model) for kind, model in models_by_kind.items()}, 'algorithm_fingerprint': algorithm_id, 'nucleus_mask_path': str(paths['nucleus_mask']), 'nucleus_mask_sha256': sha256_file(paths['nucleus_mask']), 'nucleus_instances_path': str(paths['nucleus_instances']), 'nucleus_instances_sha256': sha256_file(paths['nucleus_instances']), 'cell_mask_path': str(paths['cell_mask']), 'cell_mask_sha256': sha256_file(paths['cell_mask']), 'cell_instances_path': str(paths['cell_instances']), 'cell_instances_sha256': sha256_file(paths['cell_instances']), 'pairs_path': str(paths['pairs']), 'pairs_sha256': sha256_file(paths['pairs']), 'nucleus_foreground_fraction': float((nucleus_labels > 0).mean()), 'cell_foreground_fraction': float((cell_labels > 0).mean()), 'preprocessing': preprocessing, **pairing['summary'], 'elapsed_seconds': time.time() - started}
    write_json(paths['status'], status)
    return {**status, '_reused': False}

def run(config_path: Path, worker_id: int, limit: int | None=None, start_index: int=0, stop_index: int | None=None, state_tag: str | None=None) -> dict[str, Any]:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    worker = worker_config(cfg, worker_id)
    root = Path(cfg['result_root'])
    manifest = read_tsv(root / '00_snapshot/roi_layer_manifest.tsv')
    virtual_slots = {int(value) for value in worker['virtual_slots']}
    rows = [row for row in manifest if stable_slot(row['roi_layer_id'], int(cfg['runtime']['virtual_shards'])) in virtual_slots]
    rows.sort(key=lambda row: (int(row['section_id']), row['track_id'], row['roi_layer_id']))
    full_assignment_count = len(rows)
    if start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if stop_index is not None and stop_index <= start_index:
        raise ValueError('--stop-index must be greater than --start-index')
    rows = rows[start_index:stop_index]
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        rows = rows[:limit]
    qc_ids = set((root / '00_snapshot/qc_roi_layer_ids.txt').read_text(encoding='utf-8').splitlines())
    installed_version = str(getattr(__import__('cellpose'), 'version', 'unknown'))
    if installed_version != str(cfg['runtime']['cellpose_version']):
        raise RuntimeError(f"Cellpose version {installed_version} != {cfg['runtime']['cellpose_version']}")
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in Cellpose worker')
    models_by_kind = {'nucleus': models.Cellpose(gpu=True, model_type=cfg['nucleus']['model_type']), 'cell': models.Cellpose(gpu=True, model_type=cfg['cell']['model_type'])}
    actual_model_devices = {kind: assert_model_on_cuda(kind, model) for kind, model in models_by_kind.items()}
    state_name = state_tag or f'worker_{worker_id:02d}'
    if not state_name.replace('_', '').replace('-', '').isalnum():
        raise ValueError('--state-tag may contain only letters, numbers, _ and -')
    heartbeat_path = root / 'runtime/workers' / f'{state_name}.json'
    summary_path = root / '04_status/workers' / f'{state_name}_summary.json'
    started = time.time()
    completed = 0
    reused = 0
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if (root / 'runtime/STOP').is_file():
            summary = {'status': 'stopped', 'worker_id': worker_id, 'state_tag': state_name, 'full_assignment_count': full_assignment_count, 'slice_start_index': start_index, 'slice_stop_index': stop_index, 'assigned': len(rows), 'processed': index - 1, 'completed': completed, 'reused': reused, 'failed': len(failures), 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()}
            write_json(heartbeat_path, summary)
            write_json(summary_path, summary)
            return summary
        error_record = None
        for attempt in range(2):
            try:
                status = process_one(row, cfg, worker_id, models_by_kind, qc_ids)
                completed += 1
                reused += int(bool(status.get('_reused')))
                error_record = None
                break
            except Exception as error:
                error_record = {'status': 'failed', 'roi_layer_id': row['roi_layer_id'], 'worker_id': worker_id, 'attempt': attempt, 'hostname': socket.gethostname(), 'reason': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(), 'failed_epoch': time.time()}
                torch.cuda.empty_cache()
        if error_record is not None:
            paths = output_paths(root, row['roi_layer_id'], worker_id)
            write_json(paths['status'], error_record)
            failures.append(error_record)
        elapsed = time.time() - started
        write_json(heartbeat_path, {'status': 'running', 'worker_id': worker_id, 'state_tag': state_name, 'full_assignment_count': full_assignment_count, 'slice_start_index': start_index, 'slice_stop_index': stop_index, 'host_config': worker['host'], 'gpu_index_config': worker['gpu_index'], 'gpu_class': worker['gpu_class'], 'actual_model_devices': actual_model_devices, 'virtual_slots': sorted(virtual_slots), 'assigned': len(rows), 'processed': index, 'completed': completed, 'reused': reused, 'failed': len(failures), 'current_roi_layer_id': row['roi_layer_id'], 'elapsed_seconds': elapsed, 'mean_seconds_per_processed': elapsed / index, 'hostname': socket.gethostname(), 'heartbeat_epoch': time.time()})
    summary = {'status': 'PASS' if not failures else 'FAIL', 'worker_id': worker_id, 'state_tag': state_name, 'full_assignment_count': full_assignment_count, 'slice_start_index': start_index, 'slice_stop_index': stop_index, 'host_config': worker['host'], 'gpu_index_config': worker['gpu_index'], 'gpu_class': worker['gpu_class'], 'virtual_slots': sorted(virtual_slots), 'assigned': len(rows), 'processed': len(rows), 'completed': completed, 'reused': reused, 'failed': len(failures), 'failure_roi_layer_ids': [row['roi_layer_id'] for row in failures], 'hostname': socket.gethostname(), 'elapsed_seconds': time.time() - started, 'python': platform.python_version(), 'cellpose_version': installed_version, 'torch_version': torch.__version__, 'cuda_device': torch.cuda.get_device_name(0), 'actual_model_devices': actual_model_devices, 'heartbeat_epoch': time.time()}
    write_json(summary_path, summary)
    write_json(heartbeat_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--worker-id', required=True, type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--stop-index', type=int)
    parser.add_argument('--state-tag')
    args = parser.parse_args()
    summary = run(args.config, args.worker_id, limit=args.limit, start_index=args.start_index, stop_index=args.stop_index, state_tag=args.state_tag)
    return 0 if summary['status'] == 'PASS' else 2
if __name__ == '__main__':
    raise SystemExit(main())
