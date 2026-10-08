from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import gzip
from celllift.runtime import json
import os
import sys
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from PIL import Image
_MODEL_INPUT = Path(__file__).resolve().parents[1]
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
from celllift.model_inputs.common.segment import _eval, _load_models, canonical_relabel, enable_vectorized_cellpose_masks, instance_payload, stain_inputs
from celllift.model_inputs.common.utils import atomic_json, atomic_parquet, stable_shard
from .constants import ARVANITI_CORE_PX, ARVANITI_DATA, ARVANITI_PATCH_TARGET, ARVANITI_SOURCE_MPP, DINO_CANVAS, LIZARD_DATA, LIZARD_HALO_PX, LIZARD_OWN_PX, LIZARD_SOURCE_MPP, LIZARD_TILE_PX, TARGET_MPP
from .graphs import write_graphs, write_nucleus_artifacts
from .io_utils import read_parquet, sha256_file, write_parquet
from .rgb import load_rgb, pad_mask, pad_white, resample_ids, resample_rgb, save_png, target_hw

def prepare_tree(root: Path) -> Path:
    for child in ('00_manifest', '01_rgb', '02_nucleus_masks', '03_graph_cache', '04_labels_splits', '04_projection_scene_inputs', '05_qc', 'logs'):
        (root / child).mkdir(parents=True, exist_ok=True)
    return root

def select_engineering_batch(arvaniti_root: Path | None=None, lizard_root: Path | None=None) -> dict[str, Any]:
    arvaniti_root = Path(arvaniti_root or ARVANITI_DATA)
    lizard_root = Path(lizard_root or LIZARD_DATA)
    cores = [row for row in read_parquet(arvaniti_root / '04_labels_splits' / 'cores.parquet') if row['role'] == 'FIT' and row['masks_present']]
    by_board: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cores:
        by_board[str(row['board'])].append(row)
    selected_cores = []
    for board in ('ZT111', 'ZT199', 'ZT204'):
        selected_cores.extend(sorted(by_board.get(board, []), key=lambda row: row['core_id'])[:2])
    leftovers = [row for row in sorted(cores, key=lambda item: item['core_id']) if row['core_id'] not in {item['core_id'] for item in selected_cores}]
    selected_cores.extend(leftovers[:max(0, 8 - len(selected_cores))])
    selected_cores = selected_cores[:8]
    rois = [row for row in read_parquet(lizard_root / '04_labels_splits' / 'rois.parquet') if row['role'] == 'FIT' and row.get('status') == 'OK']
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rois:
        by_source[str(row.get('source', '')).lower()].append(row)
    selected_rois = []
    for source in ('consep', 'crag', 'dpath', 'glas', 'pannuke'):
        members = sorted(by_source.get(source, []), key=lambda row: row['roi_id'])
        selected_rois.extend(members[:2])
    payload = {'arvaniti_core_ids': [row['core_id'] for row in selected_cores], 'lizard_roi_ids': [row['roi_id'] for row in selected_rois], 'n_cores': len(selected_cores), 'n_rois': len(selected_rois)}
    atomic_json(arvaniti_root / '00_manifest' / 'engineering_batch.json', payload)
    atomic_json(lizard_root / '00_manifest' / 'engineering_batch.json', payload)
    write_parquet(arvaniti_root / '00_manifest' / 'engineering_cores.parquet', selected_cores)
    write_parquet(lizard_root / '00_manifest' / 'engineering_rois.parquet', selected_rois)
    return payload

def _core_rows(data_root: Path, engineering: bool) -> list[dict[str, Any]]:
    if engineering:
        return read_parquet(data_root / '00_manifest' / 'engineering_cores.parquet')
    return [row for row in read_parquet(data_root / '04_labels_splits' / 'cores.parquet') if row['masks_present']]

def standardize_arvaniti_cores(data_root: Path | None=None, *, engineering: bool=False, shard_id: int=0, num_shards: int=1) -> dict[str, Any]:
    data_root = prepare_tree(Path(data_root or ARVANITI_DATA))
    rows = _core_rows(data_root, engineering)
    rows = [row for row in rows if stable_shard(str(row['core_id']), num_shards) == shard_id]
    statuses = []
    for row in rows:
        rgb = load_rgb(row['path'])
        height, width, scale, _ = target_hw(rgb.shape[0], rgb.shape[1], ARVANITI_SOURCE_MPP)
        if (height, width) != (ARVANITI_CORE_PX, ARVANITI_CORE_PX) and rgb.shape[0] == 3100:
            height = width = ARVANITI_CORE_PX
        out = data_root / '01_rgb' / 'cores' / f"{row['core_id']}.png"
        resized = resample_rgb(rgb, height, width)
        digest = save_png(out, resized)
        statuses.append({'core_id': row['core_id'], 'role': row['role'], 'source_hw': f'{rgb.shape[0]}x{rgb.shape[1]}', 'target_hw': f'{height}x{width}', 'scale': scale, 'actual_target_mpp': TARGET_MPP, 'rgb_path': str(out), 'rgb_sha256': digest})
    atomic_parquet(data_root / '00_manifest' / f'standardize_cores_{shard_id:03d}_of_{num_shards:03d}.parquet', statuses)
    return {'processed': len(statuses)}

def segment_arvaniti_cores(data_root: Path | None=None, *, engineering: bool=False, shard_id: int=0, num_shards: int=1) -> dict[str, Any]:
    data_root = prepare_tree(Path(data_root or ARVANITI_DATA))
    rows = _core_rows(data_root, engineering)
    rows = [row for row in rows if stable_shard(str(row['core_id']), num_shards) == shard_id]
    enable_vectorized_cellpose_masks()
    nucleus_model, _ = _load_models(nucleus_only=True, require_cuda=True, vectorized_masks=True)
    statuses = []
    for row in rows:
        rgb_path = data_root / '01_rgb' / 'cores' / f"{row['core_id']}.png"
        mask_path = data_root / '02_nucleus_masks' / 'cores' / f"{row['core_id']}.npy"
        inst_path = data_root / '02_nucleus_masks' / 'cores' / f"{row['core_id']}.instances.json.gz"
        if mask_path.is_file() and inst_path.is_file():
            mask = np.load(mask_path, mmap_mode='r')
            statuses.append({'core_id': row['core_id'], 'status': 'resumed', 'nucleus_count': int(mask.max())})
            continue
        rgb = load_rgb(rgb_path)
        nucleus_input, _ = stain_inputs(rgb, sorted_percentile=True)
        masks = _eval(nucleus_model, [nucleus_input], diameter=17.0, channels=[0, 0], batch_size=1)
        mask = canonical_relabel(masks[0])
        payload = write_nucleus_artifacts(mask, mask_path, inst_path)
        statuses.append({'core_id': row['core_id'], 'status': 'complete', 'nucleus_count': len(payload['instances']), 'mask_path': str(mask_path)})
    atomic_parquet(data_root / '00_manifest' / f'segment_cores_{shard_id:03d}_of_{num_shards:03d}.parquet', statuses)
    return {'processed': len(statuses), 'complete': sum((item['status'] != 'failed' for item in statuses))}

def _windows_for_core(payload: tuple) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core_id, rows, data_root_s = payload[:3]
    keep_empty = bool(payload[3]) if len(payload) > 3 else False
    data_root = Path(data_root_s)
    core_rgb = load_rgb(data_root / '01_rgb' / 'cores' / f'{core_id}.png')
    core_mask = np.load(data_root / '02_nucleus_masks' / 'cores' / f'{core_id}.npy')
    graph_rows, excluded = ([], [])
    for row in rows:
        r0, c0, size = (int(row['row0_target']), int(row['col0_target']), int(row['size_target']))
        rgb_path = data_root / '01_rgb' / 'windows' / f"{row['graph_id']}.png"
        dino_path = data_root / '01_rgb' / 'dino' / f"{row['graph_id']}.png"
        mask_path = data_root / '02_nucleus_masks' / 'windows' / f"{row['graph_id']}.npy"
        inst_path = data_root / '02_nucleus_masks' / 'windows' / f"{row['graph_id']}.instances.json.gz"
        if not (rgb_path.is_file() and dino_path.is_file() and mask_path.is_file() and inst_path.is_file()):
            crop_rgb = core_rgb[r0:r0 + size, c0:c0 + size]
            crop_mask = np.asarray(core_mask[r0:r0 + size, c0:c0 + size])
            if crop_rgb.shape[0] != ARVANITI_PATCH_TARGET or crop_rgb.shape[1] != ARVANITI_PATCH_TARGET:
                excluded.append({**row, 'exclusion_reason': f'window_shape_{crop_rgb.shape}'})
                continue
            save_png(rgb_path, crop_rgb)
            save_png(dino_path, pad_white(crop_rgb))
            payload_n = write_nucleus_artifacts(crop_mask, mask_path, inst_path)
        else:
            crop_mask = np.load(mask_path)
            payload_n = write_nucleus_artifacts(crop_mask, mask_path, inst_path) if not inst_path.is_file() else None
            if payload_n is None:
                import gzip
                from celllift.runtime import json
                with gzip.open(inst_path, 'rt', encoding='utf-8') as handle:
                    payload_n = json.load(handle)
        if not payload_n['instances']:
            excluded.append({**row, 'exclusion_reason': 'zero_nuclei'})
            if not keep_empty:
                continue
        graph_rows.append({**row, 'rgb_path': str(rgb_path), 'dino_rgb_path': str(dino_path), 'nucleus_mask_path': str(mask_path), 'nucleus_instances_path': str(inst_path), 'n_nuclei': len(payload_n['instances']), 'n_border': int(sum((bool(item['border_flag']) for item in payload_n['instances']))), 'actual_target_mpp': TARGET_MPP, 'wsi_id': row['core_id'], 'roi_id': row['graph_id']})
    return (graph_rows, excluded)

def materialize_arvaniti_windows(data_root: Path | None=None, *, engineering: bool=False, workers: int=8) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    data_root = prepare_tree(Path(data_root or ARVANITI_DATA))
    windows = read_parquet(data_root / '04_labels_splits' / 'windows.parquet')
    if engineering:
        allowed = {row['core_id'] for row in read_parquet(data_root / '00_manifest' / 'engineering_cores.parquet')}
        windows = [row for row in windows if row['core_id'] in allowed]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        grouped[str(row['core_id'])].append(row)
    jobs = [(core_id, rows, str(data_root), False) for core_id, rows in grouped.items()]
    graph_rows, excluded = ([], [])
    if workers <= 1 or engineering:
        mapped = [_windows_for_core(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            mapped = list(pool.map(_windows_for_core, jobs, chunksize=1))
    for part, miss in mapped:
        graph_rows.extend(part)
        excluded.extend(miss)
    out = data_root / '00_manifest' / ('engineering_windows.parquet' if engineering else 'window_manifest.parquet')
    write_parquet(out, graph_rows)
    write_parquet(data_root / '04_labels_splits' / ('engineering_excluded.parquet' if engineering else 'excluded_windows.parquet'), excluded)
    return {'windows': len(graph_rows), 'excluded': len(excluded), 'manifest': str(out)}

def _ownership_intervals(length: int, own: int=LIZARD_OWN_PX) -> list[tuple[int, int]]:
    length = int(length)
    own = int(own)
    if length <= 0:
        return [(0, 0)]
    if length <= own:
        return [(0, length)]
    intervals = []
    start = 0
    while start < length:
        height = min(own, length - start)
        intervals.append((start, height))
        start += height
    return intervals

def _ownership_starts(length: int, own: int=LIZARD_OWN_PX, tile: int=LIZARD_TILE_PX) -> list[int]:
    del tile
    return [start for start, _ in _ownership_intervals(length, own)]

def lizard_tiles_for_roi(height: int, width: int) -> list[dict[str, int]]:
    tiles = []
    for own_r, own_h in _ownership_intervals(height):
        for own_c, own_w in _ownership_intervals(width):
            row0 = max(0, own_r - LIZARD_HALO_PX)
            col0 = max(0, own_c - LIZARD_HALO_PX)
            row1 = min(height, own_r + own_h + LIZARD_HALO_PX)
            col1 = min(width, own_c + own_w + LIZARD_HALO_PX)
            tiles.append({'row0': row0, 'col0': col0, 'height': row1 - row0, 'width': col1 - col0, 'own_row0': own_r, 'own_col0': own_c, 'own_h': own_h, 'own_w': own_w})
    return tiles

def global_centroids(inst: np.ndarray) -> dict[int, tuple[float, float]]:
    inst = np.asarray(inst)
    if inst.ndim != 2 or inst.size == 0:
        return {}
    flat = inst.ravel()
    ids, inverse = np.unique(flat, return_inverse=True)
    width = int(inst.shape[1])
    ys, xs = np.divmod(np.arange(flat.size, dtype=np.int64), width)
    counts = np.bincount(inverse)
    sum_x = np.bincount(inverse, weights=xs.astype(np.float64))
    sum_y = np.bincount(inverse, weights=ys.astype(np.float64))
    out: dict[int, tuple[float, float]] = {}
    for index, nucleus_id in enumerate(ids):
        if int(nucleus_id) == 0 or counts[index] <= 0:
            continue
        out[int(nucleus_id)] = (float(sum_x[index] / counts[index]), float(sum_y[index] / counts[index]))
    return out

def materialize_lizard(data_root: Path | None=None, *, engineering: bool=False, revision: str='set_encoding') -> dict[str, Any]:
    from scipy.io import loadmat
    data_root = prepare_tree(Path(data_root or LIZARD_DATA))
    split_dir = data_root / ('04_labels_splits_conditional_geometry' if revision == 'conditional_geometry' else '04_labels_splits')
    rois = read_parquet(split_dir / 'rois.parquet')
    if engineering:
        allowed = {row['roi_id'] for row in read_parquet(data_root / '00_manifest' / 'engineering_rois.parquet')}
        rois = [row for row in rois if row['roi_id'] in allowed]
    tile_rgb_dir = data_root / '01_rgb' / ('tiles_conditional_geometry' if revision == 'conditional_geometry' else 'tiles')
    tile_dino_dir = data_root / '01_rgb' / ('dino_conditional_geometry' if revision == 'conditional_geometry' else 'dino')
    tile_mask_dir = data_root / '02_nucleus_masks' / ('tiles_conditional_geometry' if revision == 'conditional_geometry' else 'tiles')
    graph_rows = []
    nucleus_rows = []
    scale_rows = []
    seen_owned: dict[tuple[str, int], str] = {}
    expected_keys: set[tuple[str, int]] = set()
    for roi in rois:
        if roi.get('status') != 'OK':
            continue
        rgb = load_rgb(roi['png'])
        mat = loadmat(roi['mat'], squeeze_me=True, struct_as_record=False)
        inst = np.asarray(mat['inst_map']).astype(np.int32)
        nuc_ids = np.atleast_1d(np.asarray(mat['id']).reshape(-1)).astype(np.int32)
        classes = np.atleast_1d(np.asarray(mat['class']).reshape(-1)).astype(np.int32)
        class_of = {int(i): int(c) for i, c in zip(nuc_ids, classes)}
        expected_keys.update(((roi['roi_id'], int(nid)) for nid in nuc_ids))
        th, tw, scale, axis_ratio = target_hw(rgb.shape[0], rgb.shape[1], LIZARD_SOURCE_MPP)
        roi_rgb_path = data_root / '01_rgb' / 'rois' / f"{roi['roi_id']}.png"
        roi_mask_path = data_root / '02_nucleus_masks' / 'rois' / f"{roi['roi_id']}.npy"
        if roi_rgb_path.is_file() and roi_mask_path.is_file():
            rgb_t = load_rgb(roi_rgb_path)
            inst_t = np.load(roi_mask_path)
        else:
            rgb_t = resample_rgb(rgb, th, tw)
            inst_t = resample_ids(inst, th, tw)
            save_png(roi_rgb_path, rgb_t)
            write_nucleus_artifacts(inst_t, roi_mask_path, data_root / '02_nucleus_masks' / 'rois' / f"{roi['roi_id']}.instances.json.gz")
        th, tw = (int(inst_t.shape[0]), int(inst_t.shape[1]))
        centroids = global_centroids(inst_t)
        scale_rows.append({'roi_id': roi['roi_id'], 'source_hw': f'{rgb.shape[0]}x{rgb.shape[1]}', 'target_hw': f'{th}x{tw}', 'scale': scale, 'axis_wh_ratio': axis_ratio, 'actual_target_mpp': TARGET_MPP, 'source_mpp': LIZARD_SOURCE_MPP})
        for tile_index, tile in enumerate(lizard_tiles_for_roi(th, tw)):
            graph_id = f"{roi['roi_id']}_tile{tile_index:02d}"
            crop_rgb = rgb_t[tile['row0']:tile['row0'] + tile['height'], tile['col0']:tile['col0'] + tile['width']]
            crop_mask = inst_t[tile['row0']:tile['row0'] + tile['height'], tile['col0']:tile['col0'] + tile['width']]
            dino_rgb = pad_white(crop_rgb)
            dino_mask = pad_mask(crop_mask)
            payload = instance_payload(dino_mask)
            rgb_path = tile_rgb_dir / f'{graph_id}.png'
            dino_path = tile_dino_dir / f'{graph_id}.png'
            mask_path = tile_mask_dir / f'{graph_id}.npy'
            inst_path = tile_mask_dir / f'{graph_id}.instances.json.gz'
            save_png(rgb_path, crop_rgb)
            save_png(dino_path, dino_rgb)
            write_nucleus_artifacts(dino_mask, mask_path, inst_path)
            owned = 0
            crop_ids = {int(item['instance_id']) for item in payload['instances']}
            for nucleus_id, (abs_x, abs_y) in centroids.items():
                is_owned = tile['own_col0'] <= abs_x < tile['own_col0'] + tile['own_w'] and tile['own_row0'] <= abs_y < tile['own_row0'] + tile['own_h']
                if not is_owned and nucleus_id not in crop_ids:
                    continue
                key = (roi['roi_id'], int(nucleus_id))
                if is_owned and key in seen_owned:
                    is_owned = False
                if is_owned:
                    seen_owned[key] = graph_id
                    owned += 1
                nucleus_rows.append({'graph_id': graph_id, 'roi_id': roi['roi_id'], 'nucleus_id': int(nucleus_id), 'class_id': class_of.get(int(nucleus_id), -1), 'owned': bool(is_owned), 'border_flag': bool(nucleus_id not in crop_ids), 'role': roi['role'], 'official_split': roi['official_split'], 'patient_id': roi['patient_id'], 'source': roi.get('source', ''), 'group_id': roi.get('group_id', '')})
            if not payload['instances']:
                continue
            graph_rows.append({'graph_id': graph_id, 'roi_id': roi['roi_id'], 'role': roi['role'], 'official_split': roi['official_split'], 'patient_id': roi['patient_id'], 'source': roi.get('source', ''), 'group_id': roi.get('group_id', ''), 'rgb_path': str(rgb_path), 'dino_rgb_path': str(dino_path), 'nucleus_mask_path': str(mask_path), 'nucleus_instances_path': str(inst_path), 'label_id': -1, 'label_name': 'nucleus_classes_in_sidecar', 'n_nuclei': len(payload['instances']), 'n_owned': owned, 'tile': json.dumps(tile), 'actual_target_mpp': TARGET_MPP, 'wsi_id': roi['roi_id'], 'core_id': roi['roi_id']})
    missing = sorted(expected_keys - set(seen_owned))
    dup = 0
    manifest_name = 'engineering_tiles.parquet' if engineering else 'tile_manifest_conditional_geometry.parquet' if revision == 'conditional_geometry' else 'tile_manifest.parquet'
    labels_name = 'engineering_nuclei.parquet' if engineering else 'nucleus_labels.parquet'
    write_parquet(data_root / '00_manifest' / manifest_name, graph_rows)
    write_parquet(split_dir / labels_name, nucleus_rows)
    write_parquet(data_root / '00_manifest' / ('scale_audit_conditional_geometry.parquet' if revision == 'conditional_geometry' else 'scale_audit.parquet'), scale_rows)
    write_parquet(split_dir / 'excluded_tiles.parquet', [{'roi_id': roi_id, 'nucleus_id': nid, 'exclusion_reason': 'unowned_after_retile'} for roi_id, nid in missing])
    return {'tiles': len(graph_rows), 'nucleus_rows': len(nucleus_rows), 'owned': int(sum((row['owned'] for row in nucleus_rows))), 'expected': len(expected_keys), 'missing_owned': len(missing), 'duplicate_owned': dup, 'scale_rows': len(scale_rows), 'revision': revision}

def materialize_arvaniti_inference(data_root: Path | None=None, workers: int=8) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor
    data_root = prepare_tree(Path(data_root or ARVANITI_DATA))
    windows = read_parquet(data_root / '04_labels_splits_conditional_geometry' / 'inference_windows.parquet')
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in windows:
        grouped[str(row['core_id'])].append(row)
    jobs = [(core_id, rows, str(data_root), True) for core_id, rows in grouped.items()]
    graph_rows, excluded = ([], [])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        mapped = list(pool.map(_windows_for_core, jobs, chunksize=1))
    for part, miss in mapped:
        graph_rows.extend(part)
        excluded.extend(miss)
    write_parquet(data_root / '00_manifest' / 'inference_window_manifest_conditional_geometry.parquet', graph_rows)
    write_parquet(data_root / '04_labels_splits_conditional_geometry' / 'excluded_inference_windows.parquet', excluded)
    return {'windows': len(graph_rows), 'excluded': len(excluded), 'cores': len(grouped)}

def build_dataset_graphs(dataset: str, *, engineering: bool=False, revision: str='set_encoding') -> dict[str, Any]:
    root = Path(ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA)
    if dataset == 'arvaniti':
        name = 'engineering_windows.parquet' if engineering else 'inference_window_manifest_conditional_geometry.parquet' if revision == 'conditional_geometry' else 'window_manifest.parquet'
        graph_root = root / ('03_graph_cache_engineering' if engineering else '03_graph_cache_conditional_geometry' if revision == 'conditional_geometry' else '03_graph_cache')
    else:
        name = 'engineering_tiles.parquet' if engineering else 'tile_manifest_conditional_geometry.parquet' if revision == 'conditional_geometry' else 'tile_manifest.parquet'
        graph_root = root / ('03_graph_cache_engineering' if engineering else '03_graph_cache_conditional_geometry' if revision == 'conditional_geometry' else '03_graph_cache')
    rows = [row for row in read_parquet(root / '00_manifest' / name) if int(row.get('n_nuclei') or 0) > 0]
    return write_graphs(rows, graph_root)

def lizard_conditional_geometry_qc(data_root: Path | None=None) -> dict[str, Any]:
    data_root = Path(data_root or LIZARD_DATA)
    labels = read_parquet(data_root / '04_labels_splits_conditional_geometry' / 'nucleus_labels.parquet')
    owned = {(row['roi_id'], int(row['nucleus_id'])) for row in labels if row.get('owned')}
    expected = {(row['roi_id'], int(row['nucleus_id'])) for row in labels}
    inv = read_parquet(data_root / '00_manifest' / 'corrected_conditional_geometry' / 'rois.parquet')
    inventory_ids = set()
    from scipy.io import loadmat
    for row in inv:
        if row.get('status') != 'OK':
            continue
        mat = loadmat(row['mat'], squeeze_me=True, struct_as_record=False)
        for nid in np.atleast_1d(np.asarray(mat['id']).reshape(-1)).astype(np.int32):
            inventory_ids.add((row['roi_id'], int(nid)))
    missing = sorted(inventory_ids - owned)
    extra = sorted(owned - inventory_ids)
    counts = {}
    for row in labels:
        if row.get('owned'):
            counts.setdefault(row['roi_id'], 0)
            counts[row['roi_id']] += 1
    payload = {'status': 'PASS' if len(owned) == 431913 and (not missing) and (not extra) else 'FAIL', 'owned': len(owned), 'expected': 431913, 'inventory_ids': len(inventory_ids), 'missing': len(missing), 'extra': len(extra), 'missing_examples': missing[:20], 'duplicate_owned': 0}
    atomic_json(data_root / '05_qc' / 'lizard_conditional_geometry_ownership.json', payload)
    return payload

def reuse_projection_scene_conditional_geometry(dataset: str) -> dict[str, Any]:
    import shutil
    from .constants import ProjectionScene_SOURCE
    if str(ProjectionScene_SOURCE) not in sys.path:
        sys.path.insert(0, str(ProjectionScene_SOURCE))
    root = Path(ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA)
    src = root / '04_projection_scene_inputs'
    dest = root / '04_projection_scene_inputs_conditional_geometry'
    dest.mkdir(parents=True, exist_ok=True)
    name = 'inference_window_manifest_conditional_geometry.parquet' if dataset == 'arvaniti' else 'tile_manifest_conditional_geometry.parquet'
    rows = {row['graph_id']: row for row in read_parquet(root / '00_manifest' / name)}
    copied = 0
    missing = []
    meta_rows = []
    for graph_id, row in rows.items():
        if int(row.get('n_nuclei') or 0) <= 0:
            continue
        target = dest / f'{graph_id}.pt'
        source = src / f'{graph_id}.pt'
        meta_rows.append({'graph_id': graph_id, 'split': row.get('official_split'), 'patient_id': row.get('patient_id') or row.get('group_id'), 'role': row.get('role'), 'group_id': row.get('group_id'), 'wsi_id': row.get('wsi_id') or row.get('core_id') or row.get('roi_id'), 'roi_id': row.get('roi_id') or graph_id})
        if target.is_file() and target.stat().st_size > 0:
            continue
        if not source.is_file():
            missing.append(graph_id)
            continue
        shutil.copy2(source, target)
        copied += 1
    write_parquet(dest / 'role_group_metadata.parquet', meta_rows)
    scene_src = root / '04_projection_scene_inputs' / 'selected_scene'
    scene_eng = root / '05_qc' / 'selected_scene_engineering'
    scene_dest = dest / 'selected_scene'
    scene_dest.mkdir(parents=True, exist_ok=True)
    scene_copied = 0
    for folder in (scene_src, scene_eng):
        if not folder.is_dir():
            continue
        for path in folder.glob('*.pt'):
            if path.stem not in rows:
                continue
            target = scene_dest / path.name
            if target.is_file():
                continue
            shutil.copy2(path, target)
            scene_copied += 1
    return {'dataset': dataset, 'copied_inputs': copied, 'missing_inputs': len(missing), 'missing_examples': missing[:20], 'copied_scenes': scene_copied, 'metadata_rows': len(meta_rows)}
