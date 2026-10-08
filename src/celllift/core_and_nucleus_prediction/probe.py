from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
import numpy as np
from celllift.runtime import torch
from .constants import ARVANITI_DATA, LIZARD_DATA, RESULT_ROOT, SEED
from .data import _align, _graph_roots, _x2_from_record, scene_path
import sys
from celllift.runtime import ResourcePath as _P
_MODEL_INPUT = _P(__file__).resolve().parents[1] / 'model_inputs'
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
_PUBLIC = _P(__file__).resolve().parents[1]
if str(_PUBLIC) not in sys.path:
    sys.path.insert(0, str(_PUBLIC))
from celllift.model_inputs.arvaniti_lizard.graphs import GraphReader
from celllift.model_inputs.arvaniti_lizard.io_utils import atomic_json, read_parquet
from celllift.model_inputs.common.graph import GraphRecord
from celllift.matched_geometry_controls.residual import _fit, _predict, graph_context, load_mask_probe
from celllift.matched_geometry_controls.tokens import FeatureNormalizer
conditional_geometry_ROOT = Path(_resource_path('artifact_0046'))

def _collect(dataset: str):
    root = ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA
    name = 'inference_window_manifest_conditional_geometry.parquet' if dataset == 'arvaniti' else 'tile_manifest_conditional_geometry.parquet'
    if not (root / '00_manifest' / name).is_file():
        name = 'window_manifest.parquet' if dataset == 'arvaniti' else 'tile_manifest.parquet'
    rows = [row for row in read_parquet(root / '00_manifest' / name) if int(row.get('n_nuclei') or 0) > 0]
    graph_root, scene_root, _ = _graph_roots(root)
    if not scene_root.is_dir():
        scene_root = root / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
    reader = GraphReader(graph_root)
    rays, target, valid, groups, graph_ids, nucleus_ids, roles = ([], [], [], [], [], [], [])
    skipped = {'no_scene': 0, 'no_graph': 0, 'empty': 0}
    for row in rows:
        scene_file = scene_path(scene_root, row['graph_id'])
        if not scene_file.is_file():
            skipped['no_scene'] += 1
            continue
        ids = None
        x2 = None
        try:
            record = GraphRecord.from_bytes(reader.get(row['graph_id']))
            ids = np.asarray(record.nucleus_id, np.int64)
            x2 = _x2_from_record(record)[:, 2:]
        except Exception:
            inp = root / '04_projection_scene_inputs_conditional_geometry' / f"{row['graph_id']}.pt"
            if not inp.is_file():
                inp = root / '04_projection_scene_inputs' / f"{row['graph_id']}.pt"
            if inp.is_file():
                try:
                    graph = torch.load(inp, map_location='cpu', weights_only=False)['graph']
                    ids = np.asarray(graph.nucleus_id.cpu() if hasattr(graph.nucleus_id, 'cpu') else graph.nucleus_id, np.int64)
                    ray_arr = np.asarray(graph.nucleus_rays_um.cpu() if hasattr(graph.nucleus_rays_um, 'cpu') else graph.nucleus_rays_um, np.float32)
                    x2 = ray_arr.reshape(len(ids), -1)[:, :36]
                except Exception:
                    skipped['no_graph'] += 1
                    continue
            else:
                skipped['no_graph'] += 1
                continue
        payload = torch.load(scene_file, map_location='cpu', weights_only=False)
        if ids is None or len(ids) == 0:
            skipped['empty'] += 1
            continue
        x3 = _align(ids, {'nucleus_id': np.asarray(payload.get('nucleus_id', []), np.int64).reshape(-1), 'x3': np.asarray(payload.get('direct_geometry9', np.zeros((len(ids), 9))), np.float32)})
        ncr = np.asarray(payload.get('valid_ncr', np.ones(len(ids))), bool).reshape(-1)
        if len(ncr) != len(ids):
            ncr = np.ones(len(ids), bool)
        rays.append(x2)
        target.append(x3)
        valid.append(ncr)
        group = str(row.get('group_id') or row.get('patient_id') or row.get('core_id') or row['graph_id'])
        groups.extend([group] * len(ids))
        graph_ids.extend([row['graph_id']] * len(ids))
        nucleus_ids.append(ids)
        roles.extend([str(row.get('role', 'FIT')).upper()] * len(ids))
    reader.close()
    packed = {'rays': np.concatenate(rays) if rays else np.zeros((0, 36), np.float32), 'target': np.concatenate(target) if target else np.zeros((0, 9), np.float32), 'valid': np.concatenate(valid) if valid else np.zeros((0,), bool), 'groups': np.asarray(groups), 'graph_id': np.asarray(graph_ids), 'nucleus_id': np.concatenate(nucleus_ids) if nucleus_ids else np.zeros((0,), np.int64), 'role': np.asarray(roles), 'skipped': skipped, 'scene_root': str(scene_root), 'n_rows': len(rows)}
    return packed

def fit_probe(dataset: str, device: str='cuda') -> dict:
    packed = _collect(dataset)
    if len(packed['rays']) == 0:
        return {'status': 'WAIT', 'dataset': dataset, 'reason': 'no selected-scene', 'skipped': packed.get('skipped'), 'scene_root': packed.get('scene_root'), 'n_rows': packed.get('n_rows')}
    train = packed['role'] == 'FIT'
    groups = packed['groups']
    unique = sorted(set(groups[train].tolist()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique)
    n_val = max(1, int(round(0.2 * len(unique)))) if len(unique) >= 2 else 0
    inner_val_groups = set(unique[:n_val])
    inner_val = train & np.isin(groups, list(inner_val_groups))
    inner_train = train & ~inner_val
    rays = packed['rays'].astype(np.float32)
    valid = packed['valid'].astype(bool)
    normalizer = FeatureNormalizer.fit(packed['target'].astype(np.float32), valid, train)
    target = normalizer.transform(packed['target'].astype(np.float32), valid)
    codes = {gid: i for i, gid in enumerate(sorted(set(packed['graph_id'].tolist())))}
    graph_code = np.asarray([codes[g] for g in packed['graph_id']], np.int64)
    ray_mean, ray_std = (rays[train].mean(0), np.maximum(rays[train].std(0), 1e-06))
    rays_z = ((rays - ray_mean) / ray_std).astype(np.float32)
    context_raw = graph_context(rays_z, graph_code)
    context = ((context_raw - context_raw[train].mean(0)) / np.maximum(context_raw[train].std(0), 1e-06)).astype(np.float32)
    model_class = load_mask_probe(conditional_geometry_ROOT)
    model, epochs, history = _fit(model_class, rays_z, context, target, valid, inner_train, inner_val if inner_val.any() else inner_train, seed=SEED, device=device)
    model, _, history = _fit(model_class, rays_z, context, target, valid, train, inner_val if inner_val.any() else train, seed=SEED + 1000, device=device, fixed_epochs=max(1, epochs))
    pred = _predict(model, rays_z, context, device)
    residual = (target - pred).astype(np.float32)
    residual[~valid, 8] = 0.0
    dest = (ARVANITI_DATA if dataset == 'arvaniti' else LIZARD_DATA) / '04_projection_scene_inputs_conditional_geometry' / 'residual'
    dest.mkdir(parents=True, exist_ok=True)
    by_graph = defaultdict(list)
    for i, graph_id in enumerate(packed['graph_id']):
        by_graph[str(graph_id)].append(i)
    for graph_id, idxs in by_graph.items():
        ids = packed['nucleus_id'][idxs]
        np.savez_compressed(dest / f'{graph_id}.npz', nucleus_id=ids, residual9=residual[idxs], prediction9=pred[idxs])
    report = {'status': 'PASS', 'dataset': dataset, 'n': int(len(rays)), 'fit': int(train.sum()), 'inner_epochs': int(epochs), 'graphs': len(by_graph)}
    atomic_json(dest / 'probe.json', report)
    (RESULT_ROOT / dataset / 'probe').mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / dataset / 'probe' / 'probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
