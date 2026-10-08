from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from ..data import Sample, SceneCache, build_batch, group_equal_edge_stats, group_equal_stats, prepare_standardization
from ..dataset import B_ROLE, DATASETS, PREDICTION_FORM, batch_result_root, cache_index_path, dev_index_rows, train_rows
from ..io_utils import atomic_json, atomic_torch, read_json
from .models import RouteD

def pretrain_dir(cfg: Mapping[str, Any], target: str, fold: int | None=None) -> Path:
    root = batch_result_root(cfg, 'd') / 'pretrain' / target
    return root if fold is None else root / f'fold_{int(fold):02d}'

def run_pretrain(cfg: Mapping[str, Any], target: str, device: str, dataset: str='sicapv2', max_epochs: int=6, fold: int | None=None, official_fit: list[str] | None=None) -> dict[str, Any]:
    destination = pretrain_dir(cfg, target, fold)
    manifest_path = destination / 'job.json'
    if manifest_path.is_file() and read_json(manifest_path).get('status') == 'PASS':
        return {'status': 'REUSED', **read_json(manifest_path)}
    used_pool = 'downstream_train_fit' if fold is not None else 'downstream_train'
    isolate = bool(cfg.get('official_result_subdir') or official_fit is not None)
    serial = Path(cfg['paths'].get('serial_cache') or '')
    if not isolate and serial.is_dir() and any(serial.rglob('scene_index.json')):
        used_pool = 'serial_train'
    if not cache_index_path(cfg, dataset).is_file():
        for name in DATASETS:
            if cache_index_path(cfg, name).is_file():
                dataset = name
                break
    cache = SceneCache(cfg, dataset)
    cache.preload()
    rows = train_rows(dev_index_rows(dataset, cfg))
    if official_fit:
        allowed = set(map(str, official_fit))
        expanded = [row for row in rows or dev_index_rows(dataset, cfg) if row['graph_id'] in allowed or str(row.get('roi_id') or '') in allowed or str(row.get('patient_id') or '') in allowed]
        rows = expanded or [{'graph_id': gid, 'group_id': 'g'} for gid in official_fit if gid in cache.graphs]
        used_pool = f'official_fit_fold{int(fold)}' if fold is not None else 'official_fit'
    elif fold is not None:
        rows = [row for row in rows if row.get('fold') is not None and int(row['fold']) != int(fold)]
        used_pool = f'downstream_train_except_fold{int(fold)}'
    ids = [row['graph_id'] for row in rows if row['graph_id'] in cache.graphs]
    groups = [str(row['group_id']) for row in rows if row['graph_id'] in cache.graphs]
    mean, std, fill3d = group_equal_stats(cache, ids, groups)
    e2m, e2s, e3m, e3s = group_equal_edge_stats(cache, ids, groups)
    arm = 'D3' if target == '3d' else 'D2'
    prepare_standardization(cache, mean=mean, std=std, fill3d=fill3d, edge_mean2d=e2m, edge_std2d=e2s, edge_mean3d=e3m, edge_std3d=e3s, arm=arm)
    model = RouteD(classes=4, arm=arm, bag=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = np.random.default_rng(42)
    best = float('inf')
    destination.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(max_epochs)):
        model.train()
        order = rng.permutation(len(ids))
        running, steps = (0.0, 0)
        for start in range(0, len(order), 4):
            chosen = [ids[int(i)] for i in order[start:start + 4]]
            samples = [Sample(gid, [gid], 0, 0, 'g', np.ones(4, np.float32)) for gid in chosen]
            batch = build_batch(cache, samples, arm=arm, mean=mean, std=std, fill3d=fill3d, edge_mean2d=e2m, edge_std2d=e2s, edge_mean3d=e3m, edge_std3d=e3s, device=device, with_spatial=True)
            loss = model.pretrain_losses(batch, dataset)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running = loss.detach() if steps == 0 else running + loss.detach()
            steps += 1
        value = float(running) / max(1, steps)
        if value < best:
            best = value
            atomic_torch(destination / 'best.pt', {'model': model.state_dict(), 'pool': used_pool, 'target': target, 'loss': best})
    manifest = {'status': 'PASS', 'route': 'd', 'target': target, 'dataset': dataset, 'pretrain_pool': used_pool, 'prediction_form': PREDICTION_FORM, 'b_role': B_ROLE, 'best_loss': best, 'parameters': model.parameter_count()}
    atomic_json(manifest_path, manifest)
    return manifest
