from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from ..data import Batch, SceneCache, build_batch
from ..dataset import DATASET_SPEC, route_arms
from ..foundation.loop import train_full
from .fields import field_sidecar, load_or_raster_field
from .models import RouteC
from .validate import run_validate

def build_model(**kwargs):
    return RouteC(**kwargs)

def prefetch_fields(cache: SceneCache, dataset: str, arm: str, workers: int=4) -> None:
    items = []
    for graph_id, graph in cache.graphs.items():
        entry = cache.entries.get(graph_id) or {}
        path = entry.get('path') or ''
        if path and field_sidecar(path, arm).is_file():
            continue
        items.append((graph_id, graph))

    def one(pair):
        graph_id, graph = pair
        load_or_raster_field(graph, cache.entries.get(graph_id) or {}, dataset, arm)
    if not items:
        return
    if workers <= 1 or len(items) < 8:
        for item in items:
            one(item)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        list(pool.map(one, items))

def load_field_ready(graph, entry: dict, dataset: str, arm: str, timeout: float=600.0):
    import time
    cached = getattr(graph, '_fields', None)
    if cached is None:
        cached = {}
        graph._fields = cached
    hit = cached.get(arm)
    if hit is not None:
        return hit
    sidecar = field_sidecar(entry.get('path') or '', arm) if entry.get('path') else None
    deadline = time.monotonic() + float(timeout)
    while sidecar is not None:
        if sidecar.is_file():
            try:
                field = np.asarray(np.load(sidecar), np.float32)
            except (OSError, ValueError):
                time.sleep(0.05)
                continue
            cached[arm] = field
            return field
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    return load_or_raster_field(graph, entry, dataset, arm)

def _attach_fields(cache: SceneCache, batch: Batch, samples, dataset: str, arm: str, device: str) -> Batch:
    pairs = []
    for sample in samples:
        for graph_id in sample.graph_ids:
            pairs.append((cache.get(graph_id), cache.entries.get(graph_id) or {}))

    def one(pair):
        return load_field_ready(pair[0], pair[1], dataset, arm)
    if len(pairs) <= 1:
        fields = [one(pair) for pair in pairs]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as pool:
            fields = list(pool.map(one, pairs))
    batch.field = torch.from_numpy(np.stack(fields, 0) if fields else np.zeros((0, 10, 74, 74), np.float32)).to(device, non_blocking=True)
    return batch

def train_outer(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, seed: int, device: str, **kwargs) -> dict[str, Any]:
    gate = run_validate(cfg, dataset)
    if not gate.get('allow_train'):
        raise RuntimeError(f'C field validation failed: {gate}')
    cache = kwargs.pop('cache', None)
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload(load_spatial_maps=True)
    original = build_batch

    def wrapped(*args, **batch_kwargs):
        batch = original(*args, **batch_kwargs)
        samples = args[1]
        return _attach_fields(cache, batch, samples, dataset, arm, device)
    import celllift.morphology_interaction.foundation.loop as loop
    loop.build_batch = wrapped
    try:
        return train_full(cfg, dataset, 'c', arm, outer, seed, device, build_model, cache=cache, with_spatial=True, pretrain_pool='field_validate_pass', **kwargs)
    finally:
        loop.build_batch = original

def job_matrix(dataset: str, seed: int=42) -> list[dict]:
    return [{'dataset': dataset, 'stage': 'task', 'route': 'c', 'arm': arm, 'outer': outer, 'seed': seed} for arm in route_arms('c') for outer in range(int(DATASET_SPEC[dataset]['folds']))]
