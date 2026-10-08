from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
import numpy as np
from ..data import SceneCache
from ..dataset import DATASET_SPEC, route_arms
from ..foundation.loop import build_samples, fold_split, partition, train_full
from ..baseline import load_baseline
from .models import RouteB, fit_prototypes

def build_model(**kwargs):
    return RouteB(**kwargs)

def _fit_on_train(cache: SceneCache, samples, seed: int):
    dinos = []
    for sample in samples:
        for graph_id in sample.graph_ids:
            graph = cache.get(graph_id)
            if graph.include.any():
                dinos.append(graph.dino[graph.include])
    stacked = np.concatenate(dinos, 0) if dinos else np.zeros((8, 384), np.float32)
    return fit_prototypes(stacked, seed=seed)

def train_outer(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, seed: int, device: str, **kwargs) -> dict[str, Any]:
    import torch
    cache = kwargs.pop('cache', None)
    published = kwargs.pop('published', None)
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload()
    if published is None:
        published = load_baseline(dataset, cfg)
    from ..dataset import train_rows, dev_index_rows
    samples = build_samples(dataset, train_rows(dev_index_rows(dataset, cfg)), published, set(cache.graphs))
    official_fit = kwargs.get('official_fit')
    if official_fit:
        allowed = set(map(str, official_fit))
        train = [sample for sample in samples if sample.bag_id in allowed]
    else:
        outer_train, _ = fold_split(samples, outer)
        train, _ = partition(outer_train, seed=seed + 17 * outer)
    if not train:
        raise RuntimeError('empty prototype fit set')
    mean, basis, centers = _fit_on_train(cache, train, seed)

    def builder(**model_kwargs):
        model = RouteB(**model_kwargs)
        model.pca_mean.copy_(torch.as_tensor(mean))
        model.pca_basis.copy_(torch.as_tensor(basis))
        model.prototypes.data.copy_(torch.as_tensor(centers))
        return model
    return train_full(cfg, dataset, 'b', arm, outer, seed, device, builder, cache=cache, published=published, with_spatial=False, pretrain_pool='train_partition_prototypes', **kwargs)

def job_matrix(dataset: str, seed: int=42) -> list[dict]:
    return [{'dataset': dataset, 'stage': 'task', 'route': 'b', 'arm': arm, 'outer': outer, 'seed': seed} for arm in route_arms('b') for outer in range(int(DATASET_SPEC[dataset]['folds']))]
