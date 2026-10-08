from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from ..dataset import route_arms
from ..foundation.loop import train_full
from .models import RouteA

def build_model(**kwargs):
    return RouteA(**kwargs)

def train_outer(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, seed: int, device: str, **kwargs) -> dict[str, Any]:
    init = None
    start = kwargs.pop('init_path', None)
    if start:
        import torch
        payload = torch.load(start, map_location=device, weights_only=False)
        init = payload.get('model')
    extra = {key: kwargs.pop(key) for key in ('official_fit', 'official_val', 'official_predict', 'fixed_epochs') if key in kwargs}
    return train_full(cfg, dataset, 'a', arm, outer, seed, device, build_model, with_spatial=True, init_state=init, pretrain_pool=str(kwargs.pop('pretrain_pool', 'task_or_upstream')), **extra, **kwargs)

def job_matrix(dataset: str, seed: int=42) -> list[dict]:
    from ..dataset import DATASET_SPEC
    return [{'dataset': dataset, 'stage': 'task', 'route': 'a', 'arm': arm, 'outer': outer, 'seed': seed} for arm in route_arms('a') for outer in range(int(DATASET_SPEC[dataset]['folds']))]
