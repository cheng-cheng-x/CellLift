from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import torch
from ..dataset import DATASET_SPEC, route_arms
from ..foundation.loop import train_full
from .models import RouteD
from .pretrain import pretrain_dir

def build_model(**kwargs):
    return RouteD(**kwargs)

def train_outer(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, seed: int, device: str, **kwargs) -> dict[str, Any]:
    target = '3d' if arm == 'D3' else '2d'
    init = None
    start = kwargs.pop('init_path', None)
    path = Path(start) if start else pretrain_dir(cfg, target, fold=outer) / 'best.pt'
    if not path.is_file():
        path = pretrain_dir(cfg, target) / 'best.pt'
    if path.is_file():
        payload = torch.load(path, map_location=device, weights_only=False)
        init = payload.get('model')
    extra = {key: kwargs.pop(key) for key in ('official_fit', 'official_val', 'official_predict', 'fixed_epochs') if key in kwargs}
    return train_full(cfg, dataset, 'd', arm, outer, seed, device, build_model, with_spatial=True, init_state=init, pretrain_pool=str(kwargs.pop('pretrain_pool', 'paired_adapter')), **extra, **kwargs)

def job_matrix(dataset: str, seed: int=42) -> list[dict]:
    return [{'dataset': dataset, 'stage': 'task', 'route': 'd', 'arm': arm, 'outer': outer, 'seed': seed} for arm in route_arms('d') for outer in range(int(DATASET_SPEC[dataset]['folds']))]
