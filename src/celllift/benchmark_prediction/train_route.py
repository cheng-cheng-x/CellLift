from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
from celllift.morphology_interaction.dataset import load_config as load_route_yaml
from celllift.morphology_interaction.run import _train_one
from .io import job_dir, reused, write_pass
from .paths import BATCH, PACKAGE, result_root

def _cfg(dataset: str, route: str) -> dict:
    cfg = load_route_yaml(str(PACKAGE.parent / 'morphology_interaction' / 'configs' / f'{dataset}.yaml'))
    cfg = dict(cfg)
    cfg['official_result_subdir'] = f'{BATCH}/routes/{route}'
    cfg['paths'] = dict(cfg['paths'])
    cfg['paths']['result_root'] = str(result_root().parent)
    return cfg

def train_route(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    route = job['route']
    cfg = _cfg(job['dataset'], route)
    payload = {'dataset': job['dataset'], 'route': route, 'arm': job['arm'], 'outer': int(job.get('fold') or 0), 'seed': int(job['seed']), 'official_fit': job.get('official_fit'), 'official_val': job.get('official_val'), 'official_predict': job.get('official_predict'), 'fixed_epochs': job.get('fixed_epochs')}
    result = _train_one(cfg, payload, device, int(job.get('max_epochs', 60)), int(job.get('patience', 10)))
    return write_pass(destination, {'wrapped': result, 'arm': job['arm'], 'route': route, 'dataset': job['dataset']})

def train_pretrain(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    cfg = _cfg(job['dataset'], job['route'])
    fold = job.get('fold')
    if job['dataset'] != 'sicapv2':
        fold = None
    if job['route'] == 'a':
        from celllift.morphology_interaction.route_a.pretrain import run_pretrain
    else:
        from celllift.morphology_interaction.route_d.pretrain import run_pretrain
    result = run_pretrain(cfg, job['target'], device, dataset=job['dataset'], fold=fold, official_fit=list(map(str, job.get('official_fit') or [])))
    return write_pass(destination, {'wrapped': result, 'arm': job['arm'], 'target': job['target'], 'pretrain_pool': result.get('pretrain_pool')})

def train_expert(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    from .train_geometry import train_geometry
    mapped = dict(job)
    mapped['family'] = 'expert'
    mapped['encoder'] = 'deepsets'
    mapped['arm'] = {'E2': 'G2', 'ES': 'G2R', 'ER': 'G2S'}.get(job['arm'], job['arm'])
    mapped['arm_tag'] = job['arm']
    return train_geometry(mapped, device)
