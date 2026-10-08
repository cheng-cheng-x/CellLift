from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet, atomic_torch
from .paths import baseline_result_root, comparison_result_root, result_root

def job_dir(job: Mapping[str, Any]) -> Path:
    arm = job.get('arm_tag') or job['arm']
    fold = int(job.get('fold') or 0)
    if job.get('r3'):
        root = comparison_result_root()
    elif job.get('baseline_qualification'):
        root = baseline_result_root()
    else:
        root = result_root()
    return root / job['dataset'] / job['family'] / arm / job.get('unit', 'u') / f"seed_{int(job['seed'])}" / f'fold_{fold:02d}'

def compatibility_job_dir(job: Mapping[str, Any]) -> Path:
    arm = job.get('arm_tag') or job['arm']
    fold = int(job.get('fold') or 0)
    return result_root() / job['dataset'] / job['family'] / arm / job.get('unit', 'u') / f"seed_{int(job['seed'])}" / f'fold_{fold:02d}'

def reused(destination: Path) -> dict[str, Any] | None:
    path = destination / 'job.json'
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if payload.get('status') in {'PASS', 'REUSED'}:
        return {**payload, 'reused': True}
    return None

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_pass(destination: Path, payload: Mapping[str, Any], predictions: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if predictions is not None:
        atomic_parquet(destination / 'predictions.parquet', predictions)
    manifest = {'status': 'PASS', **dict(payload)}
    atomic_json(destination / 'job.json', manifest)
    return manifest

def save_model(destination: Path, state: Mapping[str, Any]) -> None:
    atomic_torch(destination / 'best.pt', dict(state))
