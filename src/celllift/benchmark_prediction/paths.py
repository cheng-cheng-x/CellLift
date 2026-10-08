from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import yaml
PACKAGE = Path(__file__).resolve().parent
PUBLIC = PACKAGE.parent
PYTHON = _resource_path('artifact_0008')
REMOTE_CODE = _resource_path('artifact_0009')
REMOTE_DATA = _resource_path('artifact_0010')
REMOTE_RESULT = _resource_path('artifact_0011')
BATCH = 'benchmark_prediction_seed42'
BASELINE_BATCH = 'baseline_qualification_seed42'
COMPARISON_BATCH = 'paired_analysis_seed42'

def load_yaml(path: str | Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)

def config_path(dataset: str) -> Path:
    return PACKAGE / 'configs' / f'{dataset}.yaml'

def load_config(dataset: str) -> dict[str, Any]:
    return load_yaml(config_path(dataset))

def split_root(cfg: Mapping[str, Any] | None=None) -> Path:
    if cfg and cfg.get('paths', {}).get('split_root'):
        return Path(cfg['paths']['split_root'])
    return Path(REMOTE_DATA) / 'benchmark_prediction' / 'splits'

def result_root(cfg: Mapping[str, Any] | None=None) -> Path:
    if cfg and cfg.get('paths', {}).get('result_root'):
        return Path(cfg['paths']['result_root'])
    return Path(REMOTE_RESULT) / 'benchmark_prediction' / BATCH

def baseline_result_root() -> Path:
    return Path(REMOTE_RESULT) / 'benchmark_prediction' / BASELINE_BATCH

def comparison_result_root() -> Path:
    return Path(REMOTE_RESULT) / 'benchmark_prediction' / COMPARISON_BATCH

def compatibility_result_root() -> Path:
    return Path(REMOTE_RESULT) / 'benchmark_prediction' / BATCH

def runtime_root(cfg: Mapping[str, Any] | None=None) -> Path:
    return result_root(cfg) / 'runtime'
