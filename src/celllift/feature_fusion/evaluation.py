from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.geometry_baselines.io_utils import atomic_json, atomic_parquet
from celllift.cross_fitted_correction.data import read_parquet
from .protocol import PROTOCOL_ID
from .training import metric

def load_arm(cfg: Mapping[str, Any], arm: str, seed: int) -> list[dict[str, Any]]:
    root = Path(cfg['paths']['result_root']) / 'screen' / arm / f'seed_{seed}'
    rows = []
    for fold in range(int(cfg['split']['validation_folds'])):
        rows.extend(read_parquet(root / f'fold_{fold}' / 'validation_predictions.parquet'))
    return rows

def evaluate(cfg: Mapping[str, Any], *, seed: int, arms: Sequence[str]) -> dict[str, Any]:
    table = []
    predictions = []
    for arm in arms:
        rows = load_arm(cfg, arm, seed)
        value = metric(cfg['dataset'], rows)
        paper = metric(cfg['dataset'], rows, logits_key='paper_logits')
        table.append({'arm_id': arm, 'seed': seed, 'metric': value, 'paper_rgb_metric': paper, 'delta_vs_paper_rgb': value - paper})
        predictions.extend([{'arm_id': arm, 'seed': seed, **row} for row in rows])
    output = Path(cfg['paths']['result_root']) / 'metrics' / f'seed_{seed}'
    atomic_parquet(output / 'metrics.parquet', table)
    atomic_parquet(output / 'predictions.parquet', predictions)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'validation_only': True, 'official_test_touched': False, 'metrics': table}
    atomic_json(output / 'summary.json', payload)
    return payload
