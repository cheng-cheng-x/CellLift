from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.geometry_baselines.io_utils import atomic_json
from .evaluation import evaluate
from .protocol import ARMS, PROTOCOL_ID

def build_report(cfg: Mapping[str, Any], *, seed: int=42) -> dict[str, Any]:
    evaluation = evaluate(cfg, seed=seed, arms=tuple(ARMS))
    root = Path(cfg['paths']['result_root']) / 'statistics' / f'seed_{seed}'
    families = {}
    for family in ('residual', 'raw'):
        path = root / family / 'summary.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        families[family] = json.loads(path.read_text(encoding='utf-8'))
    residual = {row['comparison']: row for row in families['residual']['comparisons']}
    raw = {row['comparison']: row for row in families['raw']['comparisons']}
    gate = residual['C3-C1']['ci_low'] > 0 and residual['C3-C3S']['ci_low'] > 0 or (raw['C5-C1']['ci_low'] > 0 and raw['C5-C5S']['ci_low'] > 0)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'seed': seed, 'metric_name': 'QWK' if cfg['dataset'] == 'sicapv2' else 'five_fold_mean_patient_AUROC', 'arms': evaluation['metrics'], 'statistics': families, 'screening_gate_passed': gate, 'five_seed_confirmation_launched': False, 'stop_reason': None if gate else 'feature fusion did not make real 3D exceed both matched 2D and shuffled-3D controls', 'validation_only': True, 'official_test_touched': False}
    atomic_json(Path(cfg['paths']['result_root']) / 'report' / 'seed_42_screening.json', payload)
    return payload
