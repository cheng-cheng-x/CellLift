from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .io_utils import atomic_json, sha256
from .protocol import PROTOCOL_ID

def evaluate_seed42_gate(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(cfg['paths']['result_root'])
    jobs, values = ([], [])
    for fold in range(int(cfg['split']['validation_folds'])):
        path = root / 'paper_baseline' / f'fold_{fold:02d}' / 'seed_42' / 'job.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        job = json.loads(path.read_text(encoding='utf-8'))
        if job.get('status') != 'PASS':
            raise RuntimeError(f'paper seed42 job is not PASS: {path}')
        values.append(float(job['outer_validation_metric']))
        jobs.append({'fold': fold, 'path': str(path), 'sha256': sha256(path), 'value': values[-1]})
    mean = float(np.mean(values))
    if cfg['dataset'] == 'sicapv2':
        lower, upper = (0.6863, 0.7793)
    else:
        lower, upper = (0.7427, float('inf'))
    passed = lower <= mean <= upper
    payload = {'status': 'PASS' if passed else 'FAIL', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'seed': 42, 'fold_values': values, 'mean': mean, 'accepted_interval': [lower, upper], 'jobs': jobs, 'official_test_touched': False}
    destination = root / 'paper_baseline' / 'seed42_gate.json'
    atomic_json(destination, payload)
    if not passed:
        raise RuntimeError(f'paper seed42 gate failed: {mean} not in [{lower}, {upper}]')
    return payload

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    from celllift.runtime import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    print(json.dumps(evaluate_seed42_gate(cfg), indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
