from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from .io_utils import atomic_json
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS, require_test_gate

def _inventory(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(cfg['paths']['result_root'])
    jobs = list(root.rglob('job.json')) if root.exists() else []
    pass_jobs = 0
    from celllift.runtime import json
    failures = []
    for path in jobs:
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            failures.append({'path': str(path), 'error': str(error)})
            continue
        if value.get('status') == 'PASS':
            pass_jobs += 1
        else:
            failures.append({'path': str(path), 'status': value.get('status')})
    return {'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'jobs': len(jobs), 'pass_jobs': pass_jobs, 'non_pass_jobs': failures, 'paper_jobs_expected': int(cfg['split']['validation_folds']) * len(SEEDS), 'geometry_jobs_expected': int(cfg['split']['validation_folds']) * len(SEEDS) * len(ENCODERS) * 10, 'registered_arms': list(ARMS), 'validation_summary_exists': (root / 'metrics/validation/summary.json').is_file(), 'test_gate_exists': (root / 'selection_frozen.json').is_file()}

def run_stage(stage: str, cfg: Mapping[str, Any], args: Any) -> dict[str, Any]:
    if stage == 'predict_paper_rgb':
        from .paper_rgb import write_paper_fidelity
        result = _inventory(cfg)
        result['fidelity'] = write_paper_fidelity(str(cfg['dataset']), cfg['paths']['result_root'])
        result.update({'status': 'PASS', 'action': 'reuse_atomic_oof_predictions'})
        return result
    if stage == 'report':
        from .reporting import render_report
        inventory = _inventory(cfg)
        destination = Path(cfg['paths']['result_root']) / 'runtime' / 'inventory.json'
        atomic_json(destination, inventory)
        report = render_report(cfg)
        return {'status': 'PASS', 'inventory': inventory, 'report': report}
    if stage in {'predict_test', 'evaluate_test'}:
        gate = require_test_gate(cfg['paths']['result_root'])
        from .official_test import run_official_stage
        return run_official_stage(stage, cfg, args, gate)
    raise ValueError(stage)
