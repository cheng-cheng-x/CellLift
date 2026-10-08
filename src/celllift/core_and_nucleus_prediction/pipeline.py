from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import subprocess
import sys
from celllift.runtime import ResourcePath as Path
from .constants import CODE_ROOT, RECONSTRUCT_PYTHON, RESULT_ROOT
MODEL_INPUT = CODE_ROOT / 'model_inputs'
PYTHON = RECONSTRUCT_PYTHON
LOG = RESULT_ROOT / 'logs'

def _run(args: list[str], log_name: str, env: dict | None=None) -> dict:
    LOG.mkdir(parents=True, exist_ok=True)
    log = LOG / log_name
    command_env = os.environ.copy()
    command_env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    command_env.setdefault('OMP_NUM_THREADS', '1')
    command_env.setdefault('MKL_NUM_THREADS', '1')
    command_env.setdefault('TORCH_NUM_THREADS', '1')
    command_env['PYTHONPATH'] = str(CODE_ROOT) + os.pathsep + str(MODEL_INPUT) + (os.pathsep + command_env['PYTHONPATH'] if command_env.get('PYTHONPATH') else '')
    if env:
        command_env.update(env)
    with log.open('w', encoding='utf-8') as handle:
        code = subprocess.call(args, cwd=str(CODE_ROOT), env=command_env, stdout=handle, stderr=subprocess.STDOUT)
    return {'cmd': args, 'log': str(log), 'code': code}

def correction_cpu() -> dict:
    jobs = {'lizard-conditional_geometry': _run([PYTHON, '-m', 'arvaniti_lizard.run', 'lizard-conditional_geometry'], 'lizard_conditional_geometry.log'), 'arvaniti-official': _run([PYTHON, '-m', 'arvaniti_lizard.run', 'arvaniti-official'], 'arvaniti_official.log')}
    jobs['lizard-conditional_geometry-qc'] = _run([PYTHON, '-m', 'arvaniti_lizard.run', 'lizard-conditional_geometry-qc'], 'lizard_conditional_geometry_qc.log')
    jobs['conditional_geometry-graphs'] = _run([PYTHON, '-m', 'arvaniti_lizard.run', 'conditional_geometry-graphs', '--dataset', 'both'], 'conditional_geometry_graphs.log')
    jobs['reuse-conditional_geometry'] = _run([PYTHON, '-m', 'arvaniti_lizard.run', 'reuse-conditional_geometry', '--dataset', 'both'], 'reuse_conditional_geometry.log')
    (RESULT_ROOT / 'status').mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / 'status' / 'correction_cpu.json').write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    return jobs

def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else 'correction-cpu'
    if stage == 'correction-cpu':
        print(json.dumps(correction_cpu(), indent=2))
        return
    raise SystemExit(stage)
if __name__ == '__main__':
    main()
