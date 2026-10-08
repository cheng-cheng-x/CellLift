from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import hashlib
from celllift.runtime import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
try:
    from celllift.runtime import yaml
except ModuleNotFoundError:
    yaml = None
STAGES = ['preflight', 'thumbnail', 'order_qc', 'global_pilot', 'global_register', 'global_qc', 'track_enumerate', 'local_pilot', 'edge_register', 'track_optimize', 'fine_qc', 'export', 'final_audit', 'report']
conditional_geometry_STAGES = ['reuse_set_encoding', 'hierarchical_pilot', 'parent_register', 'child_refine', 'edge_select', 'track_optimize', 'fine_qc', 'export', 'final_audit', 'report']
ALL_STAGES = list(dict.fromkeys([*STAGES, *conditional_geometry_STAGES]))
RESULT_DIRS = ['00_manifests', '01_preflight', '02_thumbnails_order_qc/thumbnails_10um', '02_thumbnails_order_qc/thumbnails_3p68um', '02_thumbnails_order_qc/masks_10um', '03_global_pilots', '04_global_registration/previews', '04_global_registration/transforms', '04_global_registration/qc', '05_track_candidates', '06_local_registration/jobs', '06_local_registration/edges', '06_local_registration/qc', '06_local_registration/parent_edges', '06_local_registration/child_edges', '06_local_registration/pilot', '07_track_optimization', '08_registered_layers', '09_triplets', '10_acceptance', 'logs', 'runtime', 'runtime/stop']

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config_text = config_path.read_text(encoding='utf-8')
    if config_path.suffix.lower() == '.json':
        cfg = json.loads(config_text)
    else:
        if yaml is None:
            raise RuntimeError('PyYAML is required for YAML controller configs; use the JSON worker config inside the VALIS container')
        cfg = yaml.safe_load(config_text)
    required = {'code_root', 'result_root', 'raw_root', 'observer1', 'observer2', 'valis_env_script', 'valis_image', 'physical', 'global', 'tracks', 'fine', 'runtime'}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f'configuration missing keys: {missing}')
    physical = cfg['physical']
    frozen = {'mpp_um_per_px': 0.46, 'section_spacing_um': 5.0, 'width_px': 14863, 'height_px': 22387, 'roi_px': 1024}
    for key, expected in frozen.items():
        if float(physical[key]) != float(expected):
            raise ValueError(f'frozen physical.{key} must be {expected}, got {physical[key]}')
    if int(cfg['global']['reference_section']) != 130:
        raise ValueError('global.reference_section is frozen at 130')
    runtime = cfg['runtime']
    compute_backend = str(runtime.get('compute_backend', 'gpu')).lower()
    if compute_backend == 'gpu':
        if int(runtime['gpu_workers']) != 4:
            raise ValueError('runtime.gpu_workers is frozen at 4 for the GPU backend')
    elif compute_backend == 'cpu':
        if int(runtime.get('gpu_workers', 0)) != 0:
            raise ValueError('runtime.gpu_workers must be 0 for the CPU backend')
        if int(runtime.get('cpu_workers_total', 0)) < 1:
            raise ValueError('runtime.cpu_workers_total must be positive for the CPU backend')
        if int(runtime.get('cpu_threads_per_worker', 0)) != 1:
            raise ValueError('runtime.cpu_threads_per_worker must be 1')
        cpu_nodes = [str(node) for node in runtime.get('cpu_nodes', [])]
        cpu_layout = {str(node): int(worker_count) for node, worker_count in runtime.get('cpu_worker_layout', {}).items()}
        if not cpu_nodes or set(cpu_nodes) != set(cpu_layout):
            raise ValueError('runtime.cpu_nodes must exactly match runtime.cpu_worker_layout keys')
        if any((worker_count < 1 for worker_count in cpu_layout.values())):
            raise ValueError('runtime.cpu_worker_layout values must be positive')
        if sum(cpu_layout.values()) != int(runtime['cpu_workers_total']):
            raise ValueError('runtime.cpu_worker_layout must sum to runtime.cpu_workers_total')
    else:
        raise ValueError(f'unsupported runtime.compute_backend: {compute_backend}')
    if str(cfg.get('pipeline_mode', 'set_encoding')) == 'hierarchical_conditional_geometry':
        hierarchical = cfg.get('hierarchical', {})
        if int(hierarchical.get('parent_context_px', 0)) != 8192:
            raise ValueError('hierarchical.parent_context_px must be 8192')
        if [int(value) for value in hierarchical.get('child_context_ladder_px', [])] != [2048, 3072, 4096]:
            raise ValueError('hierarchical.child_context_ladder_px must be [2048, 3072, 4096]')
        parent_layout = {str(node): int(count) for node, count in hierarchical.get('parent_worker_layout', {}).items()}
        child_layout = {str(node): int(count) for node, count in hierarchical.get('child_worker_layout', {}).items()}
        if set(parent_layout) != set(cpu_layout):
            raise ValueError('hierarchical.parent_worker_layout keys must equal runtime.cpu_worker_layout keys')
        if any((count < 1 for count in parent_layout.values())):
            raise ValueError('hierarchical.parent_worker_layout values must be positive')
        if child_layout != cpu_layout:
            raise ValueError('hierarchical.child_worker_layout must equal runtime.cpu_worker_layout')
    return cfg

def stages_for_config(cfg: dict[str, Any]) -> list[str]:
    if str(cfg.get('pipeline_mode', 'set_encoding')) == 'hierarchical_conditional_geometry':
        return conditional_geometry_STAGES
    return STAGES

def result_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg['result_root'])

def ensure_layout(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    for rel in RESULT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def write_tsv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str] | None=None) -> None:
    path = Path(path)
    records = list(rows)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in records:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
        temporary = Path(handle.name)
    os.replace(temporary, path)

def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, path)

def read_json(path: str | Path, default: Any=None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))

def stage_status(cfg: dict[str, Any], stage: str, status: str, **details: Any) -> None:
    path = result_root(cfg) / '00_manifests/stage_status.json'
    payload = read_json(path, {})
    payload[stage] = {'status': status, 'updated_utc': utc_now(), **details}
    write_json(path, payload)

def stage_complete(cfg: dict[str, Any], stage: str) -> bool:
    row = read_json(result_root(cfg) / '00_manifests/stage_status.json', {}).get(stage, {})
    return row.get('status') in {'PASS', 'COMPLETE', 'PASS_WITH_TERMINAL_FAILURES'}

def sha256_file(path: str | Path, block_size: int=16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while (block := handle.read(block_size)):
            digest.update(block)
    return digest.hexdigest()

def run_logged(command: list[str], log_path: str | Path, env: dict[str, str] | None=None) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        raise RuntimeError(f'command failed with exit {completed.returncode}; see {log_path}')

def valis_shell(cfg: dict[str, Any], script: Path, args: list[str]) -> list[str]:
    import shlex
    threads = int(cfg['runtime'].get('cpu_threads_per_worker', 1))
    vips_concurrency = int(cfg['runtime'].get('vips_concurrency', 1))
    vips_access = str(cfg['runtime'].get('vips_access', 'sequential'))
    exports = ' '.join([f'export OMP_NUM_THREADS={threads};', f'export OPENBLAS_NUM_THREADS={threads};', f'export MKL_NUM_THREADS={threads};', f'export NUMEXPR_NUM_THREADS={threads};', f'export VIPS_CONCURRENCY={vips_concurrency};', f'export REGROI_VIPS_ACCESS={shlex.quote(vips_access)};', f'export SINGULARITYENV_OMP_NUM_THREADS={threads};', f'export SINGULARITYENV_OPENBLAS_NUM_THREADS={threads};', f'export SINGULARITYENV_MKL_NUM_THREADS={threads};', f'export SINGULARITYENV_NUMEXPR_NUM_THREADS={threads};', f'export SINGULARITYENV_VIPS_CONCURRENCY={vips_concurrency};', f'export SINGULARITYENV_REGROI_VIPS_ACCESS={shlex.quote(vips_access)};'])
    shell = ' && '.join([f"source {shlex.quote(cfg['valis_env_script'])}", exports + ' ' + ' '.join(['singularity exec --cleanenv', _public_resource('artifact_0064'), shlex.quote(cfg['valis_image']), shlex.quote(cfg.get('valis_python', _public_resource('artifact_0065'))), shlex.quote(str(script)), *[shlex.quote(str(value)) for value in args]])])
    return ['bash', '-lc', shell]
