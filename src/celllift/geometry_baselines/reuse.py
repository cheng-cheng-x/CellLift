from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
import shutil
from typing import Any, Mapping
from .io_utils import atomic_json, sha256
from .protocol import ARMS, PROTOCOL_ID, SEEDS
SOURCE_MAP = {'O1': ('set_encoder_mask_conditioning_mask_input', 'M3_MASK2D'), 'O3': ('set_encoder_raw_residual_comparison_full3d_experts', 'M5_MASK_FULL3D'), 'O3S': ('set_encoder_raw_residual_comparison_full3d_experts', 'M5_MASK_SHUF_FULL3D'), 'O5': ('set_encoder_mask_conditioning_mask_input', 'M3_MASK_RES3D'), 'O5S': ('set_encoder_mask_conditioning_mask_input', 'M3_MASK_SHUF_RES3D')}

def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)

def materialize_reuse(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    public_root = Path(cfg['paths']['result_root']).parents[1]
    destination_root = Path(cfg['paths']['result_root']) / 'geometry_only' / 'screen'
    folds = range(int(cfg['split']['validation_folds']))
    reused, missing = ([], [])
    for encoder in ('meanpool', 'deepsets'):
        for target_arm, (source_run, source_arm) in SOURCE_MAP.items():
            source_root = public_root / source_run / dataset / 'screen' / encoder / source_arm
            for seed in SEEDS:
                for fold in folds:
                    source = source_root / f'seed_{seed}' / f'fold_{fold}'
                    destination = destination_root / encoder / target_arm / f'seed_{seed}' / f'fold_{fold}'
                    source_job, destination_job = (source / 'job.json', destination / 'job.json')
                    if destination_job.is_file():
                        continue
                    if not source_job.is_file():
                        missing.append(str(source_job))
                        continue
                    job = json.loads(source_job.read_text(encoding='utf-8'))
                    if job.get('status') != 'PASS' or job.get('encoder') != encoder or int(job.get('seed', -1)) != seed or (int(job.get('fold', -1)) != fold):
                        raise RuntimeError(f'incompatible reusable geometry job: {source_job}')
                    artifacts = {}
                    for filename, key in (('best.pt', 'checkpoint'), ('validation_predictions.parquet', 'predictions'), ('metrics.json', 'metrics')):
                        source_artifact = source / filename
                        recorded = job['outputs'][key]['sha256']
                        if not source_artifact.is_file() or sha256(source_artifact) != recorded:
                            raise RuntimeError(f'reusable artifact checksum mismatch: {source_artifact}')
                        destination_artifact = destination / filename
                        _link_or_copy(source_artifact, destination_artifact)
                        artifacts[key] = {'path': str(destination_artifact), 'sha256': recorded}
                    payload = {**{key: value for key, value in job.items() if key not in {'arm', 'outputs', 'protocol_id', 'status'}}, 'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'arm': ARMS[target_arm].to_dict(), 'arm_id': target_arm, 'outputs': artifacts, 'reused': True, 'reused_from': str(source_job), 'source_protocol_id': job.get('protocol_id'), 'source_job_sha256': sha256(source_job), 'equivalence': 'same normalized 41D token values, dual independent 128D branches, optimizer/training and prediction; geometry_baselines changes only arm naming'}
                    atomic_json(destination_job, payload)
                    reused.append(str(destination_job))
    manifest = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'reused_jobs': len(reused), 'missing_source_jobs': len(missing), 'reused': reused, 'missing': missing}
    atomic_json(Path(cfg['paths']['result_root']) / 'geometry_only' / 'reuse_manifest.json', manifest)
    return manifest
