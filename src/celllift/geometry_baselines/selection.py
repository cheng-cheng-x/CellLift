from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .io_utils import atomic_json, sha256
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS

def _json(path: Path) -> dict[str, Any]:
    from celllift.runtime import json
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))

def _artifact(path: str | Path, expected: str, context: Path) -> None:
    value = Path(path)
    if not value.is_file() or sha256(value) != expected:
        raise RuntimeError(f'validation artifact checksum mismatch: {context}: {value}')

def _verify_paper_job(job: Mapping[str, Any], path: Path) -> None:
    _artifact(job['checkpoint'], job['checkpoint_sha256'], path)
    _artifact(job['predictions'], job['predictions_sha256'], path)

def _verify_geometry_job(job: Mapping[str, Any], path: Path) -> None:
    outputs = job.get('outputs', {})
    for name in ('checkpoint', 'predictions', 'metrics'):
        value = outputs.get(name, {})
        _artifact(value['path'], value['sha256'], path)

def _verify_tile_job(job: Mapping[str, Any], path: Path) -> None:
    if job.get('deterministic_job_seed') != job.get('seed'):
        raise RuntimeError(f'tile job lacks deterministic per-job initialization provenance: {path}')
    _artifact(job['checkpoint'], job['checkpoint_sha256'], path)
    _artifact(job['predictions'], job['predictions_sha256'], path)

def freeze_selection(cfg: Mapping[str, Any]) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    public_root = result_root.parent
    sibling = 'tcga_crc_msi' if cfg['dataset'] == 'sicapv2' else 'sicapv2'
    sibling_validation = public_root / sibling / 'metrics' / 'validation' / 'summary.json'
    sibling_value = _json(sibling_validation)
    if sibling_value.get('status') != 'PASS' or sibling_value.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError(f'cross-dataset TEST lock: {sibling} validation is incomplete')
    crc_tile_cross = public_root / 'tcga_crc_msi' / 'metrics' / 'validation_tile' / 'summary.json'
    crc_tile_value = _json(crc_tile_cross)
    if crc_tile_value.get('status') != 'PASS' or crc_tile_value.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('cross-dataset TEST lock: CRC tile validation is incomplete')
    folds = range(int(cfg['split']['validation_folds']))
    paper_values = []
    evidence = []
    evidence.extend([{'path': str(sibling_validation), 'sha256': sha256(sibling_validation)}, {'path': str(crc_tile_cross), 'sha256': sha256(crc_tile_cross)}])
    fidelity = result_root / 'paper_baseline' / 'fidelity.json'
    fidelity_value = _json(fidelity)
    if fidelity_value.get('status') != 'PASS' or fidelity_value.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('paper fidelity manifest is absent or incompatible')
    if cfg['dataset'] == 'tcga_crc_msi':
        weights = Path(fidelity_value['weights_cache'])
        if not weights.is_file() or sha256(weights) != fidelity_value.get('weights_sha256'):
            raise RuntimeError('paper ImageNet initialization weights changed after fidelity capture')
    elif int(fidelity_value.get('parameters', -1)) != 630276:
        raise RuntimeError('paper FSConv parameter fidelity changed')
    evidence.append({'path': str(fidelity), 'sha256': sha256(fidelity)})
    for fold in folds:
        for seed in SEEDS:
            path = result_root / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}' / 'job.json'
            job = _json(path)
            if job.get('status') != 'PASS':
                raise RuntimeError(f'paper job is not PASS: {path}')
            _verify_paper_job(job, path)
            evidence.append({'path': str(path), 'sha256': sha256(path)})
            if seed == 42:
                paper_values.append(float(job['outer_validation_metric']))
    paper_mean = float(np.mean(paper_values))
    lower = 0.6863 if cfg['dataset'] == 'sicapv2' else 0.7427
    upper = 0.7793 if cfg['dataset'] == 'sicapv2' else float('inf')
    if not lower <= paper_mean <= upper:
        raise RuntimeError(f'seed42 paper baseline gate failed: {paper_mean} not in [{lower},{upper}]')
    for encoder in ENCODERS:
        for arm in [key for key in ARMS if key.startswith('O')]:
            for seed in SEEDS:
                for fold in folds:
                    path = result_root / 'geometry_only' / 'screen' / encoder / arm / f'seed_{seed}' / f'fold_{fold}' / 'job.json'
                    job = _json(path)
                    if job.get('status') != 'PASS':
                        raise RuntimeError(f'geometry job is not PASS: {path}')
                    _verify_geometry_job(job, path)
                    evidence.append({'path': str(path), 'sha256': sha256(path)})
    validation = result_root / 'metrics' / 'validation' / 'summary.json'
    if _json(validation).get('status') != 'PASS':
        raise RuntimeError('validation summary is absent or failed')
    qc = result_root / 'metrics' / 'qc' / 'ncr_invalid_train_only.json'
    if _json(qc).get('status') != 'PASS':
        raise RuntimeError('TRAIN-only NCR QC is absent or failed')
    evidence.append({'path': str(qc), 'sha256': sha256(qc)})
    if cfg['dataset'] == 'tcga_crc_msi':
        tile_summary = result_root / 'metrics' / 'validation_tile' / 'summary.json'
        if _json(tile_summary).get('status') != 'PASS':
            raise RuntimeError('CRC tile-fusion validation is incomplete')
        for encoder in ENCODERS:
            for arm in [key for key in ARMS if key.startswith('O')]:
                for seed in SEEDS:
                    for fold in folds:
                        path = result_root / 'tile_fusion' / 'experts' / encoder / arm / f'seed_{seed}' / f'fold_{fold}' / 'job.json'
                        job = _json(path)
                        if job.get('status') != 'PASS':
                            raise RuntimeError(f'tile geometry job is not PASS: {path}')
                        _verify_tile_job(job, path)
                        evidence.append({'path': str(path), 'sha256': sha256(path)})
        evidence.append({'path': str(tile_summary), 'sha256': sha256(tile_summary)})
    evidence.append({'path': str(validation), 'sha256': sha256(validation)})
    gate = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'validation_complete': True, 'paper_rgb_gates_passed': True, 'cross_dataset_validation_complete': True, 'paper_seed42_mean': paper_mean, 'paper_gate': [lower, upper], 'arms': list(ARMS), 'encoders': list(ENCODERS), 'seeds': list(SEEDS), 'families_frozen': {'primary': ['R5-R1', 'R5-R5S', 'R4-R0', 'R4-R4S'], 'direct_3d_vs_2d': ['R4-R1', 'R2-R1', 'O4-O1', 'O2-O1']}, 'evidence': evidence}
    atomic_json(result_root / 'selection_frozen.json', gate)
    return gate
