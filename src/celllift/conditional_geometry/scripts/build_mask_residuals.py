from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData, load_mask_prepared_fold
from celllift.conditional_geometry.common.probe_data import RunningMoments
from celllift.conditional_geometry.mask_protocol import INNER_FOLDS, MASK_PROTOCOL_ID, TARGET_COLUMNS, TARGET_DIM
from celllift.conditional_geometry.models.mask_probe import MaskOnly3DProbe
from celllift.conditional_geometry.scripts.fit_probe import _atomic_json, _sha256

def _load_model(path: Path, cfg: Mapping[str, Any], device: torch.device) -> MaskOnly3DProbe:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('protocol_id') != MASK_PROTOCOL_ID:
        raise RuntimeError(f'mask probe checkpoint protocol mismatch: {path}')
    model = MaskOnly3DProbe(int(cfg['probe']['hidden_dim']), float(cfg['probe']['dropout']))
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval()

@torch.no_grad()
def _predict(model: MaskOnly3DProbe, graph_np: np.ndarray, rays_np: np.ndarray, context: torch.Tensor, device: torch.device) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for start in range(0, len(graph_np), 16384):
        graph = torch.as_tensor(graph_np[start:start + 16384], device=device)
        rays = torch.as_tensor(rays_np[start:start + 16384], device=device)
        pieces.append(model(rays, context[graph]).float().cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty((0, TARGET_DIM), np.float32)

def build_mask_fold_residuals(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', graph_limit: int | None=None) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    target = torch.device(device)
    result_root, data_root = (Path(cfg['paths']['result_root']), Path(cfg['paths']['data_root']))
    if graph_limit is not None:
        tag = f'graph_limit_{int(graph_limit)}'
        result_root, data_root = (result_root / 'smoke' / tag, data_root / 'smoke' / tag)
    result_fold = result_root / 'probe' / f'fold_{fold:02d}'
    metrics_path = result_fold / 'metrics.json'
    if not metrics_path.is_file():
        raise RuntimeError('fit_mask_probe must PASS before residual construction')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('status') != 'PASS' or metrics.get('protocol_id') != MASK_PROTOCOL_ID:
        raise RuntimeError('mask probe metrics gate is not a compatible PASS')
    output_root = data_root / 'residuals' / f'fold_{fold:02d}'
    manifest_path = output_root / 'manifest.json'
    if manifest_path.is_file():
        completed = json.loads(manifest_path.read_text(encoding='utf-8'))
        files = completed.get('files', [])
        compatible = completed.get('status') == 'PASS' and completed.get('protocol_id') == MASK_PROTOCOL_ID and (completed.get('dataset') == cfg['dataset']) and (int(completed.get('fold', -1)) == int(fold)) and (completed.get('official_test_touched') is False) and (completed.get('probe_metrics_sha256') == _sha256(metrics_path)) and bool(files) and all((Path(str(item.get('path', ''))).is_file() and item.get('sha256') == _sha256(Path(str(item['path']))) for item in files))
        if not compatible:
            raise RuntimeError(f'incompatible completed mask residual cache blocks resume: {manifest_path}')
        return completed
    data = MaskFoldProbeData(cfg, fold, graph_limit=graph_limit)
    prepared = data_root / 'prepared' / f'fold_{fold:02d}'
    load_mask_prepared_fold(data, prepared)
    context = torch.as_tensor(data.transformed_context(), device=target)
    final = _load_model(result_fold / 'final.pt', cfg, target)
    inner = {index: _load_model(result_fold / f'inner_{index}.pt', cfg, target) for index in range(int(cfg['probe'].get('inner_folds', INNER_FOLDS)))}
    output_root.mkdir(parents=True, exist_ok=True)
    geometry_moments, ncr_moments = (RunningMoments(8), RunningMoments(1))
    outputs: list[dict[str, Any]] = []
    train_rows = val_rows = 0
    for shard_index, batch in enumerate(data.iter_transformed()):
        prediction = np.empty_like(batch.targets)
        validation, training = (batch.roles == 'validation', batch.roles == 'train')
        if validation.any():
            prediction[validation] = _predict(final, batch.graph_indices[validation], batch.anchor_2d[validation], context, target)
        for inner_fold, model in inner.items():
            selected = training & (batch.inner_folds == inner_fold)
            if selected.any():
                prediction[selected] = _predict(model, batch.graph_indices[selected], batch.anchor_2d[selected], context, target)
        residual = batch.targets - prediction
        residual[~batch.target_valid[:, 8], 8] = 0.0
        if not np.isfinite(residual).all():
            raise RuntimeError('non-finite mask cross-fitted residual')
        geometry_moments.update(residual[training, :8])
        ncr_moments.update(residual[training & batch.target_valid[:, 8], 8:9])
        train_rows += int(training.sum())
        val_rows += int(validation.sum())
        rows: dict[str, Any] = {'graph_id': pa.array([str(value) for value in batch.graph_ids]), 'anchor_id': pa.array(batch.anchor_ids), 'role': pa.array([str(value) for value in batch.roles]), 'inner_fold': pa.array(batch.inner_folds.astype(np.int8)), 'valid_ncr3d': pa.array(batch.target_valid[:, 8])}
        for column, name in enumerate(TARGET_COLUMNS):
            rows[f'residual_{name}'] = pa.array(residual[:, column].astype(np.float32))
        destination = output_root / f'residual_{shard_index:04d}.parquet'
        temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
        pq.write_table(pa.table(rows), temporary, compression='zstd')
        os.replace(temporary, destination)
        outputs.append({'path': str(destination), 'rows': len(batch.targets), 'sha256': _sha256(destination)})
    geometry_mean, geometry_std = geometry_moments.finish()
    ncr_mean, ncr_std = ncr_moments.finish()
    payload = {'status': 'PASS', 'protocol_id': MASK_PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': int(fold), 'official_test_touched': False, 'training_residual_route': 'patient-grouped inner-fold OOF RGB+mask prediction' if str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays' else 'patient-grouped inner-fold OOF mask-only prediction', 'validation_residual_route': 'final RGB+mask probe trained on all outer-training graphs' if str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays' else 'final mask-only probe trained on all outer-training graphs', 'training_rows': train_rows, 'validation_rows': val_rows, 'residual_mean_training_only': np.concatenate((geometry_mean, ncr_mean)).tolist(), 'residual_std_training_only': np.concatenate((geometry_std, ncr_std)).tolist(), 'target_columns': list(TARGET_COLUMNS), 'files': outputs, 'probe_metrics': str(metrics_path), 'probe_metrics_sha256': _sha256(metrics_path)}
    _atomic_json(manifest_path, payload)
    return payload
