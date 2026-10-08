from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import socket
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch.utils.data import DataLoader
from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData, collect_mask_probe_sample, load_mask_prepared_fold, save_mask_prepared_fold
from celllift.conditional_geometry.common.probe_data import ProbeSample
from celllift.conditional_geometry.mask_protocol import INNER_FOLDS, MASK_PROTOCOL_ID, PROBE_SEED
from celllift.conditional_geometry.models.mask_probe import MaskOnly3DProbe
from celllift.conditional_geometry.scripts.fit_probe import R2Accumulator, _atomic_json, _atomic_torch, _loader, _seed, _sha256

def _loss_epoch(model: MaskOnly3DProbe, loader: DataLoader, context: torch.Tensor, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = (0.0, 0)
    for graph, rays, target, valid in loader:
        graph = graph.to(device, non_blocking=True)
        rays = rays.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(rays, context[graph])
        loss = torch.square(prediction - target).masked_select(valid).mean()
        if training:
            loss.backward()
            optimizer.step()
        valid_count = int(valid.sum().item())
        total += float(loss.detach()) * valid_count
        count += valid_count
    return total / max(1, count)

def _train_one(sample: ProbeSample, train_mask: np.ndarray, val_mask: np.ndarray | None, context: torch.Tensor, spec: Mapping[str, Any], device: torch.device, seed: int, fixed_epochs: int | None=None) -> tuple[MaskOnly3DProbe, dict[str, Any]]:
    _seed(seed)
    model = MaskOnly3DProbe(int(spec['hidden_dim']), float(spec['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec['learning_rate']), weight_decay=float(spec['weight_decay']))
    train_loader = _loader(sample, train_mask, int(spec['batch_size']), seed, True)
    val_loader = None if val_mask is None else _loader(sample, val_mask, int(spec['batch_size']), seed + 1, False)
    best_state, best_loss, best_epoch, stale = (None, float('inf'), -1, 0)
    history: list[dict[str, float | int]] = []
    epochs = int(fixed_epochs if fixed_epochs is not None else spec['max_epochs'])
    for epoch in range(epochs):
        train_loss = _loss_epoch(model, train_loader, context, device, optimizer)
        val_loss = train_loss if val_loader is None else _loss_epoch(model, val_loader, context, device, None)
        history.append({'epoch': epoch, 'train_mse': train_loss, 'validation_mse': val_loss})
        if val_loader is not None and val_loss < best_loss - 1e-07:
            best_loss, best_epoch, stale = (val_loss, epoch, 0)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif val_loader is not None:
            stale += 1
            if stale >= int(spec['patience']):
                break
    if val_loader is not None:
        if best_state is None:
            raise RuntimeError('mask probe did not produce an inner-validation checkpoint')
        model.load_state_dict(best_state)
    else:
        best_epoch, best_loss = (epochs - 1, history[-1]['train_mse'])
    return (model, {'best_epoch': int(best_epoch), 'best_loss': float(best_loss), 'history': history})

@torch.no_grad()
def evaluate_outer_validation(data: MaskFoldProbeData, model: MaskOnly3DProbe, device: torch.device) -> dict[str, Any]:
    context = torch.as_tensor(data.transformed_context(), device=device)
    model.eval()
    accumulator = R2Accumulator(len(data.graph_ids))
    for batch in data.iter_transformed():
        selected = batch.roles == 'validation'
        if not selected.any():
            continue
        graph_np = batch.graph_indices[selected]
        rays_np = batch.anchor_2d[selected]
        predictions: list[np.ndarray] = []
        for start in range(0, len(graph_np), 16384):
            graph = torch.as_tensor(graph_np[start:start + 16384], device=device)
            rays = torch.as_tensor(rays_np[start:start + 16384], device=device)
            predictions.append(model(rays, context[graph]).float().cpu().numpy())
        accumulator.update(graph_np, batch.targets[selected], np.concatenate(predictions), batch.target_valid[selected])
    return accumulator.finish()

def fit_mask_fold_probe(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', graph_limit: int | None=None, sample_limit: int | None=None) -> dict[str, Any]:
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    result_root, data_root = (Path(cfg['paths']['result_root']), Path(cfg['paths']['data_root']))
    if graph_limit is not None:
        tag = f'graph_limit_{int(graph_limit)}'
        result_root, data_root = (result_root / 'smoke' / tag, data_root / 'smoke' / tag)
    output = result_root / 'probe' / f'fold_{fold:02d}'
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / 'metrics.json'
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding='utf-8'))
        checkpoint = Path(str(completed.get('final_checkpoint', '')))
        compatible = completed.get('status') == 'PASS' and completed.get('protocol_id') == MASK_PROTOCOL_ID and (completed.get('dataset') == cfg['dataset']) and (int(completed.get('fold', -1)) == int(fold)) and (completed.get('official_test_touched') is False) and checkpoint.is_file() and (completed.get('final_checkpoint_sha256') == _sha256(checkpoint))
        if not compatible:
            raise RuntimeError(f'incompatible completed mask probe blocks resume: {completed_path}')
        return completed
    data = MaskFoldProbeData(cfg, fold, graph_limit=graph_limit)
    prepared = data_root / 'prepared' / f'fold_{fold:02d}'
    if all(((prepared / name).is_file() for name in ('statistics.json', 'context_raw.npy', 'manifest.json'))):
        load_mask_prepared_fold(data, prepared)
    else:
        data.prepare()
        save_mask_prepared_fold(data, prepared)
    spec = cfg['probe']
    sample = collect_mask_probe_sample(data, int(sample_limit or spec['sample_anchors']), int(spec.get('seed', PROBE_SEED)))
    context = torch.as_tensor(data.transformed_context(), device=target)
    inner_records: dict[str, Any] = {}
    best_epochs: list[int] = []
    for inner in range(int(spec.get('inner_folds', INNER_FOLDS))):
        train_mask, val_mask = (sample.inner_folds != inner, sample.inner_folds == inner)
        checkpoint = output / f'inner_{inner}.pt'
        if checkpoint.is_file():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if previous.get('protocol_id') != MASK_PROTOCOL_ID or int(previous.get('fold', -1)) != fold or int(previous.get('inner_fold', -1)) != inner:
                raise RuntimeError(f'incompatible mask inner checkpoint blocks resume: {checkpoint}')
            model = MaskOnly3DProbe(int(spec['hidden_dim']), float(spec['dropout'])).to(target)
            model.load_state_dict(previous['model'], strict=True)
            record, resumed = (previous['record'], True)
        else:
            model, record = _train_one(sample, train_mask, val_mask, context, spec, target, int(spec['seed']) + fold * 101 + inner)
            _atomic_torch(checkpoint, {'protocol_id': MASK_PROTOCOL_ID, 'fold': fold, 'inner_fold': inner, 'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'record': record})
            resumed = False
        best_epochs.append(int(record['best_epoch']) + 1)
        inner_records[str(inner)] = {'best_epoch': int(record['best_epoch']), 'best_loss': float(record['best_loss']), 'epochs_run': len(record['history']), 'checkpoint': str(checkpoint), 'sha256': _sha256(checkpoint), 'train_anchors': int(train_mask.sum()), 'validation_anchors': int(val_mask.sum()), 'resumed_compatible_checkpoint': resumed}
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final, final_record = _train_one(sample, np.ones(len(sample.targets), bool), None, context, spec, target, int(spec['seed']) + fold * 101 + 97, fixed_epochs=final_epochs)
    final_path = output / 'final.pt'
    _atomic_torch(final_path, {'protocol_id': MASK_PROTOCOL_ID, 'fold': fold, 'model': {key: value.detach().cpu() for key, value in final.state_dict().items()}, 'epochs': final_epochs, 'record': final_record})
    payload = {'status': 'PASS', 'protocol_id': MASK_PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': int(fold), 'official_test_touched': False, 'conditioning': 'fold-specific frozen RGB512 + raw preprocessed nucleus-mask rays[36] + ray-derived graph summaries; no XY/edges/cell2D/NCR2D' if str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays' else 'raw preprocessed nucleus-mask rays[36] + graph mean/std/count derived only from those rays; no RGB/XY/edges/cell2D/NCR2D', 'sample_anchors': len(sample.targets), 'inner_folds': inner_records, 'final_epochs': final_epochs, 'final_checkpoint': str(final_path), 'final_checkpoint_sha256': _sha256(final_path), 'statistics_path': str(prepared / 'statistics.json'), 'statistics_sha256': _sha256(prepared / 'statistics.json'), 'prepared_manifest': str(prepared / 'manifest.json'), 'prepared_manifest_sha256': _sha256(prepared / 'manifest.json'), 'validation_r2': evaluate_outer_validation(data, final, target), 'runtime': {'host': socket.gethostname(), 'ended_at': datetime.now(timezone.utc).isoformat()}}
    _atomic_json(completed_path, payload)
    return payload
