from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
import random
import socket
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from celllift.conditional_geometry.common.probe_data import FoldProbeData, ProbeSample, collect_probe_sample, load_prepared_fold, save_prepared_fold
from celllift.conditional_geometry.models import Conditional3DProbe
from celllift.conditional_geometry.protocol import INNER_FOLDS, PROBE_SEED, PROTOCOL_ID, TARGET_COLUMNS, TARGET_DIM

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def _seed(seed: int) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def _loader(sample: ProbeSample, mask: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(sample.graph_indices[mask], dtype=torch.long), torch.as_tensor(sample.anchor_2d[mask], dtype=torch.float32), torch.as_tensor(sample.targets[mask], dtype=torch.float32), torch.as_tensor(sample.target_valid[mask], dtype=torch.bool))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True)

def _loss_epoch(model: Conditional3DProbe, loader: DataLoader, rgb: torch.Tensor, context: torch.Tensor, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = (0.0, 0)
    for graph, anchor, target, valid in loader:
        graph = graph.to(device, non_blocking=True)
        anchor = anchor.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(rgb[graph], anchor, context[graph])
        squared = torch.square(prediction - target)
        loss = squared.masked_select(valid).mean()
        if training:
            loss.backward()
            optimizer.step()
        valid_count = int(valid.sum().item())
        total += float(loss.detach()) * valid_count
        count += valid_count
    return total / max(1, count)

def _train_one(sample: ProbeSample, train_mask: np.ndarray, val_mask: np.ndarray | None, rgb: torch.Tensor, context: torch.Tensor, spec: Mapping[str, Any], device: torch.device, seed: int, fixed_epochs: int | None=None) -> tuple[Conditional3DProbe, dict[str, Any]]:
    _seed(seed)
    model = Conditional3DProbe(int(spec['hidden_dim']), float(spec['dropout'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec['learning_rate']), weight_decay=float(spec['weight_decay']))
    train_loader = _loader(sample, train_mask, int(spec['batch_size']), seed, True)
    val_loader = None if val_mask is None else _loader(sample, val_mask, int(spec['batch_size']), seed + 1, False)
    best_state, best_loss, best_epoch, stale = (None, float('inf'), -1, 0)
    history: list[dict[str, float | int]] = []
    epochs = int(fixed_epochs if fixed_epochs is not None else spec['max_epochs'])
    for epoch in range(epochs):
        train_loss = _loss_epoch(model, train_loader, rgb, context, device, optimizer)
        val_loss = train_loss if val_loader is None else _loss_epoch(model, val_loader, rgb, context, device, None)
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
            raise RuntimeError('probe did not produce an inner-validation checkpoint')
        model.load_state_dict(best_state)
    else:
        best_epoch, best_loss = (epochs - 1, history[-1]['train_mse'])
    return (model, {'best_epoch': int(best_epoch), 'best_loss': float(best_loss), 'history': history})

class R2Accumulator:

    def __init__(self, graph_count: int) -> None:
        self.count = np.zeros(TARGET_DIM, np.int64)
        self.sum_y = np.zeros(TARGET_DIM, np.float64)
        self.sum_y2 = np.zeros(TARGET_DIM, np.float64)
        self.sse = np.zeros(TARGET_DIM, np.float64)
        self.graph_count = np.zeros((graph_count, TARGET_DIM), np.int64)
        self.graph_y = np.zeros((graph_count, TARGET_DIM), np.float64)
        self.graph_y2 = np.zeros((graph_count, TARGET_DIM), np.float64)
        self.graph_sse = np.zeros((graph_count, TARGET_DIM), np.float64)

    def update(self, graph: np.ndarray, target: np.ndarray, prediction: np.ndarray, valid: np.ndarray | None=None) -> None:
        if valid is None:
            valid = np.ones_like(target, dtype=bool)
        error2 = np.square(target.astype(np.float64) - prediction.astype(np.float64))
        y2 = np.square(target.astype(np.float64))
        self.count += valid.sum(axis=0)
        self.sum_y += np.where(valid, target, 0).sum(axis=0, dtype=np.float64)
        self.sum_y2 += np.where(valid, y2, 0).sum(axis=0)
        self.sse += np.where(valid, error2, 0).sum(axis=0)
        for column in range(TARGET_DIM):
            selected = valid[:, column]
            graph_selected = graph[selected]
            width = self.graph_count.shape[0]
            self.graph_count[:, column] += np.bincount(graph_selected, minlength=width)
            self.graph_y[:, column] += np.bincount(graph_selected, weights=target[selected, column], minlength=width)
            self.graph_y2[:, column] += np.bincount(graph_selected, weights=y2[selected, column], minlength=width)
            self.graph_sse[:, column] += np.bincount(graph_selected, weights=error2[selected, column], minlength=width)

    def finish(self) -> dict[str, Any]:
        denominator = self.sum_y2 - np.square(self.sum_y) / self.count
        r2 = 1.0 - self.sse / np.maximum(denominator, 1e-12)
        graph_balanced = np.empty(TARGET_DIM, np.float64)
        active_graphs = np.empty(TARGET_DIM, np.int64)
        for column in range(TARGET_DIM):
            active = self.graph_count[:, column] > 1
            graph_n = self.graph_count[active, column]
            graph_mse = self.graph_sse[active, column] / graph_n
            graph_variance = self.graph_y2[active, column] / graph_n - np.square(self.graph_y[active, column] / graph_n)
            informative = graph_variance > 1e-12
            graph_balanced[column] = np.mean(1.0 - graph_mse[informative] / graph_variance[informative])
            active_graphs[column] = int(informative.sum())
        return {'anchors_by_target': {name: int(self.count[index]) for index, name in enumerate(TARGET_COLUMNS)}, 'graphs_by_target': {name: int(active_graphs[index]) for index, name in enumerate(TARGET_COLUMNS)}, 'targets': {name: {'r2_anchor_weighted': float(r2[index]), 'r2_graph_balanced': float(graph_balanced[index]), 'mostly_explained_r2_gt_0_9': bool(r2[index] > 0.9)} for index, name in enumerate(TARGET_COLUMNS)}, 'macro_r2_anchor_weighted': float(r2.mean()), 'macro_r2_graph_balanced': float(graph_balanced.mean())}

@torch.no_grad()
def evaluate_outer_validation(data: FoldProbeData, model: Conditional3DProbe, device: torch.device) -> dict[str, Any]:
    rgb_np, context_np = data.transformed_graph_tables()
    rgb = torch.as_tensor(rgb_np, device=device)
    context = torch.as_tensor(context_np, device=device)
    model.eval()
    accumulator = R2Accumulator(len(data.graph_ids))
    for batch in data.iter_transformed():
        mask = batch.roles == 'validation'
        if not mask.any():
            continue
        graph_np, anchor_np, target_np = (batch.graph_indices[mask], batch.anchor_2d[mask], batch.targets[mask])
        predictions: list[np.ndarray] = []
        for start in range(0, len(graph_np), 16384):
            graph = torch.as_tensor(graph_np[start:start + 16384], device=device)
            anchor = torch.as_tensor(anchor_np[start:start + 16384], device=device)
            output = model(rgb[graph], anchor, context[graph])
            predictions.append(output.float().cpu().numpy())
        accumulator.update(graph_np, target_np, np.concatenate(predictions), batch.target_valid[mask])
    return accumulator.finish()

def fit_fold_probe(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', graph_limit: int | None=None, sample_limit: int | None=None) -> dict[str, Any]:
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    result_root = Path(cfg['paths']['result_root'])
    data_root = Path(cfg['paths']['data_root'])
    if graph_limit is not None:
        smoke_tag = f'graph_limit_{int(graph_limit)}'
        result_root = result_root / 'smoke' / smoke_tag
        data_root = data_root / 'smoke' / smoke_tag
    output = result_root / 'probe' / f'fold_{fold:02d}'
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / 'metrics.json'
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding='utf-8'))
        checkpoint = Path(str(completed.get('final_checkpoint', '')))
        compatible = completed.get('status') == 'PASS' and completed.get('protocol_id') == PROTOCOL_ID and (completed.get('dataset') == cfg['dataset']) and (int(completed.get('fold', -1)) == int(fold)) and (completed.get('official_test_touched') is False) and checkpoint.is_file() and (completed.get('final_checkpoint_sha256') == _sha256(checkpoint))
        if not compatible:
            raise RuntimeError(f'incompatible completed probe blocks resume: {completed_path}')
        return completed
    data = FoldProbeData(cfg, fold, graph_limit=graph_limit)
    prepared = data_root / 'prepared' / f'fold_{fold:02d}'
    if all(((prepared / name).is_file() for name in ('statistics.json', 'context_raw.npy', 'manifest.json'))):
        load_prepared_fold(data, prepared)
    else:
        data.prepare()
        save_prepared_fold(data, prepared)
    spec = cfg['probe']
    sample = collect_probe_sample(data, int(sample_limit or spec['sample_anchors']), int(spec.get('seed', PROBE_SEED)))
    rgb_np, context_np = data.transformed_graph_tables()
    rgb, context = (torch.as_tensor(rgb_np, device=target), torch.as_tensor(context_np, device=target))
    inner_models: dict[int, Conditional3DProbe] = {}
    inner_records: dict[str, Any] = {}
    best_epochs: list[int] = []
    for inner in range(int(spec.get('inner_folds', INNER_FOLDS))):
        train_mask, val_mask = (sample.inner_folds != inner, sample.inner_folds == inner)
        checkpoint = output / f'inner_{inner}.pt'
        if checkpoint.is_file():
            previous = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if previous.get('protocol_id') != PROTOCOL_ID or int(previous.get('fold', -1)) != fold or int(previous.get('inner_fold', -1)) != inner:
                raise RuntimeError(f'incompatible inner checkpoint blocks resume: {checkpoint}')
            model = Conditional3DProbe(int(spec['hidden_dim']), float(spec['dropout'])).to(target)
            model.load_state_dict(previous['model'], strict=True)
            record = previous['record']
            resumed = True
        else:
            model, record = _train_one(sample, train_mask, val_mask, rgb, context, spec, target, int(spec['seed']) + fold * 101 + inner)
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            _atomic_torch(checkpoint, {'protocol_id': PROTOCOL_ID, 'fold': fold, 'inner_fold': inner, 'model': cpu_state, 'record': record})
            resumed = False
        inner_models[inner] = model
        best_epochs.append(int(record['best_epoch']) + 1)
        inner_records[str(inner)] = {'best_epoch': int(record['best_epoch']), 'best_loss': float(record['best_loss']), 'epochs_run': len(record['history']), 'checkpoint': str(checkpoint), 'sha256': _sha256(checkpoint), 'train_anchors': int(train_mask.sum()), 'validation_anchors': int(val_mask.sum()), 'resumed_compatible_checkpoint': resumed}
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final, final_record = _train_one(sample, np.ones(len(sample.targets), bool), None, rgb, context, spec, target, int(spec['seed']) + fold * 101 + 97, fixed_epochs=final_epochs)
    final_path = output / 'final.pt'
    final_cpu_state = {key: value.detach().cpu() for key, value in final.state_dict().items()}
    _atomic_torch(final_path, {'protocol_id': PROTOCOL_ID, 'fold': fold, 'model': final_cpu_state, 'epochs': final_epochs, 'record': final_record})
    r2 = evaluate_outer_validation(data, final, target)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': int(fold), 'official_test_touched': False, 'conditioning': 'fold-specific RGB512 + paired local 2D20 + graph set mean/std/count38', 'sample_anchors': len(sample.targets), 'inner_folds': inner_records, 'final_epochs': final_epochs, 'final_checkpoint': str(final_path), 'final_checkpoint_sha256': _sha256(final_path), 'statistics_path': str(prepared / 'statistics.json'), 'statistics_sha256': _sha256(prepared / 'statistics.json'), 'prepared_manifest': str(prepared / 'manifest.json'), 'prepared_manifest_sha256': _sha256(prepared / 'manifest.json'), 'validation_r2': r2, 'runtime': {'host': socket.gethostname(), 'ended_at': datetime.now(timezone.utc).isoformat()}}
    _atomic_json(output / 'metrics.json', payload)
    return payload
