from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import random
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.breast_roi_baseline.cache import atomic_json, sha256_file
from celllift.breast_roi_baseline.features import FoldFeatureNormalizer
from celllift.breast_roi_baseline.splits import assign_grouped_folds

class MaskOnly3DProbe(nn.Module):

    def __init__(self, dropout: float=0.1) -> None:
        super().__init__()
        self.local = nn.Sequential(nn.Linear(36, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, 128), nn.GELU())
        self.context = nn.Sequential(nn.Linear(73, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 9))

    def forward(self, rays: Tensor, context: Tensor) -> Tensor:
        if rays.shape[-1] != 36 or context.shape[-1] != 73:
            raise ValueError('mask probe input width mismatch')
        return self.head(torch.cat((self.local(rays), self.context(context)), dim=-1))

@dataclass(frozen=True)
class ProbeArrays:
    graph_id: np.ndarray
    graph_code: np.ndarray
    anchor_id: np.ndarray
    roi_id: np.ndarray
    wsi_id: np.ndarray
    split: np.ndarray
    label7: np.ndarray
    validation_fold: np.ndarray
    final_validation_fold: np.ndarray
    rays: np.ndarray
    raw_geometry: np.ndarray
    valid_ncr: np.ndarray

def _fixed_list_to_numpy(column: Any, width: int) -> np.ndarray:
    values = column.combine_chunks()
    flattened = values.values.to_numpy(zero_copy_only=False)
    return np.asarray(flattened, np.float32).reshape(len(values), width)

def load_anchor_arrays(manifest_path: str | Path) -> ProbeArrays:
    import pyarrow as pa
    import pyarrow.parquet as pq
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('status') != 'PASS':
        raise RuntimeError('anchor cache manifest is not PASS')
    tables = [pq.read_table(item['path'], partitioning=None) for item in manifest['shards']]
    table = pa.concat_tables(tables, promote_options='default')

    def strings(name: str) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_pylist(), object)

    def numeric(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype)
    graph_column = table['graph_id'].combine_chunks()
    encoded_graph = graph_column.dictionary_encode()
    validation_fold = np.asarray([-1 if value is None else int(value) for value in table['validation_fold'].combine_chunks().to_pylist()], np.int8)
    final_fold = np.asarray([-1 if value is None else int(value) for value in table['final_validation_fold'].combine_chunks().to_pylist()], np.int8)
    return ProbeArrays(graph_id=np.asarray(graph_column.to_pylist(), object), graph_code=np.asarray(encoded_graph.indices.to_numpy(zero_copy_only=False), np.int32), anchor_id=numeric('anchor_id', np.int64), roi_id=strings('roi_id'), wsi_id=strings('wsi_id'), split=strings('split_new'), label7=numeric('label_7', np.int8), validation_fold=validation_fold, final_validation_fold=final_fold, rays=_fixed_list_to_numpy(table['rays36'], 36), raw_geometry=_fixed_list_to_numpy(table['raw_geometry9'], 9), valid_ncr=numeric('valid_ncr3d', bool))

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _contexts(rays: np.ndarray, graph_code: np.ndarray) -> np.ndarray:
    result = np.empty((len(rays), 73), np.float32)
    direct_boundaries = np.flatnonzero(np.r_[True, graph_code[1:] != graph_code[:-1], True])
    if len(direct_boundaries) - 1 == len(np.unique(graph_code)):
        order = None
        boundaries = direct_boundaries
    else:
        order = np.argsort(graph_code, kind='stable')
        ordered_graph = graph_code[order]
        boundaries = np.flatnonzero(np.r_[True, ordered_graph[1:] != ordered_graph[:-1], True])
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        index = slice(begin, end) if order is None else order[begin:end]
        block = rays[index]
        context = np.concatenate((block.mean(0), block.std(0), [np.log1p(len(block))])).astype(np.float32)
        result[index] = context
    return result

def _masked_mse(prediction: Tensor, target: Tensor, valid_ncr: Tensor) -> Tensor:
    squared = torch.square(prediction - target)
    weights = torch.ones_like(squared)
    weights[:, 8] = valid_ncr.to(squared.dtype)
    return (squared * weights).sum() / weights.sum().clamp_min(1)

def _predict(model: nn.Module, rays: np.ndarray, context: np.ndarray, device: torch.device, *, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(rays), int(batch_size)):
            end = min(len(rays), begin + int(batch_size))
            local = torch.from_numpy(rays[begin:end])
            graph = torch.from_numpy(context[begin:end])
            outputs.append(model(local.to(device), graph.to(device)).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)

def _fit(rays: np.ndarray, context: np.ndarray, targets: np.ndarray, valid_ncr: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray, *, seed: int, device: torch.device, max_epochs: int=100, patience: int=15, fixed_epochs: int | None=None, train_batch_size: int=65536, prediction_batch_size: int=131072) -> tuple[MaskOnly3DProbe, int, list[dict[str, float | int]]]:
    _seed_everything(seed)
    model = MaskOnly3DProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    train_rays = torch.from_numpy(rays[train_mask]).to(device)
    train_context = torch.from_numpy(context[train_mask]).to(device)
    train_targets = torch.from_numpy(targets[train_mask]).to(device)
    train_valid_ncr = torch.from_numpy(valid_ncr[train_mask]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    train_count = len(train_rays)
    best_loss, best_epoch, stale = (float('inf'), -1, 0)
    best_state = None
    history: list[dict[str, float | int]] = []
    epochs = int(fixed_epochs or max_epochs)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        permutation = torch.randperm(train_count, generator=generator, device=device)
        for begin in range(0, train_count, int(train_batch_size)):
            index = permutation[begin:min(train_count, begin + int(train_batch_size))]
            local = train_rays.index_select(0, index)
            graph = train_context.index_select(0, index)
            target = train_targets.index_select(0, index)
            valid = train_valid_ncr.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_mse(model(local, graph), target, valid)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(local)
            count += len(local)
        if fixed_epochs is not None:
            history.append({'epoch': epoch, 'train_loss': total / max(1, count)})
            print(json.dumps({'event': 'probe_epoch', **history[-1]}), flush=True)
            continue
        prediction = _predict(model, rays[validation_mask], context[validation_mask], device, batch_size=prediction_batch_size)
        target = targets[validation_mask]
        valid = valid_ncr[validation_mask]
        weights = np.ones_like(target, np.float32)
        weights[~valid, 8] = 0
        validation_loss = float((np.square(prediction - target) * weights).sum() / weights.sum())
        history.append({'epoch': epoch, 'train_loss': total / max(1, count), 'validation_loss': validation_loss})
        print(json.dumps({'event': 'probe_epoch', **history[-1]}), flush=True)
        if validation_loss < best_loss - 1e-10:
            best_loss, best_epoch, stale = (validation_loss, epoch, 0)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError('probe did not produce a best checkpoint')
        model.load_state_dict(best_state)
        selected_epochs = best_epoch + 1
    else:
        selected_epochs = fixed_epochs
    return (model, int(selected_epochs), history)

def _r2(target: np.ndarray, prediction: np.ndarray, valid_ncr: np.ndarray) -> list[float]:
    result = []
    for column in range(9):
        selected = valid_ncr if column == 8 else np.ones(len(target), bool)
        truth, estimate = (target[selected, column], prediction[selected, column])
        denominator = float(np.square(truth - truth.mean()).sum())
        result.append(float(1 - np.square(truth - estimate).sum() / denominator) if denominator > 0 else float('nan'))
    return result

def fit_nested_probe(*, anchor_manifest: str | Path, phase: str, outer_fold: int, output_dir: str | Path, seed: int=42, device: str='cuda', train_batch_size: int=65536, prediction_batch_size: int=131072) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(output_dir)
    manifest_path = destination / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            return previous
    arrays = load_anchor_arrays(anchor_manifest)
    if phase not in {'development', 'final_test'}:
        raise ValueError('phase must be development or final_test')
    fold_values = arrays.validation_fold if phase == 'development' else arrays.final_validation_fold
    eligible = np.isin(arrays.split, ['train'] if phase == 'development' else ['train', 'val'])
    outer_train = eligible & (fold_values != outer_fold)
    outer_heldout = eligible & (fold_values == outer_fold)
    external = arrays.split == ('val' if phase == 'development' else 'test')
    if not outer_train.any() or not outer_heldout.any() or (not external.any()):
        raise RuntimeError('empty outer train/heldout/external probe role')
    normalizer = FoldFeatureNormalizer.fit(arrays.raw_geometry, arrays.valid_ncr, outer_train)
    target = normalizer.transform_direct(arrays.raw_geometry, arrays.valid_ncr)
    ray_mean, ray_std = (arrays.rays[outer_train].mean(0), np.maximum(arrays.rays[outer_train].std(0), 1e-06))
    rays = ((arrays.rays - ray_mean) / ray_std).astype(np.float32)
    context = _contexts(rays, arrays.graph_code)
    context_mean, context_std = (context[outer_train].mean(0), np.maximum(context[outer_train].std(0), 1e-06))
    context = ((context - context_mean) / context_std).astype(np.float32)
    groups = []
    for wsi in sorted(set(map(str, arrays.wsi_id[outer_train]))):
        selected = outer_train & (arrays.wsi_id == wsi)
        groups.append({'wsi_id': wsi, 'label_7': int(arrays.label7[np.flatnonzero(selected)[0]])})
    inner_assignment = assign_grouped_folds(groups, folds=3, seed=seed + outer_fold)
    prediction = np.full((len(arrays.rays), 9), np.nan, np.float32)
    selected_epochs = []
    target_device = torch.device(device)
    for inner in range(3):
        inner_validation = outer_train & np.asarray([inner_assignment.get(str(wsi), -1) == inner for wsi in arrays.wsi_id])
        inner_train = outer_train & ~inner_validation
        model, epochs, _ = _fit(rays, context, target, arrays.valid_ncr, inner_train, inner_validation, seed=seed + 100 * outer_fold + inner, device=target_device, train_batch_size=train_batch_size, prediction_batch_size=prediction_batch_size)
        prediction[inner_validation] = _predict(model, rays[inner_validation], context[inner_validation], target_device, batch_size=prediction_batch_size)
        selected_epochs.append(epochs)
    final_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    final_model, _, history = _fit(rays, context, target, arrays.valid_ncr, outer_train, outer_heldout, seed=seed + 1000 + outer_fold, device=target_device, fixed_epochs=final_epochs, train_batch_size=train_batch_size, prediction_batch_size=prediction_batch_size)
    final_role = outer_heldout | external
    prediction[final_role] = _predict(final_model, rays[final_role], context[final_role], target_device, batch_size=prediction_batch_size)
    written = outer_train | final_role
    if not np.isfinite(prediction[written]).all():
        raise RuntimeError('probe prediction coverage contains NaN/Inf')
    residual = target - prediction
    residual[~arrays.valid_ncr, 8] = 0.0
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / 'probe_predictions.parquet'
    temporary = output_path.with_name(f'.{output_path.name}.tmp.{os.getpid()}')
    flat_prediction = pa.FixedSizeListArray.from_arrays(pa.array(prediction[written].reshape(-1)), 9)
    flat_residual = pa.FixedSizeListArray.from_arrays(pa.array(residual[written].reshape(-1)), 9)
    pq.write_table(pa.table({'graph_id': pa.array(arrays.graph_id[written]), 'anchor_id': pa.array(arrays.anchor_id[written]), 'roi_id': pa.array(arrays.roi_id[written]), 'wsi_id': pa.array(arrays.wsi_id[written]), 'role': pa.array(np.where(outer_train[written], 'train_inner_oof', np.where(outer_heldout[written], 'outer_heldout', 'external'))), 'valid_ncr3d': pa.array(arrays.valid_ncr[written]), 'prediction9': flat_prediction, 'residual9': flat_residual}), temporary, compression='zstd', row_group_size=65536)
    os.replace(temporary, output_path)
    heldout_prediction = prediction[outer_heldout]
    payload = {'status': 'PASS', 'phase': phase, 'outer_fold': outer_fold, 'seed': seed, 'training_anchors': int(outer_train.sum()), 'heldout_anchors': int(outer_heldout.sum()), 'external_anchors': int(external.sum()), 'inner_selected_epochs': selected_epochs, 'final_epochs': final_epochs, 'heldout_r2': _r2(target[outer_heldout], heldout_prediction, arrays.valid_ncr[outer_heldout]), 'train_batch_size': int(train_batch_size), 'prediction_batch_size': int(prediction_batch_size), 'normalizer': normalizer.as_dict(), 'ray_mean': ray_mean.tolist(), 'ray_std': ray_std.tolist(), 'context_mean': context_mean.tolist(), 'context_std': context_std.tolist(), 'output': str(output_path), 'output_sha256': sha256_file(output_path), 'history': history}
    atomic_json(manifest_path, payload)
    return payload
