from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict
from celllift.runtime import ResourcePath as Path
from typing import Sequence
import hashlib
import importlib.util
import sys
import numpy as np
from .io_utils import atomic_json, atomic_npz, atomic_torch
from .tokens import FeatureNormalizer

def graph_context(rays36: np.ndarray, graph_code: np.ndarray) -> np.ndarray:
    rays = np.asarray(rays36, np.float32)
    code = np.asarray(graph_code, np.int64)
    if rays.ndim != 2 or rays.shape[1] != 36 or code.shape != (len(rays),) or (len(rays) == 0):
        raise ValueError('rays/graph_code must be nonempty [N,36]/[N]')
    if code.min() < 0:
        raise ValueError('graph_code must be nonnegative')
    size = int(code.max()) + 1
    counts = np.bincount(code, minlength=size).astype(np.float64)
    if np.any(counts[code] <= 0):
        raise RuntimeError('graph context encountered an empty referenced graph')
    sums = np.stack([np.bincount(code, weights=rays[:, column], minlength=size) for column in range(36)], axis=1)
    squares = np.stack([np.bincount(code, weights=np.square(rays[:, column], dtype=np.float64), minlength=size) for column in range(36)], axis=1)
    mean = sums / np.maximum(counts[:, None], 1.0)
    variance = np.maximum(squares / np.maximum(counts[:, None], 1.0) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return np.concatenate((mean[code], std[code], np.log1p(counts[code])[:, None]), axis=1).astype(np.float32)

def grouped_assignment(groups: Sequence[str], folds: int=3, seed: int=20260909) -> dict[str, int]:
    unique = sorted(set(map(str, groups)), key=lambda value: hashlib.sha256(f'{seed}|{value}'.encode()).digest())
    return {value: index % folds for index, value in enumerate(unique)}

def load_mask_probe(conditional_geometry_root: str | Path):
    root = Path(conditional_geometry_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('projection_scene_reused_mask_probe', root / 'models' / 'mask_probe.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load MaskOnly3DProbe')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MaskOnly3DProbe

def _fit(model_class, rays, context, target, valid_ncr, train, validation, *, seed, device, fixed_epochs=None, max_epochs=100, patience=15, batch_size=65536):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_class(256, 0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = None
    best_loss = float('inf')
    stale = 0
    history = []
    train_ids = np.flatnonzero(train)
    val_ids = np.flatnonzero(validation)
    epochs = fixed_epochs or max_epochs
    for epoch in range(int(epochs)):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(train_ids)
        for start in range(0, len(train_ids), batch_size):
            ids = train_ids[start:start + batch_size]
            x = torch.as_tensor(rays[ids], device=device)
            c = torch.as_tensor(context[ids], device=device)
            y = torch.as_tensor(target[ids], device=device)
            mask = torch.ones_like(y)
            mask[:, 8] = torch.as_tensor(valid_ncr[ids], device=device, dtype=y.dtype)
            pred = model(x, c)
            loss = ((pred - y).square() * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        error = 0.0
        weight = 0
        with torch.inference_mode():
            for start in range(0, len(val_ids), batch_size):
                ids = val_ids[start:start + batch_size]
                pred = model(torch.as_tensor(rays[ids], device=device), torch.as_tensor(context[ids], device=device))
                y = torch.as_tensor(target[ids], device=device)
                mask = torch.ones_like(y)
                mask[:, 8] = torch.as_tensor(valid_ncr[ids], device=device, dtype=y.dtype)
                error += float(((pred - y).square() * mask).sum().cpu())
                weight += int(mask.sum().cpu())
        value = error / max(weight, 1)
        history.append(value)
        if value < best_loss - 1e-07:
            best_loss = value
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    if fixed_epochs is None and best is not None:
        model.load_state_dict(best)
    return (model, len(history), history)

def _predict(model, rays, context, device, batch_size=131072):
    import torch
    output = np.empty((len(rays), 9), np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rays), batch_size):
            stop = min(len(rays), start + batch_size)
            output[start:stop] = model(torch.as_tensor(rays[start:stop], device=device), torch.as_tensor(context[start:stop], device=device)).cpu().numpy()
    return output

def _r2_report(target: np.ndarray, prediction: np.ndarray, valid_ncr: np.ndarray, selected: np.ndarray) -> dict:
    target = np.asarray(target, np.float64)
    prediction = np.asarray(prediction, np.float64)
    selected = np.asarray(selected, bool)
    valid = np.asarray(valid_ncr, bool)
    values = []
    for channel in range(9):
        mask = selected & (valid if channel == 8 else True)
        y = target[mask, channel]
        p = prediction[mask, channel]
        denominator = float(np.square(y - y.mean()).sum()) if len(y) else 0.0
        score = float(1.0 - np.square(y - p).sum() / denominator) if denominator > 0 else float('nan')
        values.append(score)
    finite = [value for value in values if np.isfinite(value)]
    return {'per_channel': values, 'mean': float(np.mean(finite)) if finite else float('nan'), 'anchors': int(selected.sum())}

def crossfit_residual(*, rays36: np.ndarray, direct9: np.ndarray, valid_ncr: np.ndarray, graph_code: np.ndarray, groups: Sequence[str], outer_train: np.ndarray, outer_heldout: np.ndarray, external: np.ndarray, conditional_geometry_root: str | Path, destination: str | Path, seed: int=20260909, device: str='cuda') -> dict:
    rays = np.asarray(rays36, np.float32)
    direct = np.asarray(direct9, np.float32)
    valid = np.asarray(valid_ncr, bool)
    train = np.asarray(outer_train, bool)
    normalizer = FeatureNormalizer.fit(direct, valid, train)
    target = normalizer.transform(direct, valid)
    ray_mean = rays[train].mean(0)
    ray_std = np.maximum(rays[train].std(0), 1e-06)
    rays_z = ((rays - ray_mean) / ray_std).astype(np.float32)
    context_raw = graph_context(rays_z, graph_code)
    context_mean = context_raw[train].mean(0)
    context_std = np.maximum(context_raw[train].std(0), 1e-06)
    context = ((context_raw - context_mean) / context_std).astype(np.float32)
    assignment = grouped_assignment([group for group, selected in zip(groups, train) if selected], 3, seed)
    fold_code = np.asarray([assignment.get(str(group), -1) for group in groups])
    prediction = np.full((len(rays), 9), np.nan, np.float32)
    selected_epochs = []
    model_class = load_mask_probe(conditional_geometry_root)
    for inner in range(3):
        inner_val = train & (fold_code == inner)
        inner_train = train & ~inner_val
        model, epochs, _ = _fit(model_class, rays_z, context, target, valid, inner_train, inner_val, seed=seed + inner, device=device)
        prediction[inner_val] = _predict(model, rays_z[inner_val], context[inner_val], device)
        selected_epochs.append(epochs)
    final_epochs = max(1, int(round(np.median(selected_epochs))))
    model, _, history = _fit(model_class, rays_z, context, target, valid, train, outer_heldout, seed=seed + 1000, device=device, fixed_epochs=final_epochs)
    final = np.asarray(outer_heldout, bool) | np.asarray(external, bool)
    prediction[final] = _predict(model, rays_z[final], context[final], device)
    written = train | final
    if not np.all(np.isfinite(prediction[written])):
        raise RuntimeError('residual prediction coverage incomplete')
    residual = target - prediction
    residual[~valid, 8] = 0.0
    destination = Path(destination)
    atomic_npz(destination, prediction9=prediction[written], residual9=residual[written], written=written)
    checkpoint = destination.with_suffix('.pt')
    atomic_torch(checkpoint, {'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'final_epochs': final_epochs, 'seed': seed})
    manifest = {'status': 'PASS', 'seed': seed, 'inner_epochs': selected_epochs, 'final_epochs': final_epochs, 'anchors': int(written.sum()), 'probe_r2': {'outer_training_crossfit': _r2_report(target, prediction, valid, train), 'outer_heldout': _r2_report(target, prediction, valid, np.asarray(outer_heldout, bool))}, 'normalizer': {**asdict(normalizer), 'mean': normalizer.mean.tolist(), 'std': normalizer.std.tolist()}, 'ray_mean': ray_mean.tolist(), 'ray_std': ray_std.tolist(), 'context_mean': context_mean.tolist(), 'context_std': context_std.tolist(), 'history': history, 'output': str(destination), 'checkpoint': str(checkpoint)}
    return manifest
