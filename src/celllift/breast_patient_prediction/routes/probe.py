from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
import numpy as np
from celllift.matched_geometry_controls.residual import _predict, graph_context, grouped_assignment, load_mask_probe
from ..io_utils import atomic_json, atomic_npz
from ..model_input import layout
from ..protocol import PROBE_SEED
from .bags import SlideCache, build_bags, load_tile_geometry, split_bags
from .config import probe_dir, protocol_meta
from .geometry import geometry9
from .scale import fit_node_pooled_scale, select_best_epoch
conditional_geometry_ROOT = Path(__file__).resolve().parents[2] / 'conditional_geometry'

def _collect(bags, cache: SlideCache):
    from collections import defaultdict
    by_slide: dict[str, list] = defaultdict(list)
    meta = {}
    for bag in bags:
        for tile in bag['tiles']:
            by_slide[str(tile['slide_id'])].append(tile)
            meta[tile['slide_id'], tile['graph_id']] = bag['patient_id']
    rays, direct, valid, included, patient_codes, codes, keys = ([], [], [], [], [], [], [])
    patient_to_code: dict[str, int] = {}
    code = 0
    for slide_id in sorted(by_slide):
        for tile in by_slide[slide_id]:
            row = load_tile_geometry(cache, tile)
            n = len(row.get('rays', []))
            if n == 0:
                keys.append((tile['slide_id'], tile['graph_id'], np.empty(0, np.int64), 0))
                continue
            raw_direct = geometry9(row['node3d'])
            raw_rays = np.asarray(row['rays'], np.float32)
            include = np.asarray(row['include'], bool).reshape(-1)
            finite = np.isfinite(raw_direct).all(axis=1) if len(raw_direct) else np.zeros(0, bool)
            rays.append(raw_rays)
            direct.append(raw_direct)
            included.append(include)
            valid.append(include & np.asarray(row['valid3d'], bool) & finite)
            patient_id = meta[tile['slide_id'], tile['graph_id']]
            if patient_id not in patient_to_code:
                patient_to_code[patient_id] = len(patient_to_code)
            patient_codes.append(np.full(n, patient_to_code[patient_id], np.int32))
            codes.append(np.full(n, code, np.int64))
            nucleus_id = np.asarray(row.get('nucleus_id', np.arange(n))).reshape(-1)
            keys.append((tile['slide_id'], tile['graph_id'], nucleus_id, n))
            code += 1
        cache._store.pop(slide_id, None)
    if not rays:
        raise RuntimeError('probe collected no nuclei')
    print(f'probe collected slides={len(by_slide)} tiles={code} nuclei={sum((len(item) for item in rays))}', flush=True)
    return (np.concatenate(rays, 0), np.concatenate(direct, 0), np.concatenate(valid, 0), np.concatenate(included, 0), np.concatenate(patient_codes, 0), patient_to_code, np.concatenate(codes, 0), keys)

def _standardize_context(context_raw: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = context_raw[train].mean(0)
    std = np.maximum(context_raw[train].std(0), 1e-06)
    return (((context_raw - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32))

def fit_masked_probe(model_class, rays, context, target, row_valid, train, validation, *, seed: int, device: str, fixed_epochs: int | None=None, max_epochs: int=100, patience: int=15, batch_size: int=65536):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_class(256, 0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = None
    best_loss = float('inf')
    best_epoch = 1
    stale = 0
    history = []
    train_ids = np.flatnonzero(train)
    val_ids = np.flatnonzero(validation)
    epochs = int(fixed_epochs or max_epochs)
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(train_ids)
        for start in range(0, len(train_ids), batch_size):
            ids = train_ids[start:start + batch_size]
            pred = model(torch.as_tensor(rays[ids], device=device), torch.as_tensor(context[ids], device=device))
            y = torch.as_tensor(target[ids], device=device)
            mask = torch.as_tensor(row_valid[ids], device=device, dtype=y.dtype).unsqueeze(-1).expand_as(y)
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
                mask = torch.as_tensor(row_valid[ids], device=device, dtype=y.dtype).unsqueeze(-1).expand_as(y)
                error += float(((pred - y).square() * mask).sum().cpu())
                weight += int(mask.sum().cpu())
        value = error / max(weight, 1)
        history.append(value)
        if value < best_loss - 1e-07:
            best_loss = value
            best_epoch = epoch + 1
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    stop_epoch = len(history)
    if fixed_epochs is None and best is not None:
        model.load_state_dict(best)
    chosen = select_best_epoch(history) if fixed_epochs is None else int(fixed_epochs)
    if fixed_epochs is None and chosen != best_epoch:
        raise RuntimeError(f'best epoch mismatch {chosen} != {best_epoch}')
    return (model, best_epoch, stop_epoch, history)

def _r2_valid(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray, selected: np.ndarray) -> dict:
    mask = np.asarray(valid, bool) & np.asarray(selected, bool)
    scores = []
    for channel in range(9):
        y = np.asarray(target[mask, channel], np.float64)
        p = np.asarray(prediction[mask, channel], np.float64)
        denom = float(np.square(y - y.mean()).sum()) if len(y) else 0.0
        score = float(1.0 - np.square(y - p).sum() / denom) if denom > 0 else float('nan')
        scores.append(score)
    finite = [value for value in scores if np.isfinite(value)]
    return {'per_channel': scores, 'mean': float(np.mean(finite)) if finite else float('nan'), 'anchors': int(mask.sum())}

def _write_residuals(dest: Path, keys, residual: np.ndarray, valid: np.ndarray) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    by_slide: dict[str, list] = {}
    offset = 0
    for slide_id, graph_id, nucleus_id, n in keys:
        piece = residual[offset:offset + n] if n else np.zeros((0, 9), np.float32)
        row_valid = valid[offset:offset + n] if n else np.zeros(0, bool)
        offset += n
        by_slide.setdefault(slide_id, []).append({'graph_id': graph_id, 'nucleus_id': np.asarray(nucleus_id, np.int64), 'residual9': np.asarray(piece, np.float32), 'valid_target': np.asarray(row_valid, bool), 'empty': n == 0})
    written = 0
    for slide_id, rows in by_slide.items():
        rows.sort(key=lambda item: str(item['graph_id']))
        packed = {'graph_ids': np.asarray([row['graph_id'] for row in rows], dtype='U160'), 'empty': np.asarray([row['empty'] for row in rows])}
        for key in ('nucleus_id', 'residual9', 'valid_target'):
            pieces = [row[key] for row in rows]
            counts = np.asarray([len(item) for item in pieces], np.int64)
            ptr = np.zeros(len(counts) + 1, np.int64)
            ptr[1:] = np.cumsum(counts)
            tail = pieces[0].shape[1:] if pieces[0].ndim >= 1 else ()
            dtype = pieces[0].dtype
            blob = np.zeros((int(ptr[-1]), *tail), dtype=dtype) if int(ptr[-1]) else np.zeros((0, *tail), dtype=dtype)
            cursor = 0
            for item in pieces:
                if len(item):
                    blob[cursor:cursor + len(item)] = item
                    cursor += len(item)
            packed[key] = blob
            packed[f'{key}_ptr'] = ptr
        atomic_npz(dest / f'{layout.shard_name(slide_id)}__{layout._slide_stem(slide_id)}.npz', **packed)
        written += 1
    return written

def _phase(rays, direct, include, valid, codes, train, validation, scale):
    target = scale.transform_geometry(direct, valid)
    ray_input = scale.transform_rays(rays, include)
    context_raw = graph_context(np.where(np.isfinite(ray_input), ray_input, 0).astype(np.float32), codes)
    context, context_mean, context_std = _standardize_context(context_raw, train)
    return (target, context, ray_input, context_mean, context_std)

def fit_probe(task: str, *, device: str='cuda', data_root=None, seed: int=PROBE_SEED) -> dict:
    dest = probe_dir(task)
    marker = dest / 'manifest.json'
    if marker.is_file():
        return {'status': 'skip', 'path': str(marker)}
    dest.mkdir(parents=True, exist_ok=True)
    base, bags, _ = build_bags(task, data_root=data_root, need_image=False)
    splits = split_bags(bags)
    cache = SlideCache(base, 'geometry', max_slides=8)
    ordered = splits['fit'] + splits['val'] + splits['test']
    rays, direct, valid, include, patient_codes, patient_to_code, codes, keys = _collect(ordered, cache)
    n = len(rays)
    split_of = {}
    for bag in ordered:
        for tile in bag['tiles']:
            split_of[tile['slide_id'], tile['graph_id']] = bag['split']
    train = np.zeros(n, bool)
    held = np.zeros(n, bool)
    external = np.zeros(n, bool)
    offset = 0
    for slide_id, graph_id, _, count in keys:
        part = split_of.get((slide_id, graph_id), 'fit')
        sl = slice(offset, offset + count)
        if part == 'fit':
            train[sl] = True
        elif part == 'val':
            held[sl] = True
        else:
            external[sl] = True
        offset += count
    if not (train & valid).any():
        raise RuntimeError(f'{task} probe FIT has no valid 3D targets')
    code_to_patient = {code: patient for patient, code in patient_to_code.items()}
    fit_patients = [code_to_patient[int(code)] for code in np.unique(patient_codes[train])]
    assignment = grouped_assignment(fit_patients, 3, seed)
    inner_val_codes = np.asarray([patient_to_code[patient] for patient, fold in assignment.items() if fold == 0], dtype=np.int32)
    inner_val = train & np.isin(patient_codes, inner_val_codes)
    inner_train = train & ~inner_val
    if not (inner_train & valid).any() or not (inner_val & valid).any():
        raise RuntimeError(f'{task} probe inner split has no valid targets')
    model_class = load_mask_probe(conditional_geometry_ROOT)
    inner_scale = fit_node_pooled_scale(rays[inner_train], direct[inner_train], include[inner_train], valid[inner_train])
    inner_target, inner_context, inner_rays, _, _ = _phase(rays, direct, include, valid, codes, inner_train, inner_val, inner_scale)
    _, best_epoch, stop_epoch, inner_history = fit_masked_probe(model_class, inner_rays, inner_context, inner_target, valid, inner_train, inner_val, seed=seed, device=device)
    final_scale = fit_node_pooled_scale(rays[train], direct[train], include[train], valid[train])
    target, context, ray_input, context_mean, context_std = _phase(rays, direct, include, valid, codes, train, held, final_scale)
    model, _, _, history = fit_masked_probe(model_class, ray_input, context, target, valid, train, held, seed=seed + 1000, device=device, fixed_epochs=best_epoch)
    prediction = _predict(model, ray_input, context, device)
    residual = (target - prediction).astype(np.float32)
    residual[~valid] = 0.0
    import torch
    ckpt = dest / 'probe.pt'
    torch.save({'model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'best_epoch': int(best_epoch), 'stop_epoch': int(stop_epoch), 'final_epochs': int(best_epoch), 'seed': seed, 'inner_folds_used': 1, 'scale': final_scale.as_dict(), 'context_mean': context_mean, 'context_std': context_std}, ckpt)
    slides = _write_residuals(dest / 'residuals', keys, residual, valid)
    manifest = {**protocol_meta(), 'status': 'PASS', 'task': task, 'seed': seed, 'nuclei': int(n), 'valid_fit_targets': int((train & valid).sum()), 'best_epoch': int(best_epoch), 'stop_epoch': int(stop_epoch), 'final_epochs': int(best_epoch), 'inner_split': 'grouped_assignment fold 0 only', 'slides': slides, 'probe_r2': {'fit': _r2_valid(target, prediction, valid, train), 'val': _r2_valid(target, prediction, valid, held)}, 'inner_history': inner_history, 'history': history, 'scale': final_scale.as_dict(), 'checkpoint': str(ckpt), 'reads_task_label': False, 'invalid_targets_in_loss': False}
    atomic_json(marker, manifest)
    return manifest
