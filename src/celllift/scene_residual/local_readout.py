from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .io_utils import atomic_json, atomic_npz, atomic_parquet, atomic_torch, load_npz, read_json, read_parquet
from .models import LocalReadoutAdapter
from .train import _heldout_rows, build_baseline, build_samples_for_cache
from .training import DEFAULT_DELTA_L2, DEFAULT_EPOCH_SEED_STRIDE, DEFAULT_LR, DEFAULT_MAX_EPOCHS, DEFAULT_PATIENCE, DEFAULT_WEIGHT_DECAY, Trainer, classes_for, fold_split, metric, partition, selection_score, _score
MATCH_ATOL = 1e-05
ADAPTER_HIDDEN = 16
ADAPTER_BATCH = 32
LOCAL_ARMS = {'G-Local': 'G', 'XY-Local': 'XY'}

def source_arm(arm: str) -> str:
    if arm not in LOCAL_ARMS:
        raise ValueError(f'local readout jobs must use G-Local or XY-Local, got {arm}')
    return LOCAL_ARMS[arm]

def _source_root(cfg: Mapping[str, Any]) -> Path:
    paths = cfg['paths']
    return Path(paths.get('source_root') or paths['result_root'])

def _fold_dir(root: Path, dataset: str, arm: str, seed: int, fold: int) -> Path:
    return root / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{fold:02d}'

def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'missing reference predictions: {path}')
    return {str(row['sample_id']): row for row in read_parquet(path)}

def _finite(*values: Any) -> None:
    for value in values:
        array = np.asarray(value, float) if not hasattr(value, 'isfinite') or not hasattr(value, 'detach') else None
        if array is None:
            tensor = value.detach()
            if not bool(tensor.isfinite().all()):
                raise RuntimeError('non-finite tensor during local-readout training')
        elif not np.isfinite(array).all():
            raise RuntimeError('non-finite array during local-readout training')

def _pack(records: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> dict[str, Any]:
    lookup = {str(record['sample_id']): record for record in records}
    missing = [key for key in ids if key not in lookup]
    if missing:
        raise RuntimeError(f'node cache missing {len(missing)} samples, e.g. {missing[:3]}')
    offsets = [0]
    hidden, valid, xy, depth = ([], [], [], [])
    logits, labels, baseline, folds, groups = ([], [], [], [], [])
    for key in ids:
        record = lookup[key]
        hidden.append(np.asarray(record['hidden'], np.float32))
        valid.append(np.asarray(record['valid'], bool))
        xy.append(np.asarray(record['xy'], np.float32))
        depth.append(np.asarray(record['z'], np.float32))
        logits.append(np.asarray(record['logits'], np.float32))
        labels.append(int(record['label_id']))
        baseline.append(np.asarray(record['baseline'], np.float32))
        folds.append(int(record['fold']))
        groups.append(str(record['group_id']))
        offsets.append(offsets[-1] + len(record['hidden']))
    return {'sample_id': np.asarray(ids, object), 'offsets': np.asarray(offsets, np.int64), 'hidden': np.concatenate(hidden, 0) if hidden else np.zeros((0, 64), np.float32), 'valid': np.concatenate(valid, 0) if valid else np.zeros((0,), bool), 'xy': np.concatenate(xy, 0) if xy else np.zeros((0, 2), np.float32), 'z': np.concatenate(depth, 0) if depth else np.zeros((0,), np.float32), 'logits': np.stack(logits, 0) if logits else np.zeros((0, 1), np.float32), 'labels': np.asarray(labels, np.int64), 'baseline': np.stack(baseline, 0) if baseline else np.zeros((0, 1), np.float32), 'fold': np.asarray(folds, np.int64), 'group_id': np.asarray(groups, object)}

def _slice_pack(pack: Mapping[str, Any], start: int, stop: int) -> dict[str, np.ndarray]:
    offsets = np.asarray(pack['offsets'])
    node_start = int(offsets[start])
    node_stop = int(offsets[stop])
    local = offsets[start:stop + 1] - node_start
    sample_index = np.repeat(np.arange(stop - start), np.diff(local))
    return {'sample_id': np.asarray(pack['sample_id'][start:stop], object), 'offsets': local, 'hidden': np.asarray(pack['hidden'][node_start:node_stop]), 'valid': np.asarray(pack['valid'][node_start:node_stop]), 'xy': np.asarray(pack['xy'][node_start:node_stop]), 'z': np.asarray(pack['z'][node_start:node_stop]), 'logits': np.asarray(pack['logits'][start:stop]), 'labels': np.asarray(pack['labels'][start:stop]), 'baseline': np.asarray(pack['baseline'][start:stop]), 'fold': np.asarray(pack['fold'][start:stop]), 'group_id': np.asarray(pack['group_id'][start:stop], object), 'sample_index': sample_index.astype(np.int64)}

def _encode_samples(trainer: Trainer, samples: Sequence[Any]) -> list[dict[str, Any]]:
    torch = trainer.torch
    from .data import bag_batches
    trainer.model.eval()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for part in bag_batches(list(samples), trainer.bags_per_batch, trainer.tiles_per_batch):
            batch = trainer._batch(part)
            encoded = trainer.model.encode_nodes(batch)
            output = trainer.model(batch)
            hidden = encoded.float().cpu().numpy()
            include = batch.node_include.cpu().numpy().astype(bool)
            graph_index = batch.node_graph.cpu().numpy()
            part_logits = output['logits'].float().cpu().numpy()
            if part_logits.ndim == 1:
                part_logits = part_logits.reshape(-1, 1)
            for offset, sample in enumerate(batch.samples):
                if len(sample.graph_ids) != 1:
                    raise RuntimeError('local readout set_encoding caches one graph per SICAP sample')
                graph = trainer.cache.get(sample.graph_ids[0])
                node_ids = np.flatnonzero(graph_index == offset)
                records.append({'sample_id': sample.bag_id, 'hidden': hidden[node_ids], 'valid': include[node_ids], 'xy': np.asarray(graph.center_xy, np.float32), 'z': np.asarray(graph.center_z, np.float32), 'logits': np.asarray(part_logits[offset], np.float32), 'label_id': int(sample.label_id), 'baseline': np.asarray(sample.baseline, np.float32), 'fold': int(sample.fold), 'group_id': str(sample.group_id)})
                if len(records[-1]['hidden']) != len(graph.center_xy):
                    raise RuntimeError(f'node count mismatch for {sample.bag_id}')
    if {record['sample_id'] for record in records} != {sample.bag_id for sample in samples}:
        raise RuntimeError('encoded sample set does not match the request')
    order = {sample.bag_id: position for position, sample in enumerate(samples)}
    records.sort(key=lambda row: order[row['sample_id']])
    return records

def _adapter_forward(adapter: LocalReadoutAdapter, pack: Mapping[str, Any], device: str):
    import torch
    hidden = torch.as_tensor(pack['hidden'], device=device)
    valid = torch.as_tensor(pack['valid'], device=device)
    index = torch.as_tensor(pack['sample_index'], device=device, dtype=torch.long)
    logits = torch.as_tensor(pack['logits'], device=device)
    labels = torch.as_tensor(pack['labels'], device=device, dtype=torch.long)
    size = int(len(pack['labels']))
    return (adapter(hidden, valid, index, size, logits), labels)

def _probabilities(output, classes: int):
    import torch
    logits = output['logits']
    if classes == 1:
        return torch.sigmoid(logits).reshape(-1, 1)
    return torch.softmax(logits, -1)

def _check_finite_step(loss, adapter) -> None:
    import torch
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('non-finite local-readout loss')
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f'non-finite gradient for {name}')
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f'non-finite parameter {name}')

def _predict_pack(adapter: LocalReadoutAdapter, pack: Mapping[str, Any], device: str, classes: int, batch_size: int=ADAPTER_BATCH):
    import torch
    adapter.eval()
    count = len(pack['labels'])
    width = 1 if classes == 1 else classes
    final = np.zeros((count, width), np.float32)
    epsilon = np.zeros((count, width), np.float32)
    weights = np.zeros(int(pack['offsets'][-1]), np.float32)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            chunk = _slice_pack(pack, start, stop)
            output, _ = _adapter_forward(adapter, chunk, device)
            _finite(output['logits'], output['epsilon'])
            probabilities = _probabilities(output, classes).float().cpu().numpy()
            part_eps = output['epsilon'].float().cpu().numpy()
            if part_eps.ndim == 1:
                part_eps = part_eps.reshape(-1, 1)
            final[start:stop] = probabilities.reshape(stop - start, width)
            epsilon[start:stop] = part_eps.reshape(stop - start, width)
            node_start = int(pack['offsets'][start])
            node_stop = int(pack['offsets'][stop])
            weights[node_start:node_stop] = output['weight'].float().cpu().numpy()
    return (final, epsilon, weights)

def train_fold(cfg: Mapping[str, Any], dataset: str, arm: str, fold: int, seed: int, device: str, *, cache=None, baseline=None, max_epochs: int=DEFAULT_MAX_EPOCHS, patience: int=DEFAULT_PATIENCE, max_steps: int | None=None, **_ignored) -> dict[str, Any]:
    import torch
    if dataset != 'sicapv2':
        raise RuntimeError('local readout set_encoding only runs SICAPv2')
    frozen_arm = source_arm(arm)
    result_root = Path(cfg['paths']['result_root'])
    source_root = _source_root(cfg)
    destination = _fold_dir(result_root, dataset, arm, seed, fold)
    manifest_path = destination / 'job.json'
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get('status') == 'PASS':
            return {'status': 'REUSED', **previous}
    source = _fold_dir(source_root, dataset, frozen_arm, seed, fold)
    checkpoint = source / 'best.pt'
    source_manifest = source / 'job.json'
    if not checkpoint.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f'frozen {frozen_arm} checkpoint missing under {source}')
    frozen = read_json(source_manifest)
    if frozen.get('status') != 'PASS':
        raise RuntimeError(f"frozen {frozen_arm} fold {fold} is not PASS: {frozen.get('status')}")
    if int(frozen.get('fold', fold)) != int(fold):
        raise RuntimeError(f"refusing to mix fold {frozen.get('fold')} weights into fold {fold}")
    from .data import SceneCache
    from .dataset import dev_index_rows
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload()
    if baseline is None:
        baseline = build_baseline(cfg, dataset)
    baseline_table, tile_baseline = baseline
    rows = dev_index_rows(dataset, cfg)
    samples = build_samples_for_cache(dataset, rows, baseline_table, available=set(cache.entries), tile_baseline=tile_baseline)
    training_all, heldout = fold_split(samples, fold)
    train, validation = partition(training_all, seed=seed)
    trainer = Trainer(dataset=dataset, cache=cache, arm=frozen_arm, seed=seed, device=device)
    trainer.ncr_fill = float(frozen['ncr_fill'])
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(saved['model'])
    trainer.model.eval()
    for parameter in trainer.model.parameters():
        parameter.requires_grad_(False)
    ordered = list(train) + list(validation) + list(heldout)
    records = _encode_samples(trainer, ordered)
    cache_payload = _pack(records, [sample.bag_id for sample in ordered])
    cache_payload['source_checkpoint'] = np.asarray([str(checkpoint)])
    cache_payload['source_arm'] = np.asarray([frozen_arm])
    cache_payload['source_fold'] = np.asarray([int(fold)], np.int64)
    atomic_npz(destination / 'node_cache.npz', **cache_payload)
    train_ids = [sample.bag_id for sample in train]
    val_ids = [sample.bag_id for sample in validation]
    held_ids = [sample.bag_id for sample in heldout]
    packed = {record['sample_id']: record for record in records}
    train_pack = _pack([packed[key] for key in train_ids], train_ids)
    val_pack = _pack([packed[key] for key in val_ids], val_ids)
    held_pack = _pack([packed[key] for key in held_ids], held_ids)
    classes = classes_for(dataset)
    width = int(cache_payload['hidden'].shape[1]) if cache_payload['hidden'].ndim == 2 else 64
    adapter = LocalReadoutAdapter(width=width, hidden=ADAPTER_HIDDEN, classes=classes).to(device)
    expected = width * ADAPTER_HIDDEN + ADAPTER_HIDDEN + ADAPTER_HIDDEN + width * classes
    if adapter.parameter_count() != expected:
        raise RuntimeError(f'unexpected adapter size {adapter.parameter_count()} != {expected}')
    identity, _, _ = _predict_pack(adapter, held_pack, device, classes)
    g_lookup = _rows_by_id(source / 'predictions.parquet')
    g_probability = np.stack([np.asarray(g_lookup[key]['final_probability'], float) for key in held_ids])
    match = float(np.max(np.abs(identity - g_probability)))
    if match > MATCH_ATOL:
        raise RuntimeError(f'initial z_new is not frozen {frozen_arm}: max abs {match}')
    adapter.train()
    q_grad = 0.0
    for start in range(0, len(train_ids), ADAPTER_BATCH):
        probe = _slice_pack(train_pack, start, min(len(train_ids), start + ADAPTER_BATCH))
        output, labels = _adapter_forward(adapter, probe, device)
        probabilities = _probabilities(output, classes)
        _finite(probabilities, output['epsilon'])
        if classes == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output['logits'], labels.float())
        else:
            loss = torch.nn.functional.cross_entropy(output['logits'], labels)
        loss = loss + DEFAULT_DELTA_L2 * output['epsilon'].square().mean()
        adapter.zero_grad(set_to_none=True)
        loss.backward()
        _check_finite_step(loss, adapter)
        q_grad = 0.0 if adapter.query.grad is None else float(adapter.query.grad.abs().sum())
        if q_grad > 0.0:
            break
    if q_grad == 0.0:
        raise RuntimeError('query vector received no gradient on the probe batches')
    adapter.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    val_prob, _, _ = _predict_pack(adapter, val_pack, device, classes)
    best = selection_score(classes, val_prob, val_pack['labels'])
    _finite(val_prob, best)
    history = [{'epoch': -1, 'loss': None, 'selection': best, 'task_metric': metric(dataset, val_pack['labels'], _score(classes, val_prob)), 'baseline_identity': True, 'source_match': match}]
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch(destination / 'adapter.pt', {'adapter': adapter.state_dict(), 'epoch': -1, 'metric': best, 'source_checkpoint': str(checkpoint), 'source_arm': frozen_arm})
    stale = 0
    steps_done = 0
    for epoch in range(int(max_epochs)):
        adapter.train()
        generator = np.random.default_rng(int(seed) + epoch * DEFAULT_EPOCH_SEED_STRIDE)
        order = generator.permutation(len(train_ids))
        running, steps = (0.0, 0)
        for start in range(0, len(order), ADAPTER_BATCH):
            chosen = [train_ids[int(index)] for index in order[start:start + ADAPTER_BATCH]]
            chunk = _pack([packed[key] for key in chosen], chosen)
            chunk['sample_index'] = np.repeat(np.arange(len(chosen)), np.diff(chunk['offsets']))
            output, labels = _adapter_forward(adapter, chunk, device)
            if classes == 1:
                supervised = torch.nn.functional.binary_cross_entropy_with_logits(output['logits'], labels.float())
            else:
                supervised = torch.nn.functional.cross_entropy(output['logits'], labels)
            loss = supervised + DEFAULT_DELTA_L2 * output['epsilon'].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _check_finite_step(loss, adapter)
            optimizer.step()
            for parameter in adapter.parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError('non-finite parameter after optimizer step')
            running += float(loss.detach())
            steps += 1
            steps_done += 1
            if max_steps is not None and steps_done >= max_steps:
                break
        val_prob, _, _ = _predict_pack(adapter, val_pack, device, classes)
        score = selection_score(classes, val_prob, val_pack['labels'])
        _finite(val_prob, score)
        history.append({'epoch': epoch, 'loss': running / max(1, steps), 'selection': score, 'task_metric': metric(dataset, val_pack['labels'], _score(classes, val_prob))})
        if np.isfinite(score) and score > best + 1e-12:
            best, stale = (score, 0)
            atomic_torch(destination / 'adapter.pt', {'adapter': adapter.state_dict(), 'epoch': epoch, 'metric': best, 'source_checkpoint': str(checkpoint), 'source_arm': frozen_arm})
        else:
            stale += 1
            if stale >= int(patience):
                break
        if max_steps is not None and steps_done >= max_steps:
            break
    saved_adapter = torch.load(destination / 'adapter.pt', map_location=device, weights_only=False)
    adapter.load_state_dict(saved_adapter['adapter'])
    held_prob, held_eps, held_weight = _predict_pack(adapter, held_pack, device, classes)
    _finite(held_prob, held_eps, held_weight)
    atomic_parquet(destination / 'predictions.parquet', _heldout_rows(heldout, arm, seed, baseline_probability=held_pack['baseline'], final_probability=held_prob, labels=held_pack['labels']))
    atomic_npz(destination / 'heldout_attention.npz', sample_id=np.asarray(held_ids, object), offsets=held_pack['offsets'], weight=held_weight, xy=held_pack['xy'], z=held_pack['z'], valid=held_pack['valid'], epsilon=held_eps, labels=held_pack['labels'], group_id=np.asarray(held_pack['group_id'], object))
    manifest = {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'source_arm': frozen_arm, 'source_checkpoint': str(checkpoint), 'fold': int(fold), 'seed': int(seed), 'samples': len(samples), 'train_bags': len(train), 'validation_bags': len(validation), 'heldout_bags': len(heldout), 'parameters': adapter.parameter_count(), 'best_metric': best, 'epochs': len(history), 'ncr_fill': trainer.ncr_fill, 'history': history, 'source_match': match, 'device': device, 'official_test_touched': False}
    atomic_json(manifest_path, manifest)
    return manifest

def explain_dataset(cfg: Mapping[str, Any], dataset: str, seed: int, arm: str='G-Local', per_kind: int=8) -> dict[str, Any]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result_root = Path(cfg['paths']['result_root'])
    source_root = _source_root(cfg)
    explain_root = result_root / dataset / 'explain' / arm / f'seed_{seed}'
    explain_root.mkdir(parents=True, exist_ok=True)
    from .dataset import DATASET_SPEC
    rows = []
    for fold in range(int(DATASET_SPEC[dataset]['folds'])):
        local_path = _fold_dir(result_root, dataset, arm, seed, fold) / 'predictions.parquet'
        base_path = _fold_dir(source_root, dataset, 'B', seed, fold) / 'predictions.parquet'
        geo_path = _fold_dir(source_root, dataset, source_arm(arm), seed, fold) / 'predictions.parquet'
        attention_path = _fold_dir(result_root, dataset, arm, seed, fold) / 'heldout_attention.npz'
        local = _rows_by_id(local_path)
        frozen_b = _rows_by_id(base_path)
        frozen_g = _rows_by_id(geo_path)
        with __import__('numpy').load(attention_path, allow_pickle=True) as loaded:
            attention = {key: loaded[key].copy() for key in loaded.files}
        id_to_pos = {str(key): index for index, key in enumerate(attention['sample_id'].tolist())}
        for key in local:
            label = int(local[key]['label_id'])
            pred_b = int(np.asarray(frozen_b[key]['final_probability']).argmax())
            pred_new = int(np.asarray(local[key]['final_probability']).argmax())
            pred_g = int(np.asarray(frozen_g[key]['final_probability']).argmax())
            kind = 'keep'
            if pred_b != label and pred_new == label:
                kind = 'fix'
            elif pred_b == label and pred_new != label:
                kind = 'break'
            position = id_to_pos[key]
            start = int(attention['offsets'][position])
            stop = int(attention['offsets'][position + 1])
            mass = np.asarray(attention['weight'][start:stop], float)
            valid = np.asarray(attention['valid'][start:stop], bool)
            entropy = float(-(mass[valid] * np.log(np.clip(mass[valid], 1e-12, 1.0))).sum()) if valid.any() else 0.0
            rows.append({'sample_id': key, 'fold': fold, 'group_id': str(local[key]['group_id']), 'label_id': label, 'pred_b': pred_b, 'pred_g': pred_g, 'pred_new': pred_new, 'kind': kind, 'attention_entropy': entropy, 'epsilon_l2': float(np.linalg.norm(np.asarray(attention['epsilon'][position], float)))})
    fixes = sorted((row for row in rows if row['kind'] == 'fix'), key=lambda row: -row['epsilon_l2'])
    breaks = sorted((row for row in rows if row['kind'] == 'break'), key=lambda row: -row['epsilon_l2'])
    selected = fixes[:per_kind] + breaks[:per_kind]
    figures = []
    by_fold = {}
    for fold in range(int(DATASET_SPEC[dataset]['folds'])):
        path = _fold_dir(result_root, dataset, arm, seed, fold) / 'heldout_attention.npz'
        with __import__('numpy').load(path, allow_pickle=True) as loaded:
            by_fold[fold] = {key: loaded[key].copy() for key in loaded.files}
    for row in selected:
        payload = by_fold[int(row['fold'])]
        position = {str(key): index for index, key in enumerate(payload['sample_id'].tolist())}[row['sample_id']]
        start = int(payload['offsets'][position])
        stop = int(payload['offsets'][position + 1])
        xy = np.asarray(payload['xy'][start:stop])
        depth = np.asarray(payload['z'][start:stop])
        mass = np.asarray(payload['weight'][start:stop])
        valid = np.asarray(payload['valid'][start:stop], bool)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=120)
        axes[0].scatter(xy[~valid, 0], xy[~valid, 1], s=8, c='#bbbbbb', linewidths=0)
        points = axes[0].scatter(xy[valid, 0], xy[valid, 1], s=18, c=mass[valid], cmap='magma', linewidths=0)
        axes[0].set_title(f"{row['kind']} {row['sample_id']} attention")
        axes[0].set_xlabel('x (µm)')
        axes[0].set_ylabel('y (µm)')
        axes[0].set_aspect('equal', adjustable='box')
        fig.colorbar(points, ax=axes[0], fraction=0.046, pad=0.04)
        axes[1].scatter(xy[valid, 0], xy[valid, 1], s=18, c=depth[valid], cmap='coolwarm', linewidths=0)
        axes[1].set_title('reconstructed z')
        axes[1].set_xlabel('x (µm)')
        axes[1].set_ylabel('y (µm)')
        axes[1].set_aspect('equal', adjustable='box')
        fig.tight_layout()
        name = f"fold{row['fold']:02d}_{row['kind']}_{row['sample_id']}.png"
        fig.savefig(explain_root / name)
        plt.close(fig)
        figures.append(name)
        row['figure'] = name
    payload = {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'seed': seed, 'samples': len(rows), 'n_fix': sum((row['kind'] == 'fix' for row in rows)), 'n_break': sum((row['kind'] == 'break' for row in rows)), 'n_keep': sum((row['kind'] == 'keep' for row in rows)), 'selected': selected, 'figures': figures, 'metric_note': 'counts do not replace QWK', 'official_test_touched': False}
    atomic_json(explain_root / 'summary.json', payload)
    atomic_parquet(explain_root / 'heldout_kinds.parquet', rows)
    return payload
