from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from ..baseline import baseline_probability, classes_for, load_baseline
from ..data import Sample, SceneCache, build_batch, group_batches, group_equal_edge_stats, group_equal_stats, prepare_standardization, source_uniform_batches
from ..dataset import B_ROLE, DATASET_SPEC, PREDICTION_FORM, batch_result_root, dev_index_rows, train_rows
from ..io_utils import atomic_json, atomic_parquet, atomic_torch, read_json
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_BAGS = 4
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_EPOCH_STRIDE = 1009

def build_samples(dataset: str, rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], available: set[str]) -> list[Sample]:
    output: list[Sample] = []
    classes = classes_for(dataset)
    dummy = np.full(max(1, classes), 1.0 / max(1, classes), np.float32)
    if dataset == 'sicapv2':
        for row in rows:
            if row['graph_id'] not in available:
                continue
            entry = baseline.get(row['graph_id'])
            probability = np.asarray(baseline_probability(dataset, entry), np.float32) if entry else dummy
            label = int(entry['label_id']) if entry else int(row['label_id'])
            group = str(entry['group_id']) if entry else str(row['group_id'])
            fold = int(entry['fold']) if entry and entry.get('fold') is not None else 0 if row['fold'] is None else int(row['fold'])
            output.append(Sample(row['graph_id'], [row['graph_id']], label, fold, group, probability))
        return output
    key = 'roi_id' if dataset == 'bracs' else 'patient_id'
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['graph_id'] in available:
            grouped[str(row[key])].append(row)
    minimum = int(DATASET_SPEC['tcga_crc_msi'].get('min_tiles', 10))
    for bag_id, members in sorted(grouped.items()):
        entry = baseline.get(bag_id)
        if dataset == 'tcga_crc_msi' and len(members) < minimum:
            continue
        if entry is None and dataset != 'sicapv2':
            labels = {int(member['label_id']) for member in members if member['label_id'] is not None}
            if len(labels) != 1:
                continue
            label = labels.pop()
            fold = 0 if members[0]['fold'] is None else int(members[0]['fold'])
            group = str(members[0]['group_id'])
            probability = dummy.copy()
        else:
            if entry is None:
                continue
            label = int(entry['label_id'])
            fold = int(entry['fold'])
            group = str(entry['group_id'])
            probability = np.asarray(baseline_probability(dataset, entry), np.float32)
        output.append(Sample(bag_id, [str(member['graph_id']) for member in members], label, fold, group, probability))
    return output

def fold_split(samples: Sequence[Sample], fold: int) -> tuple[list[Sample], list[Sample]]:
    training = [sample for sample in samples if sample.fold != fold]
    heldout = [sample for sample in samples if sample.fold == fold]
    if not training or not heldout:
        raise RuntimeError(f'empty train/heldout for fold {fold}')
    return (training, heldout)

def partition(samples: Sequence[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    unique = sorted({sample.group_id for sample in samples})
    if len(unique) < 2:
        return (list(samples), list(samples))
    rng = np.random.default_rng(seed)
    order = np.asarray(unique, object)
    rng.shuffle(order)
    count = min(max(1, int(round(len(unique) * DEFAULT_VAL_FRACTION))), len(unique) - 1)
    validation_groups = set(order[:count].tolist())
    train = [sample for sample in samples if sample.group_id not in validation_groups]
    validation = [sample for sample in samples if sample.group_id in validation_groups]
    return (train, validation)

def _probabilities(logits, classes: int):
    import torch
    return torch.sigmoid(logits) if classes == 1 else torch.softmax(logits, -1)

def _loss(logits, labels, classes: int):
    import torch
    if classes == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float())
    return torch.nn.functional.cross_entropy(logits, labels)

def _metric(dataset: str, labels: np.ndarray, probabilities: np.ndarray) -> float:
    if dataset == 'sicapv2':
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(labels, probabilities.argmax(1), weights='quadratic'))
    if dataset == 'bracs':
        from sklearn.metrics import f1_score
        return float(f1_score(labels, probabilities.argmax(1), average='macro'))
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(roc_auc_score(labels, probabilities[:, 0] if probabilities.shape[1] == 1 else probabilities[:, 1]))

def torch_seed_for(seed: int, outer: int) -> int:
    return int(seed) + 10007 * int(outer)

def job_dir(cfg: Mapping[str, Any], dataset: str, route: str, arm: str, seed: int, outer: int) -> Path:
    return batch_result_root(cfg, route) / dataset / 'jobs' / arm / f'seed_{seed}' / f'outer_{outer:02d}'

def prediction_dir(cfg: Mapping[str, Any], dataset: str, route: str, arm: str, seed: int, outer: int) -> Path:
    return batch_result_root(cfg, route) / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{outer:02d}'

def _stats(cache, samples):
    graph_ids = [graph_id for sample in samples for graph_id in sample.graph_ids]
    groups = [sample.group_id for sample in samples for _ in sample.graph_ids]
    mean, std, fill3d = group_equal_stats(cache, graph_ids, groups)
    edge_mean2d, edge_std2d, edge_mean3d, edge_std3d = group_equal_edge_stats(cache, graph_ids, groups)
    return (mean, std, fill3d, edge_mean2d, edge_std2d, edge_mean3d, edge_std3d)

def _batch_kwargs(mean, std, fill3d, edge_mean2d, edge_std2d, edge_mean3d, edge_std3d, arm, device, with_spatial=False):
    return dict(arm=arm, mean=mean, std=std, fill3d=fill3d, edge_mean2d=edge_mean2d, edge_std2d=edge_std2d, edge_mean3d=edge_mean3d, edge_std3d=edge_std3d, device=device, with_spatial=with_spatial)

def _early_stop_choice(dataset: str, train_labels: np.ndarray, val_labels: np.ndarray) -> str:
    train_set = set((int(value) for value in train_labels))
    val_set = set((int(value) for value in val_labels))
    if dataset == 'tcga_crc_msi':
        return 'nll' if len(val_set) < 2 else 'mean_fold_patient_auroc'
    if not train_set.issubset(val_set):
        return 'nll'
    return DATASET_SPEC[dataset]['metric']

def train_full(cfg: Mapping[str, Any], dataset: str, route: str, arm: str, outer: int, seed: int, device: str, build_model: Callable[..., Any], *, cache: SceneCache | None=None, published=None, max_epochs: int=DEFAULT_MAX_EPOCHS, patience: int=DEFAULT_PATIENCE, with_spatial: bool=False, extra_loss: Callable | None=None, init_state: Mapping[str, Any] | None=None, pretrain_pool: str='unknown', lr: float=DEFAULT_LR, stage: str='task', official_fit: Sequence[str] | None=None, official_val: Sequence[str] | None=None, official_predict: Sequence[str] | None=None, fixed_epochs: int | None=None) -> dict[str, Any]:
    import torch
    destination = job_dir(cfg, dataset, route, arm, seed, outer)
    manifest_path = destination / 'job.json'
    if manifest_path.is_file() and read_json(manifest_path).get('status') == 'PASS':
        return {'status': 'REUSED', **read_json(manifest_path)}
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload(load_spatial_maps=with_spatial)
    if official_fit is not None:
        published = {}
    elif published is None:
        published = load_baseline(dataset, cfg)
    index_rows = dev_index_rows(dataset, cfg)
    rows = index_rows if official_fit is not None else train_rows(index_rows)
    samples = build_samples(dataset, rows, published, set(cache.graphs))
    if official_fit is not None:
        fit_ids = set(map(str, official_fit))
        val_ids = set(map(str, official_val or []))
        pred_ids = set(map(str, official_predict or official_val or []))
        train = [sample for sample in samples if sample.bag_id in fit_ids]
        validation = [sample for sample in samples if sample.bag_id in val_ids] or train
        outer_heldout = [sample for sample in samples if sample.bag_id in pred_ids] or validation
    else:
        outer_train, outer_heldout = fold_split(samples, outer)
        train, validation = partition(outer_train, seed=seed + 17 * outer)
    mean, std, fill3d, e2m, e2s, e3m, e3s = _stats(cache, train)
    kwargs = _batch_kwargs(mean, std, fill3d, e2m, e2s, e3m, e3s, arm, device, with_spatial)
    prepare_standardization(cache, **{k: v for k, v in kwargs.items() if k not in {'device', 'with_spatial'}})
    classes = classes_for(dataset)
    torch_seed = torch_seed_for(seed, outer)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    model = build_model(classes=classes, arm=arm, bag=dataset != 'sicapv2').to(device)
    if init_state:
        current = model.state_dict()
        compatible = {key: value for key, value in init_state.items() if key in current and getattr(value, 'shape', None) == current[key].shape}
        model.load_state_dict(compatible, strict=False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=DEFAULT_WEIGHT_DECAY)
    early_stop_metric = _early_stop_choice(dataset, np.asarray([sample.label_id for sample in train]), np.asarray([sample.label_id for sample in validation]))
    best, stale, history = (-float('inf'), 0, [])
    first_sampling = None
    destination.mkdir(parents=True, exist_ok=True)

    def eval_split(split_samples):
        model.eval()
        labels, probabilities = ([], [])
        total, count = (0.0, 0)
        with torch.no_grad():
            for chunk in group_batches(split_samples, DEFAULT_BAGS, np.random.default_rng(0)):
                batch = build_batch(cache, chunk, **kwargs)
                output = model(batch)
                logits = output['logits']
                loss = _loss(logits, batch.labels, classes)
                total += float(loss) * len(chunk)
                count += len(chunk)
                probs = _probabilities(logits, classes).cpu().numpy()
                if classes == 1:
                    probs = probs.reshape(-1, 1)
                labels.extend((int(sample.label_id) for sample in chunk))
                probabilities.append(probs)
        nll = total / max(1, count)
        stacked = np.concatenate(probabilities, 0) if probabilities else np.zeros((0, max(1, classes)))
        return (nll, _metric(dataset, np.asarray(labels), stacked))
    epoch_limit = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    stop_patience = 10 ** 9 if fixed_epochs is not None else int(patience)
    for epoch in range(epoch_limit):
        model.train()
        rng = np.random.default_rng(int(seed) + epoch * DEFAULT_EPOCH_STRIDE)
        running, steps, skipped = (None, 0, 0)
        batches, sampling = source_uniform_batches(train, DEFAULT_BAGS, rng)
        if first_sampling is None:
            first_sampling = sampling
        for chunk in batches:
            batch = build_batch(cache, chunk, **kwargs)
            output = model(batch)
            logits = output['logits']
            loss = _loss(logits, batch.labels, classes)
            if extra_loss is not None:
                loss = loss + extra_loss(model, batch, output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            running = loss.detach() if running is None else running + loss.detach()
            steps += 1
        if steps == 0:
            raise RuntimeError(f'all {skipped} training batches had non-finite gradients')
        nll, metric = eval_split(validation)
        if early_stop_metric == 'nll' or not np.isfinite(metric):
            selection, used = (-nll, 'nll')
        else:
            selection, used = (metric, early_stop_metric)
        history.append({'epoch': epoch, 'loss': float(running) / max(1, steps), 'val_nll': nll, 'val_metric': metric, 'selection': selection, 'early_stop_used': used, 'skipped_batches': skipped})
        if fixed_epochs is not None:
            best, stale = (selection, 0)
            atomic_torch(destination / 'best.pt', {'model': model.state_dict(), 'mean': mean, 'std': std, 'fill3d': fill3d, 'edge_mean2d': e2m, 'edge_std2d': e2s, 'edge_mean3d': e3m, 'edge_std3d': e3s, 'epoch': epoch, 'arm': arm, 'route': route})
        elif np.isfinite(selection) and selection > best + 1e-12:
            best, stale = (selection, 0)
            atomic_torch(destination / 'best.pt', {'model': model.state_dict(), 'mean': mean, 'std': std, 'fill3d': fill3d, 'edge_mean2d': e2m, 'edge_std2d': e2s, 'edge_mean3d': e3m, 'edge_std3d': e3s, 'epoch': epoch, 'arm': arm, 'route': route})
        else:
            stale += 1
            if stale >= stop_patience:
                break
    checkpoint = destination / 'best.pt'
    if not checkpoint.is_file() or not np.isfinite(best):
        raise RuntimeError('no finite validation checkpoint')
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval()
    held_rows = []
    with torch.no_grad():
        for start in range(0, len(outer_heldout), DEFAULT_BAGS):
            chunk = list(outer_heldout[start:start + DEFAULT_BAGS])
            batch = build_batch(cache, chunk, **kwargs)
            logits = model(batch)['logits']
            probabilities = _probabilities(logits, classes).cpu().numpy()
            if classes == 1:
                probabilities = probabilities.reshape(-1, 1)
            for sample, probability in zip(chunk, probabilities):
                held_rows.append({'sample_id': sample.bag_id, 'fold': sample.fold, 'group_id': sample.group_id, 'label_id': sample.label_id, 'baseline_probability': np.asarray(sample.baseline, float).tolist(), 'final_probability': np.asarray(probability, float).tolist()})
    pred_root = prediction_dir(cfg, dataset, route, arm, seed, outer)
    pred_root.mkdir(parents=True, exist_ok=True)
    atomic_parquet(pred_root / 'predictions.parquet', held_rows)
    atomic_parquet(destination / 'heldout.parquet', held_rows)
    held_prob = np.stack([row['final_probability'] for row in held_rows])
    held_labels = np.asarray([row['label_id'] for row in held_rows])
    count = int(sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad)))
    manifest = {'status': 'PASS', 'dataset': dataset, 'route': route, 'arm': arm, 'outer': int(outer), 'seed': int(seed), 'stage': stage, 'parameters': count, 'epochs': len(history), 'best_selection': best, 'history': history, 'heldout_bags': len(held_rows), 'heldout_metric': _metric(dataset, held_labels, held_prob), 'prediction_form': PREDICTION_FORM, 'b_role': B_ROLE, 'early_stop_metric': early_stop_metric, 'official_test_touched': False, 'device': device, 'torch_seed': torch_seed, 'sampling': (first_sampling or {}).get('sampling', 'source_uniform_replacement'), 'steps_per_epoch': (first_sampling or {}).get('steps_per_epoch'), 'max_run_same_group': (first_sampling or {}).get('max_run_same_group'), 'pretrain_pool': pretrain_pool, 'reads_baseline': False}
    atomic_json(manifest_path, manifest)
    return manifest
