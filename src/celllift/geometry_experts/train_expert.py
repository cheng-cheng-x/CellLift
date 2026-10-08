from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import hashlib
import numpy as np
from .baseline import baseline_probability, classes_for, load_baseline
from .data import Sample, SceneCache, build_batch, group_batches, group_equal_stats, source_uniform_batches
from .dataset import DATASET_SPEC, EXPERT_ARMS, batch_result_root, dev_index_rows, train_rows
from .io_utils import atomic_json, atomic_parquet, atomic_torch, read_json
from .models import GeometryExpert, PROB_CLIP
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_BAGS = 8
DEFAULT_TRAIN_TILES = 32
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_EPOCH_STRIDE = 1009
INNER_SEED = 20260909

def grouped_assignment(groups: Sequence[str], folds: int=3, seed: int=INNER_SEED) -> dict[str, int]:
    unique = sorted(set(map(str, groups)), key=lambda value: hashlib.sha256(f'{seed}|{value}'.encode()).digest())
    return {value: index % folds for index, value in enumerate(unique)}

def build_samples(dataset: str, rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], available: set[str]) -> list[Sample]:
    output: list[Sample] = []
    if dataset == 'sicapv2':
        for row in rows:
            if row['graph_id'] not in available:
                continue
            entry = baseline.get(row['graph_id'])
            if entry is None:
                continue
            probability = np.asarray(baseline_probability(dataset, entry), np.float32)
            output.append(Sample(row['graph_id'], [row['graph_id']], int(entry['label_id']), int(entry['fold']), str(entry['group_id']), probability))
        return output
    key = 'roi_id' if dataset == 'bracs' else 'patient_id'
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['graph_id'] in available:
            grouped[str(row[key])].append(row)
    minimum = int(DATASET_SPEC['tcga_crc_msi'].get('min_tiles', 10))
    for bag_id, members in sorted(grouped.items()):
        entry = baseline.get(bag_id)
        if entry is None:
            continue
        if dataset == 'tcga_crc_msi' and len(members) < minimum:
            continue
        probability = np.asarray(baseline_probability(dataset, entry), np.float32)
        output.append(Sample(bag_id, [str(member['graph_id']) for member in members], int(entry['label_id']), int(entry['fold']), str(entry['group_id']), probability))
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

def subsample(sample: Sample, rng: np.random.Generator, maximum: int) -> Sample:
    if len(sample.graph_ids) <= maximum:
        return sample
    chosen = np.sort(rng.choice(len(sample.graph_ids), maximum, replace=False))
    return Sample(sample.bag_id, [sample.graph_ids[int(i)] for i in chosen], sample.label_id, sample.fold, sample.group_id, sample.baseline)

def _finite(*values) -> None:
    import torch
    for value in values:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('non-finite tensor during expert training')
        else:
            array = np.asarray(value, float)
            if not np.isfinite(array).all():
                raise RuntimeError('non-finite array during expert training')

def _probabilities(logits, classes: int):
    import torch
    return torch.sigmoid(logits) if classes == 1 else torch.softmax(logits, -1)

def _loss(logits, labels, classes: int):
    import torch
    if classes == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
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

def torch_seed_for(seed: int, outer: int, inner: int) -> int:
    return int(seed) + 10007 * int(outer) + 17 * int(inner)

def job_dir(cfg: Mapping[str, Any], dataset: str, arm: str, seed: int, outer: int, inner: int) -> Path:
    return batch_result_root(cfg) / dataset / 'experts' / arm / f'seed_{seed}' / f'outer_{outer:02d}' / f'inner_{inner:02d}'

def _predict(model, cache, samples, arm, mean, std, fill3d, device, classes, bag_limit=None):
    import torch
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(samples), DEFAULT_BAGS):
            chunk = list(samples[start:start + DEFAULT_BAGS])
            if bag_limit is not None:
                rng = np.random.default_rng(0)
                chunk = [subsample(sample, rng, bag_limit) if len(sample.graph_ids) > bag_limit else sample for sample in chunk]
            batch = build_batch(cache, chunk, arm=arm, mean=mean, std=std, fill3d=fill3d, device=device)
            logits = model(batch)['logits']
            _finite(logits)
            probabilities = _probabilities(logits, classes).cpu().numpy()
            for sample, probability in zip(chunk, probabilities):
                rows.append({'sample_id': sample.bag_id, 'fold': sample.fold, 'group_id': sample.group_id, 'label_id': sample.label_id, 'baseline_probability': np.asarray(sample.baseline, float).tolist(), 'final_probability': np.asarray(probability, float).tolist()})
    return rows

def train_inner(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, inner: int, seed: int, device: str, cache: SceneCache | None=None, baseline=None, max_epochs: int=DEFAULT_MAX_EPOCHS, patience: int=DEFAULT_PATIENCE) -> dict[str, Any]:
    import torch
    if arm not in EXPERT_ARMS:
        raise ValueError(arm)
    destination = job_dir(cfg, dataset, arm, seed, outer, inner)
    manifest_path = destination / 'job.json'
    if manifest_path.is_file() and read_json(manifest_path).get('status') == 'PASS':
        return {'status': 'REUSED', **read_json(manifest_path)}
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload()
    if baseline is None:
        baseline = load_baseline(dataset, cfg)
    rows = train_rows(dev_index_rows(dataset, cfg))
    samples = build_samples(dataset, rows, baseline, set(cache.graphs))
    outer_train, outer_heldout = fold_split(samples, outer)
    assignment = grouped_assignment([sample.group_id for sample in outer_train], int(DATASET_SPEC[dataset]['inner_folds']), INNER_SEED + int(outer))
    inner_oof = [sample for sample in outer_train if assignment[sample.group_id] == inner]
    inner_fit = [sample for sample in outer_train if assignment[sample.group_id] != inner]
    if not inner_oof or not inner_fit:
        raise RuntimeError(f'empty inner split outer={outer} inner={inner}')
    train, validation = partition(inner_fit, seed=seed + 17 * inner)
    graph_ids = [graph_id for sample in train for graph_id in sample.graph_ids]
    groups = [sample.group_id for sample in train for _ in sample.graph_ids]
    mean, std, fill3d = group_equal_stats(cache, graph_ids, groups)
    classes = classes_for(dataset)
    bag = dataset != 'sicapv2'
    torch_seed = torch_seed_for(seed, outer, inner)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    model = GeometryExpert(classes=classes, bag=bag).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    best, stale, history = (float('inf'), 0, [])
    first_sampling: dict[str, Any] | None = None
    destination.mkdir(parents=True, exist_ok=True)

    def eval_nll(split_samples):
        model.eval()
        total, count = (0.0, 0)
        with torch.no_grad():
            for chunk in group_batches(split_samples, DEFAULT_BAGS, np.random.default_rng(0)):
                batch = build_batch(cache, chunk, arm=arm, mean=mean, std=std, fill3d=fill3d, device=device)
                logits = model(batch)['logits']
                loss = _loss(logits, batch.labels, classes)
                _finite(loss, logits)
                total += float(loss) * len(chunk)
                count += len(chunk)
        return total / max(1, count)
    for epoch in range(int(max_epochs)):
        model.train()
        rng = np.random.default_rng(int(seed) + epoch * DEFAULT_EPOCH_STRIDE)
        running, steps, skipped = (0.0, 0, 0)
        batches, sampling = source_uniform_batches(train, DEFAULT_BAGS, rng)
        if first_sampling is None:
            first_sampling = sampling
        for chunk in batches:
            if bag:
                chunk = [subsample(sample, rng, DEFAULT_TRAIN_TILES) for sample in chunk]
            batch = build_batch(cache, chunk, arm=arm, mean=mean, std=std, fill3d=fill3d, device=device)
            logits = model(batch)['logits']
            _finite(logits)
            loss = _loss(logits, batch.labels, classes)
            _finite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if any((not bool(torch.isfinite(grad).all()) for grad in grads)):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for parameter in model.parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise RuntimeError('non-finite parameter after step')
            running += float(loss.detach())
            steps += 1
        if steps == 0:
            raise RuntimeError(f'all {skipped} training batches had non-finite gradients')
        nll = eval_nll(validation)
        history.append({'epoch': epoch, 'loss': running / max(1, steps), 'selection': -nll, 'val_nll': nll, 'skipped_batches': skipped})
        if np.isfinite(nll) and nll < best - 1e-12:
            best, stale = (nll, 0)
            atomic_torch(destination / 'best.pt', {'model': model.state_dict(), 'mean': mean, 'std': std, 'fill3d': fill3d, 'epoch': epoch, 'arm': arm})
        else:
            stale += 1
            if stale >= int(patience):
                break
    checkpoint = destination / 'best.pt'
    if not checkpoint.is_file() or not np.isfinite(best):
        raise RuntimeError('no finite validation checkpoint')
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    oof_rows = _predict(model, cache, inner_oof, arm, mean, std, fill3d, device, classes)
    held_rows = _predict(model, cache, outer_heldout, arm, mean, std, fill3d, device, classes)
    atomic_parquet(destination / 'oof.parquet', oof_rows)
    atomic_parquet(destination / 'heldout.parquet', held_rows)
    oof_prob = np.stack([row['final_probability'] for row in oof_rows])
    oof_labels = np.asarray([row['label_id'] for row in oof_rows])
    manifest = {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'outer': int(outer), 'inner': int(inner), 'seed': int(seed), 'parameters': model.parameter_count(), 'epochs': len(history), 'best_nll': best, 'history': history, 'oof_bags': len(oof_rows), 'heldout_bags': len(held_rows), 'oof_metric': _metric(dataset, oof_labels, oof_prob), 'reads_baseline': False, 'official_test_touched': False, 'device': device, 'torch_seed': torch_seed, 'sampling': (first_sampling or {}).get('sampling', 'source_uniform_replacement'), 'steps_per_epoch': (first_sampling or {}).get('steps_per_epoch'), 'max_run_same_group': (first_sampling or {}).get('max_run_same_group')}
    atomic_json(manifest_path, manifest)
    return manifest

def job_matrix(dataset: str, seed: int=42) -> list[dict]:
    spec = DATASET_SPEC[dataset]
    jobs = []
    for arm in EXPERT_ARMS:
        for outer in range(int(spec['folds'])):
            for inner in range(int(spec['inner_folds'])):
                jobs.append({'dataset': dataset, 'arm': arm, 'outer': outer, 'inner': inner, 'seed': seed})
    return jobs
