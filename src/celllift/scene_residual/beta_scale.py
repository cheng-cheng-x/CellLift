from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .io_utils import atomic_json, atomic_parquet, read_json, read_parquet
from .train import _heldout_rows, build_baseline, build_samples_for_cache
from .training import Trainer, classes_for, fold_split, metric, partition, _score
BETA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
MATCH_ATOL = 1e-05
ARM = 'G-beta'

def probabilities_from_residual(baseline: np.ndarray, delta: np.ndarray, beta: float | np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline, np.float64)
    delta = np.asarray(delta, np.float64)
    if delta.shape != baseline.shape:
        raise ValueError(f'delta shape {delta.shape} does not match baseline {baseline.shape}')
    betas = np.atleast_1d(np.asarray(beta, np.float64).reshape(-1))
    binary = baseline.shape[-1] == 1
    if binary:
        probability = np.clip(baseline[:, 0], 1e-07, 1.0 - 1e-07)
        anchor = np.log(probability) - np.log1p(-probability)
        logits = anchor[None, :] + betas[:, None] * delta[None, :, 0]
        scaled = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        result = scaled[..., None]
    else:
        probability = np.clip(baseline, 1e-07, 1.0)
        probability = probability / probability.sum(-1, keepdims=True)
        anchor = np.log(probability)
        logits = anchor[None, ...] + betas[:, None, None] * delta[None, ...]
        logits = logits - logits.max(-1, keepdims=True)
        exponent = np.exp(logits)
        result = exponent / exponent.sum(-1, keepdims=True)
    if np.ndim(beta) == 0:
        return result[0].astype(np.float32)
    return result.astype(np.float32)

def choose_beta(scores: Mapping[float, float]) -> float:
    finite = [(float(beta), float(score)) for beta, score in scores.items() if np.isfinite(score)]
    if not finite:
        raise RuntimeError('no finite inner-validation scores for any β')
    finite.sort(key=lambda item: (-item[1], abs(item[0] - 1.0), item[0]))
    return finite[0][0]

def _source_root(cfg: Mapping[str, Any]) -> Path:
    paths = cfg['paths']
    return Path(paths.get('source_root') or paths['result_root'])

def _fold_dir(root: Path, dataset: str, arm: str, seed: int, fold: int) -> Path:
    return root / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{fold:02d}'

def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'missing reference predictions: {path}')
    return {str(row['sample_id']): row for row in read_parquet(path)}

def _stack_field(lookup: Mapping[str, Mapping[str, Any]], order: Sequence[str], field: str) -> np.ndarray:
    missing = [key for key in order if key not in lookup]
    if missing:
        raise RuntimeError(f'reference parquet missing {len(missing)} samples, e.g. {missing[:3]}')
    stacked = np.stack([np.asarray(lookup[key][field], float) for key in order])
    if stacked.ndim == 1:
        stacked = stacked[:, None]
    return stacked

def _task_metric(dataset: str, labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(metric(dataset, labels, _score(classes_for(dataset), probabilities), folds=None))

def scale_fold(cfg: Mapping[str, Any], dataset: str, arm: str, fold: int, seed: int, device: str, *, cache=None, baseline=None, **_ignored) -> dict[str, Any]:
    import torch
    from .data import SceneCache
    from .dataset import dev_index_rows
    if arm != ARM:
        raise ValueError(f'beta-scale jobs must use arm {ARM}, got {arm}')
    result_root = Path(cfg['paths']['result_root'])
    source_root = _source_root(cfg)
    destination = _fold_dir(result_root, dataset, ARM, seed, fold)
    manifest_path = destination / 'job.json'
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get('status') == 'PASS':
            return {'status': 'REUSED', **previous}
    source = _fold_dir(source_root, dataset, 'G', seed, fold)
    checkpoint = source / 'best.pt'
    source_manifest = source / 'job.json'
    if not checkpoint.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f'frozen G checkpoint missing under {source}')
    frozen = read_json(source_manifest)
    if frozen.get('status') != 'PASS':
        raise RuntimeError(f"frozen G fold {fold} is not PASS: {frozen.get('status')}")
    if cache is None:
        cache = SceneCache(cfg, dataset)
        cache.preload()
    if baseline is None:
        baseline = build_baseline(cfg, dataset)
    baseline_table, tile_baseline = baseline
    rows = dev_index_rows(dataset, cfg)
    samples = build_samples_for_cache(dataset, rows, baseline_table, available=set(cache.entries), tile_baseline=tile_baseline)
    training_all, heldout = fold_split(samples, fold)
    _, validation = partition(training_all, seed=seed)
    trainer = Trainer(dataset=dataset, cache=cache, arm='G', seed=seed, device=device)
    trainer.ncr_fill = float(frozen['ncr_fill'])
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    trainer.model.load_state_dict(saved['model'])
    val_baseline, val_final, val_labels, val_delta = trainer.predict(validation, residual=True)
    held_baseline, held_final, held_labels, held_delta = trainer.predict(heldout, residual=True)
    order = [sample.bag_id for sample in heldout]
    g_lookup = _rows_by_id(source / 'predictions.parquet')
    b_lookup = _rows_by_id(_fold_dir(source_root, dataset, 'B', seed, fold) / 'predictions.parquet')
    g_probability = _stack_field(g_lookup, order, 'final_probability')
    b_probability = _stack_field(b_lookup, order, 'final_probability')
    beta0 = probabilities_from_residual(held_baseline, held_delta, 0.0)
    beta1 = probabilities_from_residual(held_baseline, held_delta, 1.0)
    match_beta0 = float(np.max(np.abs(beta0 - b_probability)))
    match_beta1_parquet = float(np.max(np.abs(held_final - g_probability)))
    match_beta1_apply = float(np.max(np.abs(beta1 - held_final)))
    if match_beta0 > MATCH_ATOL:
        raise RuntimeError(f'β=0 heldout is not frozen B: max abs {match_beta0}')
    if match_beta1_parquet > MATCH_ATOL:
        raise RuntimeError(f'loaded G does not match G parquet: max abs {match_beta1_parquet}')
    if match_beta1_apply > MATCH_ATOL:
        raise RuntimeError(f'β=1 apply does not match G forward: max abs {match_beta1_apply}')
    val_grid = probabilities_from_residual(val_baseline, val_delta, np.asarray(BETA_GRID))
    val_scores = {}
    for index, beta in enumerate(BETA_GRID):
        val_scores[float(beta)] = _task_metric(dataset, val_labels, val_grid[index])
    selected = choose_beta(val_scores)
    held_grid = probabilities_from_residual(held_baseline, held_delta, np.asarray(BETA_GRID))
    selected_index = int(np.argmin(np.abs(np.asarray(BETA_GRID) - selected)))
    selected_probability = held_grid[selected_index]
    if abs(selected - 1.0) < 1e-12:
        selected_probability = held_final
    destination.mkdir(parents=True, exist_ok=True)
    atomic_parquet(destination / 'predictions.parquet', _heldout_rows(heldout, ARM, seed, baseline_probability=held_baseline, final_probability=selected_probability, labels=held_labels))
    grid_rows = []
    for index, beta in enumerate(BETA_GRID):
        probability = held_final if abs(beta - 1.0) < 1e-12 else held_grid[index]
        for position, sample in enumerate(heldout):
            grid_rows.append({'sample_id': sample.bag_id, 'fold': int(sample.fold), 'group_id': sample.group_id, 'label_id': int(sample.label_id), 'beta': float(beta), 'probability': np.asarray(probability[position], float).tolist()})
    atomic_parquet(destination / 'grid.parquet', grid_rows)
    manifest = {'status': 'PASS', 'dataset': dataset, 'arm': ARM, 'fold': int(fold), 'seed': int(seed), 'source_checkpoint': str(checkpoint), 'ncr_fill': trainer.ncr_fill, 'validation_bags': len(validation), 'heldout_bags': len(heldout), 'beta_grid': [float(beta) for beta in BETA_GRID], 'validation_scores': {str(beta): val_scores[float(beta)] for beta in BETA_GRID}, 'selected_beta': float(selected), 'match_beta0_max_abs': match_beta0, 'match_beta1_parquet_max_abs': match_beta1_parquet, 'match_beta1_apply_max_abs': match_beta1_apply, 'device': device, 'nested': True, 'baseline_only': False}
    atomic_json(manifest_path, manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest
