from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .baseline import load_baseline
from .data import SceneCache
from .dataset import DATASET_SPEC, dev_index_rows
from .io_utils import atomic_json, atomic_parquet, read_json
from .permute import load_permutation
from .training import Trainer, build_samples_for_cache, fold_split, partition

def build_baseline(cfg: Mapping[str, Any], dataset: str):
    loaded = load_baseline(dataset, cfg)
    if dataset == 'tcga_crc_msi':
        patients, tiles = loaded
        return (patients, tiles)
    return (loaded, None)

def _heldout_rows(heldout: Sequence[Any], arm: str, seed: int, *, baseline_probability: np.ndarray | None=None, final_probability: np.ndarray | None=None, labels: np.ndarray | None=None) -> list[dict[str, Any]]:
    if baseline_probability is None:
        stacked = np.stack([np.asarray(sample.baseline, np.float32) for sample in heldout])
        if stacked.ndim == 1:
            stacked = stacked[:, None]
        baseline_probability = final_probability = stacked
        labels = np.asarray([sample.label_id for sample in heldout], np.int64)
    rows = []
    for position, sample in enumerate(heldout):
        if int(labels[position]) != int(sample.label_id):
            raise RuntimeError(f'prediction identity mismatch for {sample.bag_id}: predict label {int(labels[position])} != sample label {int(sample.label_id)}')
        rows.append({'sample_id': sample.bag_id, 'tiles': len(sample.graph_ids), 'graph_ids': list(sample.graph_ids), 'fold': int(sample.fold), 'group_id': sample.group_id, 'label_id': int(sample.label_id), 'baseline_probability': np.asarray(baseline_probability[position], float).tolist(), 'final_probability': np.asarray(final_probability[position], float).tolist(), 'arm': arm, 'seed': int(seed)})
    return rows

def train_fold(cfg: Mapping[str, Any], dataset: str, arm: str, fold: int, seed: int, device: str, *, cache: SceneCache | None=None, baseline=None, width: int=64, layers: int=2, dropout: float=0.1, learning_rate: float=0.001, weight_decay: float=0.0001, max_epochs: int=60, patience: int=10, delta_l2: float=0.001, bags_per_batch: int=8, tiles_per_batch: int=256, train_tiles: int=32, max_steps: int | None=None) -> dict[str, Any]:
    import torch
    result_root = Path(cfg['paths']['result_root'])
    destination = result_root / dataset / 'predictions' / arm / f'seed_{seed}' / f'fold_{fold:02d}'
    manifest_path = destination / 'job.json'
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous.get('status') == 'PASS':
            return {'status': 'REUSED', **previous}
    if cache is None:
        cache = SceneCache(cfg, dataset)
        if arm != 'B':
            cache.preload()
    rows = dev_index_rows(dataset, cfg)
    if baseline is None:
        baseline = build_baseline(cfg, dataset)
    baseline, tile_baseline = baseline
    samples = build_samples_for_cache(dataset, rows, baseline, available=set(cache.entries), tile_baseline=tile_baseline)
    if not samples:
        raise RuntimeError(f'no development samples for {dataset}')
    covered = {graph_id for sample in samples for graph_id in sample.graph_ids}
    if not covered <= set(cache.entries):
        raise RuntimeError(f'{dataset}: samples reference graphs that are not in the scene cache')
    training_rows = [row for row in rows if row['fold'] is not None and row['split'] == 'train']
    if dataset == 'bracs':
        expected = {row['graph_id'] for row in training_rows if str(row['roi_id']) in baseline}
    elif dataset == 'tcga_crc_msi':
        expected = {row['graph_id'] for row in training_rows if str(row['patient_id']) in baseline}
    else:
        expected = {row['graph_id'] for row in training_rows if row['graph_id'] in baseline}
    missing = expected - covered
    if missing:
        raise RuntimeError(f'{dataset}: {len(missing)} baseline-covered graphs have no sample, e.g. {sorted(missing)[:3]}')
    training_all, heldout = fold_split(samples, fold)
    if arm == 'B':
        atomic_parquet(destination / 'predictions.parquet', _heldout_rows(heldout, arm, seed))
        manifest = {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'fold': int(fold), 'seed': int(seed), 'samples': len(samples), 'train_bags': len(training_all), 'validation_bags': 0, 'heldout_bags': len(heldout), 'parameters': 0, 'best_metric': None, 'epochs': 0, 'ncr_fill': None, 'history': [], 'train_graphs': 0, 'device': device, 'baseline_only': True}
        atomic_json(manifest_path, manifest)
        return manifest
    train, validation = partition(training_all, seed=seed)
    z_offset = load_permutation(cfg, dataset) if arm == 'G-Zperm' else None
    if arm == 'G-Zperm' and (not z_offset):
        raise RuntimeError('G-Zperm requires the zperm map; run `permute` first')
    trainer = Trainer(dataset=dataset, cache=cache, arm=arm, seed=seed, device=device, width=width, layers=layers, dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay, max_epochs=max_epochs, patience=patience, delta_l2=delta_l2, bags_per_batch=bags_per_batch, tiles_per_batch=tiles_per_batch, train_tiles=train_tiles, z_offset=z_offset)
    fitted = trainer.fit(train, validation, destination=destination, max_steps=max_steps)
    baseline_probability, final_probability, labels = trainer.predict(heldout)
    atomic_parquet(destination / 'predictions.parquet', _heldout_rows(heldout, arm, seed, baseline_probability=baseline_probability, final_probability=final_probability, labels=labels))
    graph_ids = sorted({graph_id for sample in train for graph_id in sample.graph_ids})
    manifest = {'status': 'PASS', 'dataset': dataset, 'arm': arm, 'fold': int(fold), 'seed': int(seed), 'samples': len(samples), 'train_bags': len(train), 'validation_bags': len(validation), 'heldout_bags': len(heldout), 'parameters': trainer.parameter_count(), 'best_metric': fitted['best_metric'], 'epochs': fitted['epochs'], 'ncr_fill': fitted['ncr_fill'], 'history': fitted['history'], 'train_graphs': len(graph_ids), 'device': device}
    atomic_json(manifest_path, manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest

def job_matrix(cfg: Mapping[str, Any], dataset: str, arms: Sequence[str], seed: int) -> list[dict]:
    folds = int(DATASET_SPEC[dataset]['folds'])
    return [{'dataset': dataset, 'arm': arm, 'fold': fold, 'seed': seed} for arm in arms for fold in range(folds)]
