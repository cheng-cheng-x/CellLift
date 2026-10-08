from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
import numpy as np
from .dataset import DATASET_SPEC, dev_index_rows
from .io_utils import read_parquet
BASELINE_SEEDS = (17, 42, 73, 101, 137)
MINIMUM_PATIENT_TILES = 10
BRACS_HELDOUT = 'heldout'

def _seed_bundle(values, seeds, key):
    found = sorted({int(row['seed']) for row in values})
    if found != sorted(seeds) or len(values) != len(seeds):
        raise RuntimeError(f'seed bundle mismatch for {key}: {found} != {list(seeds)}')

def crc_hard_vote(tile_positive) -> float:
    values = np.asarray(list(tile_positive), np.float64).reshape(-1)
    if not len(values):
        raise RuntimeError('empty CRC tile list')
    positive = int(np.sum(values >= 0.5))
    total = int(len(values))
    continuity = float(np.log((positive + 0.5) / (total - positive + 0.5)))
    return float(1.0 / (1.0 + np.exp(-continuity)))

def crc_mean_then_vote(seed_tile_positive: np.ndarray) -> float:
    tiles = np.asarray(seed_tile_positive, np.float64)
    if tiles.ndim == 1:
        tiles = tiles[None, :]
    return crc_hard_vote(tiles.mean(axis=0))

def load_sicapv2_baseline(cfg) -> dict[str, dict]:
    root = Path(cfg['paths']['baseline_predictions'])
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for fold in range(4):
        for seed in BASELINE_SEEDS:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in read_parquet(path):
                grouped[fold, str(row['graph_id'])].append(row)
    rows = dev_index_rows('sicapv2', cfg)
    labels = {row['graph_id']: int(row['label_id']) for row in rows}
    groups = {row['graph_id']: str(row['patient_id']) for row in rows}
    output = {}
    for (fold, graph_id), values in grouped.items():
        _seed_bundle(values, BASELINE_SEEDS, (fold, graph_id))
        if graph_id not in labels:
            continue
        probability = np.mean([np.asarray(row['probabilities'], float) for row in values], 0)
        output[graph_id] = {'graph_id': graph_id, 'fold': int(fold), 'label_id': labels[graph_id], 'group_id': groups[graph_id], 'rgb_probability': probability}
    return output

def load_bracs_baseline(cfg) -> dict[str, dict]:
    root = Path(cfg['paths']['baseline_predictions'])
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for fold in range(5):
        for seed in BASELINE_SEEDS:
            path = root / f'experts/development/t7/meanpool/E0_RGB_MASK2D/fold_{fold:02d}/seed_{seed}/predictions.parquet'
            for row in read_parquet(path):
                if row['role'] != BRACS_HELDOUT:
                    continue
                grouped[fold, str(row['roi_id'])].append(row)
    rows = dev_index_rows('bracs', cfg)
    by_roi: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_roi[str(row['roi_id'])].append(row)
    output = {}
    for (fold, roi_id), values in grouped.items():
        _seed_bundle(values, BASELINE_SEEDS, (fold, roi_id))
        members = by_roi.get(roi_id)
        if not members:
            continue
        labels = {int(member['label_id']) for member in members}
        if len(labels) != 1:
            raise RuntimeError(f'BRACS ROI label mismatch: {roi_id}')
        probability = np.mean([np.asarray(row['probability'], float) for row in values], 0)
        output[roi_id] = {'graph_id': roi_id, 'fold': int(fold), 'label_id': labels.pop(), 'group_id': str(values[0]['wsi_id']), 'rgb_probability': probability}
    return output

def load_crc_baseline(cfg) -> dict[str, dict]:
    root = Path(cfg['paths']['baseline_predictions'])
    tiles: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    identity: dict[tuple[int, str], dict] = {}
    for fold in range(5):
        for seed in BASELINE_SEEDS:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in read_parquet(path):
                key = (fold, str(row['patient_id']), str(row['graph_id']))
                tiles[key].append(float(np.asarray(row['probabilities'], float)[1]))
                identity.setdefault((fold, str(row['patient_id'])), row)
    grouped: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    labels: dict[tuple[int, str], set[int]] = defaultdict(set)
    patients = {}
    for (fold, patient, _graph), values in tiles.items():
        if len(values) != len(BASELINE_SEEDS):
            raise RuntimeError(f'CRC seed bundle mismatch for {(fold, patient)}')
        grouped[fold, patient].append(values)
        labels[fold, patient].add(int(identity[fold, patient]['label_id']))
    for (fold, patient), seed_lists in grouped.items():
        if len(seed_lists) < MINIMUM_PATIENT_TILES:
            continue
        if len(labels[fold, patient]) != 1:
            raise RuntimeError(f'CRC patient label mismatch: {(fold, patient)}')
        matrix = np.asarray(seed_lists, np.float64).T
        patients[patient] = {'patient_id': patient, 'fold': int(fold), 'label_id': next(iter(labels[fold, patient])), 'group_id': patient, 'rgb_probability': np.asarray([crc_mean_then_vote(matrix)], np.float32)}
    return patients

def load_baseline(dataset: str, cfg):
    if dataset == 'sicapv2':
        return load_sicapv2_baseline(cfg)
    if dataset == 'bracs':
        return load_bracs_baseline(cfg)
    if dataset == 'tcga_crc_msi':
        return load_crc_baseline(cfg)
    raise ValueError(dataset)

def baseline_probability(dataset: str, row: dict) -> np.ndarray:
    return np.asarray(row['rgb_probability'], np.float32)

def classes_for(dataset: str) -> int:
    return int(DATASET_SPEC[dataset]['classes'])
