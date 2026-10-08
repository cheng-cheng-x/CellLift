from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
import numpy as np
from .io_utils import read_parquet
BASELINE_SEEDS = (17, 42, 73, 101, 137)
MINIMUM_PATIENT_TILES = 10
BRACS_HELDOUT = 'heldout'

def _seed_bundle(values, seeds, key):
    found = sorted({int(row['seed']) for row in values})
    if found != sorted(seeds) or len(values) != len(seeds):
        raise RuntimeError(f'seed bundle mismatch for {key}: {found} != {list(seeds)}')

def _dev_rows(dataset: str, cfg) -> list[dict]:
    from .dataset import dev_index_rows
    return dev_index_rows(dataset, cfg)

def load_sicapv2_baseline(cfg) -> dict[str, dict]:
    root = Path(cfg['paths']['baseline_predictions'])
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for fold in range(4):
        for seed in BASELINE_SEEDS:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in read_parquet(path):
                grouped[fold, str(row['graph_id'])].append(row)
    rows = _dev_rows('sicapv2', cfg)
    labels = {row['graph_id']: int(row['label_id']) for row in rows}
    groups = {row['graph_id']: str(row['patient_id']) for row in rows}
    output = {}
    for (fold, graph_id), values in grouped.items():
        _seed_bundle(values, BASELINE_SEEDS, (fold, graph_id))
        if graph_id not in labels:
            continue
        probability = np.mean([np.asarray(row['probabilities'], float) for row in values], 0)
        output[graph_id] = {'graph_id': graph_id, 'fold': int(fold), 'label_id': labels[graph_id], 'group_id': groups[graph_id], 'rgb_logits': np.log(np.clip(probability, 1e-07, 1.0)), 'rgb_probability': probability}
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
    rows = _dev_rows('bracs', cfg)
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
        output[roi_id] = {'graph_id': roi_id, 'fold': int(fold), 'label_id': labels.pop(), 'group_id': str(values[0]['wsi_id']), 'rgb_logits': np.log(np.clip(probability, 1e-07, 1.0)), 'rgb_probability': probability}
    return output

def load_crc_baseline(cfg) -> tuple[dict[str, dict], dict[str, dict]]:
    root = Path(cfg['paths']['baseline_predictions'])
    probabilities: dict[tuple[int, str], list[float]] = defaultdict(list)
    identity: dict[tuple[int, str], dict] = {}
    for fold in range(5):
        for seed in BASELINE_SEEDS:
            path = root / f'paper_baseline/fold_{fold:02d}/seed_{seed}/validation_predictions.parquet'
            for row in read_parquet(path):
                key = (fold, str(row['graph_id']))
                probabilities[key].append(float(row['probabilities'][1]))
                identity.setdefault(key, row)
    for key, values in probabilities.items():
        if len(values) != len(BASELINE_SEEDS):
            raise RuntimeError(f'CRC seed bundle mismatch for {key}: {len(values)}')
        identity[key]['_probability'] = float(np.mean(values))
    tiles: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    tile_rows: dict[str, dict] = {}
    for (fold, graph_id), row in identity.items():
        patient = str(row['patient_id'])
        tiles[fold, patient].append((float(row['_probability']), int(row['label_id'])))
        tile_rows[graph_id] = {'graph_id': graph_id, 'fold': int(fold), 'patient_id': patient, 'label_id': int(row['label_id']), 'rgb_probability': float(row['_probability'])}
    patients = {}
    for (fold, patient), values in tiles.items():
        if len(values) < MINIMUM_PATIENT_TILES:
            continue
        labels = {label for _, label in values}
        if len(labels) != 1:
            raise RuntimeError(f'CRC patient label mismatch: {(fold, patient)}')
        positive = sum((probability >= 0.5 for probability, _ in values))
        total = len(values)
        continuity = float(np.log((positive + 0.5) / (total - positive + 0.5)))
        patients[patient] = {'patient_id': patient, 'fold': int(fold), 'label_id': values[0][1], 'group_id': patient, 'rgb_logit': continuity, 'tiles': total}
    return (patients, tile_rows)

def load_baseline(dataset: str, cfg):
    if dataset == 'sicapv2':
        return load_sicapv2_baseline(cfg)
    if dataset == 'bracs':
        return load_bracs_baseline(cfg)
    if dataset == 'tcga_crc_msi':
        return load_crc_baseline(cfg)
    raise ValueError(f'unsupported dataset: {dataset}')

def baseline_probability(dataset: str, row: dict) -> np.ndarray | float:
    if dataset == 'tcga_crc_msi':
        value = float(row['rgb_logit'])
        return 1.0 / (1.0 + np.exp(-value))
    return np.asarray(row['rgb_probability'], float)
