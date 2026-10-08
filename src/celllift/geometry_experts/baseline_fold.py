from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import sys
import numpy as np
from .baseline import BASELINE_SEEDS, MINIMUM_PATIENT_TILES
from .dataset import DATASET_SPEC, batch_result_root
from .io_utils import atomic_json, atomic_parquet, read_json, read_parquet
from .models import PROB_CLIP
HELDOUT_TOL = 0.005
B_TRAIN_SOURCE = 'current_outer_fold_checkpoint_5seed_mean'

def fold_score_dir(cfg: Mapping[str, Any], dataset: str) -> Path:
    return batch_result_root(cfg) / dataset / 'baseline_fold_scores'

def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    value = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(value)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)

def _crc_patient(rows: list[dict], tile_positive: np.ndarray) -> dict[str, dict]:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row, score in zip(rows, tile_positive):
        grouped[str(row['patient_id'])].append((float(score), int(row['label_id'])))
    patients = {}
    for patient, values in grouped.items():
        if len(values) < MINIMUM_PATIENT_TILES:
            continue
        labels = {label for _, label in values}
        if len(labels) != 1:
            raise RuntimeError(f'CRC patient label mismatch: {patient}')
        positive = sum((probability >= 0.5 for probability, _ in values))
        total = len(values)
        continuity = float(np.log((positive + 0.5) / (total - positive + 0.5)))
        probability = float(1.0 / (1.0 + np.exp(-continuity)))
        patients[patient] = {'sample_id': patient, 'label_id': values[0][1], 'group_id': patient, 'probability': [probability]}
    return patients

def _paper_infer(cfg: Mapping[str, Any], dataset: str, fold: int, seed: int, device: str) -> tuple[list[dict], np.ndarray, list[dict], np.ndarray]:
    import torch
    from celllift.geometry_baselines.models import PaperFSConv, build_paper_crc_resnet18
    from celllift.geometry_baselines.paper_rgb import PaperRGBSpec, _gpu_preprocess, _loader, _select_outer, read_rows
    cache_root = Path(cfg['paths']['geometry_baselines_data_root']) / '01_rgb224_cache'
    rows = read_rows(cache_root / 'index.parquet')
    train_rows, held_rows = _select_outer(rows, fold)
    checkpoint = Path(cfg['paths']['baseline_predictions']) / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}' / 'best.pt'
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if dataset == 'sicapv2':
        model = PaperFSConv().to(device)
    else:
        model, _ = build_paper_crc_resnet18(imagenet_weights=False)
        model = model.to(device)
    model.load_state_dict(saved['model_state_dict'])
    spec = PaperRGBSpec.for_dataset(dataset, workers=4)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)

    def forward(subset):
        loader = _loader(subset, cache_root, spec, shuffle=False, seed=seed, include_labels=True)
        logits = []
        with torch.no_grad():
            for batch in loader:
                images = _gpu_preprocess(batch['image'], dataset, train=False, generator=generator)
                output = model(images)
                value = output['logits'] if isinstance(output, dict) else output
                logits.append(value.float().cpu().numpy())
        return np.concatenate(logits, 0)
    return (train_rows, forward(train_rows), held_rows, forward(held_rows))

def _as_probability(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    if value.ndim == 1:
        return value.astype(np.float32)
    totals = value.sum(axis=-1)
    if np.all((value >= -1e-06) & (value <= 1.0 + 1e-06)) and np.allclose(totals, 1.0, atol=0.001):
        return value.astype(np.float32)
    return _softmax(value)

def _check_heldout_logits(dataset: str, fold: int, seed: int, cfg: Mapping[str, Any], held_rows: list[dict], logits: np.ndarray) -> float:
    expected_path = Path(cfg['paths']['baseline_predictions']) / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}' / 'validation_predictions.parquet'
    published = {str(row['graph_id']): row for row in read_parquet(expected_path)}
    observed = []
    target = []
    for row, value in zip(held_rows, logits):
        key = str(row['graph_id'])
        if key not in published:
            raise RuntimeError(f'heldout identity missing in published OOF: {key}')
        item = published[key]
        expected = item['probabilities'] if 'probabilities' in item else item['logits']
        observed.append(value)
        target.append(np.asarray(expected, np.float32))
    observed_p = _as_probability(np.stack(observed))
    target_p = _as_probability(np.stack(target))
    delta = float(np.max(np.abs(observed_p - target_p)))
    if dataset == 'tcga_crc_msi':
        ours = _crc_patient(held_rows, observed_p[:, 1] if observed_p.ndim == 2 else observed_p.reshape(-1))
        theirs = _crc_patient(held_rows, target_p[:, 1] if target_p.ndim == 2 else target_p.reshape(-1))
        if set(ours) != set(theirs):
            raise RuntimeError(f'{dataset} fold {fold} seed {seed} patient set mismatch')
        patient_delta = float(max((abs(ours[key]['probability'][0] - theirs[key]['probability'][0]) for key in ours)))
        if delta > 0.02:
            raise RuntimeError(f'{dataset} fold {fold} seed {seed} heldout mismatch tile_abs={delta} patient_abs={patient_delta}')
        return patient_delta
    if observed_p.ndim == 1 or observed_p.shape[-1] == 1:
        mismatch = int(np.sum((observed_p.reshape(-1) >= 0.5) != (target_p.reshape(-1) >= 0.5)))
    else:
        mismatch = int(np.sum(observed_p.argmax(1) != target_p.argmax(1)))
    if delta > HELDOUT_TOL or mismatch:
        raise RuntimeError(f'{dataset} fold {fold} seed {seed} heldout mismatch max_abs={delta} decisions={mismatch}')
    return delta

def infer_paper_fold(cfg: Mapping[str, Any], dataset: str, fold: int, device: str) -> dict[str, Any]:
    train_prob: dict[str, list[np.ndarray]] = defaultdict(list)
    held_prob: dict[str, list[np.ndarray]] = defaultdict(list)
    identity: dict[str, dict] = {}
    deltas = []
    for seed in BASELINE_SEEDS:
        train_rows, train_logits, held_rows, held_logits = _paper_infer(cfg, dataset, fold, seed, device)
        deltas.append(_check_heldout_logits(dataset, fold, seed, cfg, held_rows, held_logits))
        if dataset == 'sicapv2':
            for row, logits in zip(train_rows, train_logits):
                key = str(row['graph_id'])
                train_prob[key].append(_softmax(logits[None])[0])
                identity.setdefault(key, {'sample_id': key, 'label_id': int(row['label_id']), 'group_id': str(row['patient_id']), 'in_sample': True})
            for row, logits in zip(held_rows, held_logits):
                key = str(row['graph_id'])
                held_prob[key].append(_softmax(logits[None])[0])
                identity.setdefault(key, {'sample_id': key, 'label_id': int(row['label_id']), 'group_id': str(row['patient_id']), 'in_sample': False})
        else:
            train_patients = _crc_patient(train_rows, _softmax(train_logits)[:, 1])
            held_patients = _crc_patient(held_rows, _softmax(held_logits)[:, 1])
            for key, row in train_patients.items():
                train_prob[key].append(np.asarray(row['probability'], np.float32))
                identity.setdefault(key, {**row, 'in_sample': True})
            for key, row in held_patients.items():
                held_prob[key].append(np.asarray(row['probability'], np.float32))
                identity.setdefault(key, {**row, 'in_sample': False})
    return _write_fold(cfg, dataset, fold, identity, train_prob, held_prob, deltas)

def _bracs_import():
    root = Path(__file__).resolve().parents[1] / 'breast_roi_baseline'
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from celllift.breast_roi_baseline.training import BRACSFeatureStore, BRACSROIDataset, TileBudgetBatchSampler, collate_rois, _evaluate
    from celllift.breast_roi_baseline.rgb import BRACSRGBMaskROIModel
    return (BRACSFeatureStore, BRACSROIDataset, TileBudgetBatchSampler, collate_rois, _evaluate, BRACSRGBMaskROIModel)

def infer_bracs_fold(cfg: Mapping[str, Any], fold: int, device: str) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    BRACSFeatureStore, BRACSROIDataset, TileBudgetBatchSampler, collate_rois, _evaluate, BRACSRGBMaskROIModel = _bracs_import()
    data_root = Path(cfg['paths']['bracs_data_root'])
    result_root = Path(cfg['paths']['baseline_predictions'])
    train_prob: dict[str, list[np.ndarray]] = defaultdict(list)
    held_prob: dict[str, list[np.ndarray]] = defaultdict(list)
    identity: dict[str, dict] = {}
    deltas = []
    for seed in BASELINE_SEEDS:
        store = BRACSFeatureStore(anchor_manifest=data_root / '04_direct3d' / 'anchor_cache_manifest.json', tile_manifest=data_root / '00_manifest' / 'tile_manifest.parquet', expert='E0_RGB_MASK2D', phase='development', fold=fold, seed=seed)
        train_data = BRACSROIDataset(store, 'train', 't7')
        held_data = BRACSROIDataset(store, 'heldout', 't7')
        train_loader = DataLoader(train_data, batch_sampler=TileBudgetBatchSampler(train_data, 256, 32000, False, seed), collate_fn=collate_rois, num_workers=2, pin_memory=True)
        held_loader = DataLoader(held_data, batch_sampler=TileBudgetBatchSampler(held_data, 256, 32000, False, seed), collate_fn=collate_rois, num_workers=2, pin_memory=True)
        checkpoint = result_root / 'experts/development/t7/meanpool/E0_RGB_MASK2D' / f'fold_{fold:02d}' / f'seed_{seed}' / 'best.pt'
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = BRACSRGBMaskROIModel(set_encoder='meanpool', num_classes=7).to(device)
        model.load_state_dict(payload['state_dict'])
        _, train_rows = _evaluate(model, train_loader, torch.device(device))
        held_metric, held_rows = _evaluate(model, held_loader, torch.device(device))
        published = {str(row['roi_id']): np.asarray(row['probability'], np.float32) for row in read_parquet(checkpoint.with_name('predictions.parquet')) if row['role'] == 'heldout'}
        observed = np.stack([np.asarray(row['probability'], np.float32) for row in held_rows])
        target = np.stack([published[str(row['roi_id'])] for row in held_rows])
        delta = float(np.max(np.abs(observed - target)))
        mismatch = int(np.sum(observed.argmax(1) != target.argmax(1)))
        if delta > 0.02:
            raise RuntimeError(f'bracs fold {fold} seed {seed} heldout mismatch max_abs={delta} decisions={mismatch} metric={held_metric}')
        deltas.append(delta)
        for row in train_rows:
            key = str(row['roi_id'])
            train_prob[key].append(np.asarray(row['probability'], np.float32))
            identity.setdefault(key, {'sample_id': key, 'label_id': int(row['label']), 'group_id': str(row['wsi_id']), 'in_sample': True})
        for row in held_rows:
            key = str(row['roi_id'])
            held_prob[key].append(np.asarray(row['probability'], np.float32))
            identity.setdefault(key, {'sample_id': key, 'label_id': int(row['label']), 'group_id': str(row['wsi_id']), 'in_sample': False})
    return _write_fold(cfg, 'bracs', fold, identity, train_prob, held_prob, deltas)

def _write_fold(cfg: Mapping[str, Any], dataset: str, fold: int, identity: Mapping[str, dict], train_prob: Mapping[str, list[np.ndarray]], held_prob: Mapping[str, list[np.ndarray]], deltas: list[float]) -> dict[str, Any]:
    rows = []
    for key, bundles in list(train_prob.items()) + list(held_prob.items()):
        in_sample = key in train_prob
        probability = np.mean(np.stack(bundles, 0), 0)
        meta = identity[key]
        rows.append({'sample_id': key, 'fold': int(fold), 'label_id': int(meta['label_id']), 'group_id': str(meta['group_id']), 'probability': np.asarray(probability, float).tolist(), 'in_sample': bool(in_sample)})
    rows.sort(key=lambda row: (not row['in_sample'], row['sample_id']))
    destination = fold_score_dir(cfg, dataset)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_parquet(destination / f'fold_{fold:02d}.parquet', rows)
    manifest = {'status': 'PASS', 'dataset': dataset, 'fold': int(fold), 'seeds': list(BASELINE_SEEDS), 'n_train': int(sum((row['in_sample'] for row in rows))), 'n_heldout': int(sum((not row['in_sample'] for row in rows))), 'heldout_max_abs': float(max(deltas) if deltas else 0.0), 'b_train_source': B_TRAIN_SOURCE, 'official_test_touched': False}
    atomic_json(destination / f'fold_{fold:02d}.json', manifest)
    return manifest

def infer_dataset(cfg: Mapping[str, Any], dataset: str, device: str='cuda', fold: int | None=None) -> dict[str, Any]:
    folds = [int(fold)] if fold is not None else list(range(int(DATASET_SPEC[dataset]['folds'])))
    results = []
    for current in folds:
        marker = fold_score_dir(cfg, dataset) / f'fold_{current:02d}.json'
        if marker.is_file() and read_json(marker).get('status') == 'PASS':
            results.append({'status': 'REUSED', **read_json(marker)})
            continue
        if dataset == 'bracs':
            results.append(infer_bracs_fold(cfg, current, device))
        else:
            results.append(infer_paper_fold(cfg, dataset, current, device))
    failed = [row for row in results if row.get('status') not in {'PASS', 'REUSED'}]
    return {'status': 'PASS' if not failed else 'FAIL', 'dataset': dataset, 'folds': results}
