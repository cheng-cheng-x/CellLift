from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from celllift.runtime import ResourcePath as Path
from typing import Any
from .utils import atomic_json, atomic_parquet, normalize_patch_id, one_hot_label, prepare_output_tree, require_upstream_pass, sha256_file, shard_hex, stable_shard, xlsx_dicts
SICAP_LABELS = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3}
CRC_LABELS = {'nonMSIH': 0, 'MSIH': 1}

def _key(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    folded = {str(name).strip().lower(): name for name in row}
    for candidate in candidates:
        if candidate.lower() in folded:
            return folded[candidate.lower()]
    raise KeyError(f'none of {candidates} found in columns {list(row)}')

def _hash_sources(rows: list[dict[str, Any]], workers: int) -> None:
    paths = [Path(row['source_path']) for row in rows]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digests = list(pool.map(sha256_file, paths, chunksize=16))
    for row, digest in zip(rows, digests):
        row['source_sha256'] = digest

def _output_paths(data_root: Path, row: dict[str, Any], width: int, height: int, mpp: float) -> None:
    patch_id = str(row['patch_id'])
    subdir = shard_hex(patch_id)
    segment = data_root / '02_cellpose_dual'
    row.update({'target_mpp': float(mpp), 'actual_target_mpp': float(mpp), 'output_width_px': int(width), 'output_height_px': int(height), 'physical_width_um': float(width * mpp), 'physical_height_um': float(height * mpp), 'source_to_target_affine': json.dumps([width / 512.0, 0.0, 0.0, 0.0, height / 512.0, 0.0, 0.0, 0.0, 1.0], separators=(',', ':')), 'rgb_path': str(data_root / '01_standardized_rgb' / subdir / f'{patch_id}.png'), 'nucleus_mask_path': str(segment / 'nucleus_masks' / subdir / f'{patch_id}.npy'), 'cell_mask_path': str(segment / 'cell_masks' / subdir / f'{patch_id}.npy'), 'nucleus_instances_path': str(segment / 'nucleus_instances' / subdir / f'{patch_id}.json.gz'), 'cell_instances_path': str(segment / 'cell_instances' / subdir / f'{patch_id}.json.gz'), 'pairs_path': str(segment / 'pairs' / subdir / f'{patch_id}.json.gz'), 'graph_id': f"{row['dataset_id']}:{patch_id}", 'graph_shard': stable_shard(f"{row['dataset_id']}:{patch_id}", 64), 'node_count': None, 'edge_count': None, 'processing_status': 'inventoried', 'exclusion_reason': ''})

def _even_take(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: hashlib.blake2b(f"{salt}:{row['patch_id']}".encode(), digest_size=16).digest())
    if len(ordered) < count:
        raise RuntimeError(f'pilot stratum has only {len(ordered)} rows; expected {count}')
    return ordered[:count]

def _image_features(path: str | Path) -> list[float]:
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB').resize((64, 64)), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / maximum, 0)
    brightness = rgb.mean(axis=2)
    tissue = np.mean((brightness < 0.9) & (saturation > 0.05))
    return [float(brightness.mean()), float(brightness.std()), float(saturation.mean()), float(tissue)]

def _diverse_take(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    import numpy as np
    candidates = _even_take(rows, min(256, len(rows)), salt + ':pool')
    features = np.asarray([_image_features(row['source_path']) for row in candidates], np.float64)
    scale = np.maximum(features.std(axis=0), 1e-06)
    features = (features - features.mean(axis=0)) / scale
    seed = int.from_bytes(hashlib.blake2b(salt.encode(), digest_size=8).digest(), 'big') % len(candidates)
    chosen = [seed]
    distance = np.square(features - features[seed]).sum(axis=1)
    while len(chosen) < count:
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, np.square(features - features[index]).sum(axis=1))
        distance[chosen] = -1
    return [candidates[index] for index in chosen]

def build_sicap_inventory(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_root = Path(cfg['paths']['raw_root'])
    data_root = prepare_output_tree(cfg['paths']['data_root'])
    require_upstream_pass(raw_root)
    source = raw_root / 'extracted/SICAPv2/SICAPv2'
    image_dir, mask_dir = (source / 'images', source / 'masks')
    partition = source / 'partition'
    outer_rows: dict[str, dict[str, Any]] = {}
    for split, workbook in (('train', partition / 'Test/Train.xlsx'), ('test', partition / 'Test/Test.xlsx')):
        for item in xlsx_dicts(workbook):
            image_key = _key(item, ('image_name', 'image', 'patch_name'))
            patch_id = normalize_patch_id(item[image_key])
            label_name = one_hot_label(item, ('NC', 'G3', 'G4', 'G5'))
            if patch_id in outer_rows:
                raise RuntimeError(f'duplicate official SICAP patch: {patch_id}')
            outer_rows[patch_id] = {'official_split': split, 'label_name': label_name}
    cribriform: dict[str, int] = {}
    for workbook in (partition / 'Test/TrainCribfriform.xlsx', partition / 'Test/TestCribfriform.xlsx'):
        for item in xlsx_dicts(workbook):
            image_key = _key(item, ('image_name', 'image', 'patch_name'))
            patch_id = normalize_patch_id(item[image_key])
            numeric = [int(value) for key, value in item.items() if key != image_key and value in (0, 1)]
            if not numeric:
                raise RuntimeError(f'cribriform label missing for {patch_id}')
            cribriform[patch_id] = numeric[-1]
    validation_fold: dict[str, int] = {}
    for fold in range(1, 5):
        workbook = partition / f'Validation/Val{fold}/Test.xlsx'
        for item in xlsx_dicts(workbook):
            patch_id = normalize_patch_id(item[_key(item, ('image_name', 'image', 'patch_name'))])
            if patch_id in validation_fold:
                raise RuntimeError(f'SICAP patch occurs in multiple validation folds: {patch_id}')
            validation_fold[patch_id] = fold - 1
    wsi_map: dict[str, str] = {}
    for item in xlsx_dicts(source / 'wsi_labels.xlsx'):
        slide_key = _key(item, ('slide_id', 'wsi_id', 'image_id'))
        patient_key = _key(item, ('patient_id', 'patient'))
        wsi_map[str(item[slide_key]).strip()] = str(item[patient_key]).strip()
    rows: list[dict[str, Any]] = []
    for patch_id, labels in sorted(outer_rows.items()):
        image_path = image_dir / f'{patch_id}.jpg'
        mask_path = mask_dir / f'{patch_id}.jpg'
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f'missing SICAP image/mask pair: {patch_id}')
        slide_id = patch_id.split('_Block_', 1)[0]
        if slide_id not in wsi_map:
            raise RuntimeError(f'SICAP slide lacks patient mapping: {slide_id}')
        label_name = labels['label_name']
        row = {'dataset_id': 'sicapv2_set_encoding', 'patch_id': patch_id, 'source_path': str(image_path), 'source_mask_path': str(mask_path), 'source_mpp': 0.92, 'mpp_provenance': 'inferred: Ventana iScan Coreo 40x 0.23 um/px divided by four', 'patient_id': wsi_map[slide_id], 'slide_id': slide_id, 'official_split': labels['official_split'], 'validation_fold': validation_fold.get(patch_id), 'label_id': SICAP_LABELS[label_name], 'label_name': label_name, 'label_scope': 'patch_ground_truth', 'g4c_label': cribriform.get(patch_id) if label_name == 'G4' else None, 'g4c_valid': label_name == 'G4' and patch_id in cribriform, 'patient_weight': None, 'class_patient_weight': None}
        _output_paths(data_root, row, 1024, 1024, 0.46)
        rows.append(row)
    train_ids = {row['patch_id'] for row in rows if row['official_split'] == 'train'}
    if set(validation_fold) != train_ids:
        raise RuntimeError('official SICAP validation folds do not partition the outer training set')
    _hash_sources(rows, int(cfg['runtime'].get('hash_workers', 16)))
    all_images = {path.stem: path for path in image_dir.glob('*.jpg')}
    unlabeled = [{'dataset_id': 'sicapv2_set_encoding', 'patch_id': patch_id, 'source_path': str(path), 'reason': 'not_in_official_patch_partition'} for patch_id, path in sorted(all_images.items()) if patch_id not in outer_rows]
    if len(rows) != 12081 or len(unlabeled) != 6702:
        raise RuntimeError(f'unexpected SICAP supervised/unlabeled counts: {len(rows)}/{len(unlabeled)}')
    pilots: list[dict[str, Any]] = []
    for label in ('NC', 'G3', 'G5'):
        choices = [row for row in rows if row['label_name'] == label]
        pilots.extend(_even_take([r for r in choices if r['official_split'] == 'train'], 12, f'sicap:{label}:train'))
        pilots.extend(_even_take([r for r in choices if r['official_split'] == 'test'], 4, f'sicap:{label}:test'))
    g4 = [row for row in rows if row['label_name'] == 'G4']
    pilots.extend(_even_take([r for r in g4 if r['g4c_label'] == 1], 8, 'sicap:G4:G4C1'))
    pilots.extend(_even_take([r for r in g4 if r['g4c_label'] == 0], 8, 'sicap:G4:G4C0'))
    atomic_parquet(data_root / '00_manifest/patch_manifest.parquet', rows)
    atomic_parquet(data_root / '00_manifest/unlabeled_manifest.parquet', unlabeled)
    atomic_parquet(data_root / '00_manifest/pilot_manifest.parquet', pilots)
    atomic_parquet(data_root / '04_labels_splits/labels_splits.parquet', rows)
    summary = {'status': 'PASS', 'dataset': 'sicapv2_set_encoding', 'supervised_patches': len(rows), 'unlabeled_patches': len(unlabeled), 'patients': len({row['patient_id'] for row in rows}), 'slides': len({row['slide_id'] for row in rows}), 'split_counts': Counter((row['official_split'] for row in rows)), 'label_counts': Counter((row['label_name'] for row in rows)), 'pilot_count': len(pilots)}
    atomic_json(data_root / '00_manifest/inventory.status.json', summary)
    return summary
_TCGA_PATIENT = re.compile('^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')

def _crc_fold_assignment(rows: list[dict[str, Any]], seed: int=20260811) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['official_split'] == 'train':
            grouped[row['patient_id']].append(row)
    folds = [defaultdict(int) for _ in range(5)]
    assignments: dict[str, int] = {}
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for patient, items in grouped.items():
        labels = {str(item['label_name']) for item in items}
        if len(labels) != 1:
            raise RuntimeError(f'CRC patient has inconsistent labels: {patient}')
        by_label[next(iter(labels))].append((patient, len(items)))
    for label, patients in sorted(by_label.items()):
        patients.sort(key=lambda item: (-item[1], hashlib.blake2b(f'{seed}:{item[0]}'.encode(), digest_size=8).digest()))
        for patient, tiles in patients:
            fold = min(range(5), key=lambda idx: (folds[idx][f'tiles:{label}'], folds[idx][f'patients:{label}'], folds[idx]['tiles:all'], folds[idx]['patients:all'], idx))
            assignments[patient] = fold
            folds[fold][f'tiles:{label}'] += tiles
            folds[fold][f'patients:{label}'] += 1
            folds[fold]['tiles:all'] += tiles
            folds[fold]['patients:all'] += 1
    return assignments

def build_crc_inventory(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_root = Path(cfg['paths']['raw_root'])
    data_root = prepare_output_tree(cfg['paths']['data_root'])
    require_upstream_pass(raw_root)
    extracted = raw_root / 'extracted'
    rows: list[dict[str, Any]] = []
    for split, split_dir in (('train', extracted / 'TRAIN/TRAIN'), ('test', extracted / 'TEST/TEST')):
        for label_dir in sorted((path for path in split_dir.iterdir() if path.is_dir())):
            canonical = 'MSIH' if label_dir.name.lower() == 'msih' else 'nonMSIH'
            for source_path in sorted(label_dir.glob('*.jpg')):
                patch_id = source_path.stem
                match = _TCGA_PATIENT.match(patch_id)
                if match is None:
                    raise RuntimeError(f'cannot parse TCGA patient from {patch_id}')
                patient_id = match.group(1)
                slide_id = patch_id.split('.', 1)[0]
                row = {'dataset_id': 'tcga_crc_dx_msi_2020', 'patch_id': patch_id, 'source_path': str(source_path), 'source_mask_path': '', 'source_mpp': 0.5, 'mpp_provenance': 'Zenodo 3832231: 256 um field stored as 512 pixels', 'patient_id': patient_id, 'slide_id': slide_id, 'official_split': split, 'validation_fold': None, 'label_id': CRC_LABELS[canonical], 'label_name': canonical, 'label_scope': 'patient_inherited', 'g4c_label': None, 'g4c_valid': False, 'patient_weight': None, 'class_patient_weight': None}
                actual_mpp = 256.0 / 557.0
                _output_paths(data_root, row, 557, 557, actual_mpp)
                rows.append(row)
    if len(rows) != 51918:
        raise RuntimeError(f'unexpected CRC patch count: {len(rows)}')
    assignments = _crc_fold_assignment(rows, int(cfg['split'].get('seed', 20260811)))
    patient_tiles = Counter((row['patient_id'] for row in rows if row['official_split'] == 'train'))
    class_patients: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row['official_split'] == 'train':
            class_patients[row['label_name']].add(row['patient_id'])
            row['validation_fold'] = assignments[row['patient_id']]
            row['patient_weight'] = 1.0 / patient_tiles[row['patient_id']]
            row['class_patient_weight'] = 1.0 / (len(class_patients[row['label_name']]) * patient_tiles[row['patient_id']])
    for row in rows:
        if row['official_split'] == 'train':
            row['class_patient_weight'] = 1.0 / (len(class_patients[row['label_name']]) * patient_tiles[row['patient_id']])
    train = [row for row in rows if row['official_split'] == 'train']
    for key in ('patient_weight', 'class_patient_weight'):
        mean = sum((float(row[key]) for row in train)) / len(train)
        for row in train:
            row[key] = float(row[key]) / mean
    _hash_sources(rows, int(cfg['runtime'].get('hash_workers', 16)))
    train_patients = {row['patient_id'] for row in rows if row['official_split'] == 'train'}
    test_patients = {row['patient_id'] for row in rows if row['official_split'] == 'test'}
    if train_patients & test_patients:
        raise RuntimeError('CRC official TRAIN/TEST patient leakage detected')
    train_hashes = {row['source_sha256'] for row in rows if row['official_split'] == 'train'}
    test_hashes = {row['source_sha256'] for row in rows if row['official_split'] == 'test'}
    if train_hashes & test_hashes:
        raise RuntimeError('CRC official TRAIN/TEST content hash overlap detected')
    pilots: list[dict[str, Any]] = []
    for split in ('train', 'test'):
        for label in ('MSIH', 'nonMSIH'):
            choices = [row for row in rows if row['official_split'] == split and row['label_name'] == label]
            pilots.extend(_diverse_take(choices, 16, f'crc:{split}:{label}'))
    atomic_parquet(data_root / '00_manifest/patch_manifest.parquet', rows)
    atomic_parquet(data_root / '00_manifest/pilot_manifest.parquet', pilots)
    atomic_parquet(data_root / '04_labels_splits/labels_splits.parquet', rows)
    summary = {'status': 'PASS', 'dataset': 'tcga_crc_dx_msi_2020', 'patches': len(rows), 'patients': len({row['patient_id'] for row in rows}), 'slides': len({row['slide_id'] for row in rows}), 'split_counts': Counter((row['official_split'] for row in rows)), 'label_split_counts': Counter((f"{row['official_split']}:{row['label_name']}" for row in rows)), 'fold_tile_counts': Counter((row['validation_fold'] for row in train)), 'fold_patient_counts': Counter(assignments.values()), 'pilot_count': len(pilots)}
    atomic_json(data_root / '00_manifest/inventory.status.json', summary)
    return summary

def build_inventory(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    if dataset == 'sicapv2':
        return build_sicap_inventory(cfg)
    if dataset == 'crc_msi':
        return build_crc_inventory(cfg)
    raise ValueError(dataset)
