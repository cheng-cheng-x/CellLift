from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from . import PROTOCOL_ID, SEED
from .paths import split_root
LABEL_PATHS = {'sicapv2': _resource_path('artifact_0012'), 'tcga_crc_msi': _resource_path('artifact_0013')}
BRACS_CANDIDATES = (_resource_path('artifact_0014'), _resource_path('artifact_0005'), _resource_path('artifact_0015'), _resource_path('artifact_0016'), _resource_path('artifact_0017'))

def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == '.csv':
        import csv
        with path.open('r', encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def _dump(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text + '\n', encoding='utf-8')
    return _sha(text.encode('utf-8'))

def _norm_split(value: Any) -> str:
    text = str(value or '').strip().lower()
    if text in {'train', 'training', 'fit'}:
        return 'train'
    if text in {'val', 'valid', 'validation'}:
        return 'val'
    if text in {'test', 'testing'}:
        return 'test'
    return text

def _row_id(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key) not in (None, ''):
            return str(row[key])
    raise KeyError(keys)

def build_sicap(destination: Path) -> dict[str, Any]:
    rows = _read_table(Path(LABEL_PATHS['sicapv2']))
    train = [row for row in rows if _norm_split(row.get('official_split')) == 'train']
    test = [row for row in rows if _norm_split(row.get('official_split')) == 'test']
    folds: dict[int, list[str]] = defaultdict(list)
    records = []
    for row in train:
        fold = int(row['validation_fold'])
        graph_id = str(row['graph_id'])
        folds[fold].append(graph_id)
        records.append({'sample_id': graph_id, 'graph_id': graph_id, 'patient_id': str(row.get('patient_id') or ''), 'label_id': int(row['label_id']), 'fold': fold, 'official_split': 'train'})
    test_ids = [str(row['graph_id']) for row in test]
    test_records = [{'sample_id': str(row['graph_id']), 'graph_id': str(row['graph_id']), 'patient_id': str(row.get('patient_id') or ''), 'label_id': int(row['label_id']), 'fold': None, 'official_split': 'test'} for row in test]
    units = []
    for fold in sorted(folds):
        val_ids = sorted(set(folds[fold]))
        fit_ids = sorted({graph_id for other, ids in folds.items() if other != fold for graph_id in ids})
        units.append({'name': f'fold_{fold:02d}', 'kind': 'sicap_official_fold', 'fit': fit_ids, 'val': val_ids, 'predict': val_ids, 'test': test_ids})
    units.append({'name': 'full_train', 'kind': 'sicap_full_train', 'fit': sorted({row['sample_id'] for row in records}), 'val': [], 'predict': [], 'test': test_ids})
    manifest = {'dataset': 'sicapv2', 'protocol_id': PROTOCOL_ID, 'n_train': len(records), 'n_test': len(test_ids), 'folds': {str(fold): len(set(ids)) for fold, ids in folds.items()}, 'units': units}
    hashes = {'train_rows': _dump(destination / 'train_rows.json', records), 'test_rows': _dump(destination / 'test_rows.json', test_records), 'units': _dump(destination / 'units.json', units)}
    (destination / 'summary.json').write_text(json.dumps({**manifest, 'hashes': hashes}, indent=2) + '\n', encoding='utf-8')
    return {**manifest, 'hashes': hashes, 'path': str(destination)}

def _bracs_split_key(row: dict[str, Any]) -> str:
    for key in ('official_split', 'split_new', 'split', 'set', 'subset', 'partition'):
        if row.get(key) not in (None, ''):
            return _norm_split(row[key])
    return ''

def build_bracs(destination: Path) -> dict[str, Any]:
    table = None
    source = None
    for candidate in BRACS_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            table = _read_table(path)
            source = str(path)
            break
    if not table:
        raise FileNotFoundError('BRACS official manifest not found in candidates')
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        split = _bracs_split_key(row)
        if split not in {'train', 'val', 'test'}:
            continue
        grouped[split].append(row)
    if not grouped['train'] or not grouped['val'] or (not grouped['test']):
        index = Path(_resource_path('artifact_0018'))
        if index.is_file():
            payload = json.loads(index.read_text(encoding='utf-8'))
            grouped = defaultdict(list)
            source = str(index)
            for entry in payload.get('graphs') or []:
                meta = entry.get('metadata') or {}
                split = _bracs_split_key({**meta, **entry, 'official_split': meta.get('official_split') or meta.get('split') or entry.get('split')})
                if split not in {'train', 'val', 'test'}:
                    continue
                grouped[split].append({'roi_id': meta.get('roi_id') or entry.get('graph_id'), 'graph_id': entry.get('graph_id'), 'wsi_id': meta.get('wsi_id') or meta.get('slide_id'), 'label_id': meta.get('label_id'), 'label': meta.get('label') or meta.get('label_7'), 'official_split': split})
        if not grouped['train'] or not grouped['val'] or (not grouped['test']):
            raise RuntimeError(f'BRACS missing official splits from {source}: { {k: len(v) for k, v in grouped.items()}}')

    def pack(split: str) -> list[dict[str, Any]]:
        by_roi: dict[str, dict[str, Any]] = {}
        for row in grouped[split]:
            sample_id = _row_id(row, ('roi_id', 'bag_id', 'sample_id', 'graph_id'))
            rec = by_roi.setdefault(sample_id, {'sample_id': sample_id, 'graph_id': str(row.get('graph_id') or sample_id), 'graph_ids': [], 'rgb_paths': [], 'roi_id': str(row.get('roi_id') or sample_id), 'wsi_id': str(row.get('wsi_id') or row.get('slide_id') or ''), 'label_id': None if row.get('label_id') in (None, '') and row.get('label_7') in (None, '') else int(row.get('label_id') if row.get('label_id') not in (None, '') else row.get('label_7')), 'label': row.get('label') or row.get('label_name') or row.get('label_7') or row.get('t7'), 'official_split': split})
            gid = str(row.get('graph_id') or sample_id)
            if gid not in rec['graph_ids']:
                rec['graph_ids'].append(gid)
            path = row.get('rgb_path') or row.get('link_path') or row.get('src_path')
            if path and path not in rec['rgb_paths']:
                rec['rgb_paths'].append(str(path))
        return list(by_roi.values())
    train, val, test = (pack('train'), pack('val'), pack('test'))
    units = [{'name': 'official', 'kind': 'bracs_official', 'fit': sorted({row['sample_id'] for row in train}), 'val': sorted({row['sample_id'] for row in val}), 'predict': sorted({row['sample_id'] for row in val}), 'test': sorted({row['sample_id'] for row in test})}]
    hashes = {'train_rows': _dump(destination / 'train_rows.json', train), 'val_rows': _dump(destination / 'val_rows.json', val), 'test_rows': _dump(destination / 'test_rows.json', test), 'units': _dump(destination / 'units.json', units), 'source': _sha(source.encode())}
    summary = {'dataset': 'bracs', 'protocol_id': PROTOCOL_ID, 'source': source, 'n_train': len(train), 'n_val': len(val), 'n_test': len(test), 'hashes': hashes, 'b_name': 'B_rgb', 'b_is_hact_net': False}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return {**summary, 'path': str(destination)}

def build_crc(destination: Path) -> dict[str, Any]:
    rows = _read_table(Path(LABEL_PATHS['tcga_crc_msi']))
    train = [row for row in rows if _norm_split(row.get('official_split')) == 'train']
    test = [row for row in rows if _norm_split(row.get('official_split')) == 'test']
    patients: dict[str, dict[str, Any]] = {}
    for row in train:
        pid = str(row['patient_id'])
        label = int(row['label_id'])
        bucket = patients.setdefault(pid, {'label_id': label, 'tiles': []})
        bucket['tiles'].append(str(row['graph_id']))
    eligible = {pid: info for pid, info in patients.items() if len(info['tiles']) >= 10}
    rng = np.random.default_rng(SEED)
    by_label: dict[int, list[str]] = defaultdict(list)
    for pid, info in eligible.items():
        by_label[int(info['label_id'])].append(pid)
    fit, val = ([], [])
    for label, ids in sorted(by_label.items()):
        order = np.asarray(sorted(ids), object)
        rng.shuffle(order)
        cut = max(1, int(round(0.2 * len(order))))
        if cut >= len(order):
            cut = len(order) - 1
        val.extend(order[:cut].tolist())
        fit.extend(order[cut:].tolist())
    fit_set, val_set = (set(fit), set(val))
    units = [{'name': 'official', 'kind': 'crc_seed42_patient_80_20', 'fit': sorted(fit_set), 'val': sorted(val_set), 'predict': sorted(val_set), 'test': sorted({str(row['patient_id']) for row in test}), 'fit_tiles': sorted((tile for pid in fit_set for tile in eligible[pid]['tiles'])), 'val_tiles': sorted((tile for pid in val_set for tile in eligible[pid]['tiles']))}]
    train_rows = [{'sample_id': pid, 'patient_id': pid, 'label_id': int(eligible[pid]['label_id']), 'n_tiles': len(eligible[pid]['tiles']), 'split': 'fit' if pid in fit_set else 'val'} for pid in sorted(eligible)]
    test_rows = []
    test_groups: dict[str, dict[str, Any]] = {}
    for row in test:
        pid = str(row['patient_id'])
        info = test_groups.setdefault(pid, {'label_id': int(row['label_id']), 'tiles': []})
        info['tiles'].append(str(row['graph_id']))
    for pid, info in test_groups.items():
        if len(info['tiles']) < 10:
            continue
        test_rows.append({'sample_id': pid, 'patient_id': pid, 'label_id': int(info['label_id']), 'n_tiles': len(info['tiles']), 'split': 'test'})
    hashes = {'train_rows': _dump(destination / 'train_rows.json', train_rows), 'test_rows': _dump(destination / 'test_rows.json', test_rows), 'units': _dump(destination / 'units.json', units)}
    summary = {'dataset': 'tcga_crc_msi', 'protocol_id': PROTOCOL_ID, 'seed': SEED, 'n_train_patients': len(eligible), 'n_fit': len(fit_set), 'n_val': len(val_set), 'n_test_patients': len(test_rows), 'hashes': hashes}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return {**summary, 'path': str(destination)}

def build_all() -> dict[str, Any]:
    root = split_root()
    result = {'sicapv2': build_sicap(root / 'sicapv2'), 'bracs': build_bracs(root / 'bracs'), 'tcga_crc_msi': build_crc(root / 'tcga_crc_msi')}
    (root / 'summary.json').write_text(json.dumps({key: {k: v for k, v in value.items() if k != 'units'} for key, value in result.items()}, indent=2) + '\n', encoding='utf-8')
    return result

def load_units(dataset: str) -> list[dict[str, Any]]:
    path = split_root() / dataset / 'units.json'
    return json.loads(path.read_text(encoding='utf-8'))
if __name__ == '__main__':
    print(json.dumps({key: {k: v for k, v in value.items() if k not in {'units'}} for key, value in build_all().items()}, indent=2))
