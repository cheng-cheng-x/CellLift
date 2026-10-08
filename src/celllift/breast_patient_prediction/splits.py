from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
import numpy as np
from . import paths
from .io_utils import write_json
from .protocol import FLOOR_SCHEDULE, MAX_SEED_OFFSET, PROTOCOL_ID, SEED, STRATUM_MERGE_MIN, TASKS, TEST_FRAC, VAL_FRAC

def allocate_counts(n: int) -> tuple[int, int, int]:
    if n <= 0:
        return (0, 0, 0)
    n_test = int(round(TEST_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    n_fit = n - n_test - n_val
    if n_fit < 0:
        overflow = -n_fit
        take = min(overflow, n_val)
        n_val -= take
        overflow -= take
        n_test -= overflow
        n_fit = n - n_test - n_val
    if n_fit == 0 and n >= 1:
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_fit = n - n_test - n_val
    return (n_fit, n_val, n_test)

def _histo_bin(row: dict[str, Any]) -> str:
    return row['idc_ilc'] if row.get('idc_ilc') in {'IDC', 'ILC'} else 'other'

def _er_bin(row: dict[str, Any]) -> str:
    if row.get('er') == 'Positive':
        return 'pos'
    if row.get('er') == 'Negative':
        return 'neg'
    return 'missing'

def stratum_cells(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[str]]:
    full: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (row['subtype_stratum'], _histo_bin(row), _er_bin(row))
        full[key].append(row['patient_id'])
    merged: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for key, ids in full.items():
        use = key if len(ids) >= STRATUM_MERGE_MIN else (key[0], '_', '_')
        merged[use].extend(ids)
    return {key: sorted(set(ids)) for key, ids in merged.items()}

def assign_split(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    cells = stratum_cells(rows)
    for key in sorted(cells):
        ids = list(cells[key])
        order = np.asarray(ids, dtype=object)
        rng.shuffle(order)
        n_fit, n_val, n_test = allocate_counts(len(order))
        chosen = order.tolist()
        for pid in chosen[:n_test]:
            assignment[pid] = 'test'
        for pid in chosen[n_test:n_test + n_val]:
            assignment[pid] = 'val'
        for pid in chosen[n_test + n_val:]:
            assignment[pid] = 'fit'
    return assignment

def floor_counts(rows: list[dict[str, Any]], assignment: dict[str, str]) -> dict[str, dict[str, int]]:
    counts = {part: {'her2': 0, 'ilc': 0, 'er_neg': 0, 'n': 0} for part in ('fit', 'val', 'test')}
    for row in rows:
        part = assignment[row['patient_id']]
        counts[part]['n'] += 1
        if row.get('subtype4') == 'BRCA_Her2':
            counts[part]['her2'] += 1
        if row.get('idc_ilc') == 'ILC':
            counts[part]['ilc'] += 1
        if row.get('er') == 'Negative':
            counts[part]['er_neg'] += 1
    return counts

def floors_ok(counts: dict[str, dict[str, int]], floors: dict[str, int]) -> bool:
    for part in ('val', 'test'):
        if counts[part]['her2'] < floors['her2']:
            return False
        if counts[part]['ilc'] < floors['ilc']:
            return False
        if counts[part]['er_neg'] < floors['er_neg']:
            return False
    return True

def search_split(rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    master = [row for row in rows if row['in_master']]
    if not master:
        raise RuntimeError('no master-universe patients')
    for floors in FLOOR_SCHEDULE:
        for offset in range(MAX_SEED_OFFSET):
            seed = SEED + offset
            assignment = assign_split(master, seed)
            counts = floor_counts(master, assignment)
            if floors_ok(counts, floors):
                return (assignment, {'protocol_id': PROTOCOL_ID, 'base_seed': SEED, 'seed': seed, 'offset': offset, 'floors': floors, 'lowered_floors': floors != FLOOR_SCHEDULE[0], 'counts': counts, 'n_master': len(master)})
    raise RuntimeError('no split satisfied the floor schedule')

def _task_members(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if not row['in_master']:
            continue
        if task == 'subtype4' and row['subtype4']:
            selected.append(row)
        elif task == 'luma_lumb' and row['luma_lumb']:
            selected.append(row)
        elif task == 'idc_ilc' and row['idc_ilc']:
            selected.append(row)
        elif task == 'er_ihc' and row['er']:
            selected.append(row)
    return selected

def _split_rows(rows: list[dict[str, Any]], assignment: dict[str, str], part: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if assignment.get(row['patient_id']) != part:
            continue
        out.append({'sample_id': row['patient_id'], 'patient_id': row['patient_id'], 'split': part, 'subtype4': row['subtype4'], 'luma_lumb': row['luma_lumb'], 'idc_ilc': row['idc_ilc'], 'er': row['er'], 'n_eligible_slides': row['n_eligible_slides']})
    return sorted(out, key=lambda item: item['patient_id'])

def write_splits(rows: list[dict[str, Any]], assignment: dict[str, Any], meta: dict[str, Any], root: Path | None=None) -> dict[str, Any]:
    base = paths.ensure_tree(root)
    dest = paths.split_dir(base)
    master = [row for row in rows if row['in_master']]
    hashes = {'train_rows': write_json(dest / 'train_rows.json', _split_rows(master, assignment, 'fit')), 'val_rows': write_json(dest / 'val_rows.json', _split_rows(master, assignment, 'val')), 'test_rows': write_json(dest / 'test_rows.json', _split_rows(master, assignment, 'test'))}
    units = [{'name': 'official', 'kind': 'tcga_brca_seed42_patient_70_15_15', 'fit': sorted((pid for pid, part in assignment.items() if part == 'fit')), 'val': sorted((pid for pid, part in assignment.items() if part == 'val')), 'predict': sorted((pid for pid, part in assignment.items() if part == 'val')), 'test': sorted((pid for pid, part in assignment.items() if part == 'test'))}]
    hashes['units'] = write_json(dest / 'units.json', units)
    task_counts = {}
    for task in TASKS:
        members = _task_members(master, task)
        payload = {'task': task, 'fit': sorted((row['patient_id'] for row in members if assignment[row['patient_id']] == 'fit')), 'val': sorted((row['patient_id'] for row in members if assignment[row['patient_id']] == 'val')), 'test': sorted((row['patient_id'] for row in members if assignment[row['patient_id']] == 'test'))}
        hashes[f'task_{task}'] = write_json(dest / f'task_{task}.json', payload)
        task_counts[task] = {part: len(payload[part]) for part in ('fit', 'val', 'test')}
    meta = {**meta, 'hashes': hashes, 'task_counts': task_counts, 'disjoint': _assert_disjoint(assignment), 'luma_inherits_master': True}
    hashes['split_meta'] = write_json(dest / 'split_meta.json', meta)
    write_json(dest / 'summary.json', {k: v for k, v in meta.items() if k != 'hashes'} | {'hashes': hashes})
    return meta

def _assert_disjoint(assignment: dict[str, str]) -> bool:
    buckets = defaultdict(set)
    for pid, part in assignment.items():
        buckets[part].add(pid)
    overlap = buckets['fit'] & buckets['val'] or buckets['fit'] & buckets['test'] or buckets['val'] & buckets['test']
    if overlap:
        raise RuntimeError(f'patient leakage: {sorted(overlap)[:8]}')
    return True

def build_splits(root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    table = pd.read_parquet(paths.label_dir(base) / 'patient_labels.parquet')
    rows = table.to_dict('records')
    assignment, meta = search_split(rows)
    return write_splits(rows, assignment, meta, base)
