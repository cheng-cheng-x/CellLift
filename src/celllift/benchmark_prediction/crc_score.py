from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .metrics import patient_auroc
from .paths import split_root

def hard_vote_fraction(tile_logits: np.ndarray) -> float:
    logits = np.asarray(tile_logits)
    if logits.ndim == 1:
        pred = (logits > 0).astype(np.int64)
    else:
        pred = logits.argmax(1)
    return float(pred.mean()) if pred.size else 0.0

def hard_vote_from_probs(tile_probs: np.ndarray) -> float:
    probs = np.asarray(tile_probs, np.float64)
    if probs.ndim == 1:
        pred = (probs >= 0.5).astype(np.int64)
    else:
        pred = probs.argmax(1)
    return float(pred.mean()) if pred.size else 0.0

def group_hard_vote(patient_ids: Sequence[str], labels: Sequence[int], logits: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    label_of: dict[str, int] = {}
    for pid, label, logit in zip(patient_ids, labels, logits):
        key = str(pid)
        grouped[key].append(np.asarray(logit))
        label_of[key] = int(label)
    keys = sorted(grouped)
    scores = np.asarray([hard_vote_fraction(np.stack(grouped[key], 0)) for key in keys], np.float64)
    y = np.asarray([label_of[key] for key in keys], np.int64)
    return (keys, y, scores)

def patient_auroc_hard_vote(patient_ids: Sequence[str], labels: Sequence[int], logits: np.ndarray) -> float:
    _, y, scores = group_hard_vote(patient_ids, labels, logits)
    return patient_auroc(y, scores)

def crc_rows(split: str) -> list[dict[str, Any]]:
    path = split_root() / 'tcga_crc_msi' / f'{split}_rows.json'
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding='utf-8'))

def qualified_patient_ids(split: str='test') -> list[str]:
    if split == 'test':
        return [str(row['sample_id']) for row in crc_rows('test')]
    rows = crc_rows('train')
    if split == 'fit':
        return [str(row['sample_id']) for row in rows if row.get('split') == 'fit']
    if split == 'val':
        return [str(row['sample_id']) for row in rows if row.get('split') == 'val']
    return [str(row['sample_id']) for row in rows]

def filter_to_qualified(rows: list[dict[str, Any]], split: str='test') -> list[dict[str, Any]]:
    wanted = set(qualified_patient_ids(split))
    return [row for row in rows if str(row.get('sample_id') or row.get('patient_id') or '') in wanted]

def qualification_diff() -> dict[str, Any]:
    from .splits import load_units
    unit_test = set(map(str, load_units('tcga_crc_msi')[0].get('test') or []))
    qualified = set(qualified_patient_ids('test'))
    extra = sorted(unit_test - qualified)
    missing = sorted(qualified - unit_test)
    tile_count: dict[str, int] = {}
    try:
        from .splits import LABEL_PATHS, _norm_split, _read_table
        for row in _read_table(__import__('pathlib').Path(LABEL_PATHS['tcga_crc_msi'])):
            if _norm_split(row.get('official_split')) != 'test':
                continue
            pid = str(row['patient_id'])
            tile_count[pid] = tile_count.get(pid, 0) + 1
    except Exception:
        tile_count = {}
    excluded = []
    for pid in extra:
        n = tile_count.get(pid)
        excluded.append({'patient_id': pid, 'n_tiles': n, 'reason': f'lt_10_tiles (n_tiles={n})' if n is not None and n < 10 else 'not_in_test_rows'})
    return {'unit_test_n': len(unit_test), 'qualified_n': len(qualified), 'excluded_from_unit_test': excluded, 'qualified_missing_from_unit': missing, 'qualified_ids': sorted(qualified)}
