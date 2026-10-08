from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .. import paths
from ..io_utils import atomic_json, atomic_parquet
from ..protocol import MIN_TILES_EVALUABLE, PROTOCOL_ID, TASKS
from . import jobs, layout
from .loaders import ThreeViewLoader

def _read_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)

def merge_tile_dino(base: Path) -> list[dict[str, Any]]:
    import pandas as pd
    frames = []
    for path in sorted((base / 'features').glob('tile_dino_*.parquet')):
        frames.append(pd.read_parquet(path))
    if not frames:
        return []
    merged = pd.concat(frames, ignore_index=True)
    rows = merged.to_dict('records')
    atomic_parquet(base / 'features' / 'tile_dino.parquet', rows)
    return rows

def write_indices(data_root: Path | None=None) -> dict[str, Any]:
    base = layout.ensure(data_root)
    manifest = jobs.load_tile_manifest(data_root)
    inventory = jobs.load_slide_inventory(data_root)
    labels = jobs.load_patient_labels(data_root)
    dino_rows = {str(row['tile_id']): row for row in merge_tile_dino(base)}
    tiles = []
    failures = []
    for raw in manifest.to_dict('records'):
        row = jobs.enrich_tile(raw, inventory, base)
        rgb = Path(row['rgb_path'])
        dino = dino_rows.get(str(row['tile_id']))
        status = 'success' if rgb.is_file() and dino is not None else 'fail'
        record = {'tile_id': row['tile_id'], 'graph_id': row['graph_id'], 'patient_id': row['patient_id'], 'slide_id': row['slide_id'], 'split': row['split'], 'wsi_source_path': row['wsi_source_path'], 'rgb_path': row['rgb_path'], 'status': status, 'empty_nuclei': bool(dino['empty_nuclei']) if dino else None, 'n_nodes': int(dino['n_nodes']) if dino else None, 'dino': dino['dino'] if dino else [], 'mpp_x': row['mpp_x'], 'mpp_y': row['mpp_y']}
        tiles.append(record)
        if status != 'success':
            failures.append({'tile_id': row['tile_id'], 'slide_id': row['slide_id'], 'reason': 'missing_rgb_or_dino'})
    atomic_parquet(base / 'indices' / 'tiles.parquet', tiles)
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tiles:
        by_patient[str(row['patient_id'])].append(row)
    uneval = set()
    uneval_path = paths.tile_dir(data_root) / 'unevaluable.json'
    if uneval_path.is_file():
        from celllift.runtime import json
        uneval = {str(item['patient_id']) for item in json.loads(uneval_path.read_text(encoding='utf-8'))}
    patients = []
    for _, label in labels.iterrows():
        pid = str(label['patient_id'])
        if not bool(label.get('in_master', False)):
            continue
        bag = by_patient.get(pid, [])
        n_success = sum((1 for item in bag if item['status'] == 'success'))
        n_empty = sum((1 for item in bag if item.get('empty_nuclei')))
        patients.append({'patient_id': pid, 'split': bag[0]['split'] if bag else '', 'n_tiles': len(bag), 'n_success': n_success, 'n_empty_nuclei': n_empty, 'evaluable': n_success >= MIN_TILES_EVALUABLE and pid not in uneval, 'unevaluable_original': pid in uneval, 'subtype4': label.get('subtype4') or '', 'subtype4_id': label.get('subtype4_id'), 'luma_lumb': label.get('luma_lumb') or '', 'luma_lumb_id': label.get('luma_lumb_id'), 'idc_ilc': label.get('idc_ilc') or '', 'idc_ilc_id': label.get('idc_ilc_id'), 'er': label.get('er') or '', 'er_id': label.get('er_id')})
    atomic_parquet(base / 'indices' / 'patients.parquet', patients)
    import pyarrow as pa
    import pyarrow.parquet as pq
    fail_path = base / 'qc' / 'failures.parquet'
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        atomic_parquet(fail_path, failures)
    else:
        table = pa.table({'tile_id': pa.array([], pa.string()), 'slide_id': pa.array([], pa.string()), 'reason': pa.array([], pa.string())})
        pq.write_table(table, fail_path)
    return {'tiles': len(tiles), 'patients': len(patients), 'failures': len(failures)}

def coverage_summary(data_root: Path | None=None) -> dict[str, Any]:
    base = layout.root(data_root)
    tiles = _read_parquet(base / 'indices' / 'tiles.parquet')
    patients = _read_parquet(base / 'indices' / 'patients.parquet')
    loader = ThreeViewLoader(base, tiles.to_dict('records'))
    probe = tiles[tiles['status'] == 'success'].head(3)
    for _, row in probe.iterrows():
        loader.aligned(str(row['graph_id']))
    split_cov = {}
    for split, part in tiles.groupby('split'):
        n = len(part)
        ok = int((part['status'] == 'success').sum())
        empty = int(part['empty_nuclei'].fillna(False).sum())
        split_cov[str(split)] = {'tiles': n, 'success': ok, 'coverage': ok / n if n else 0.0, 'empty_nuclei': empty, 'empty_rate': empty / ok if ok else 0.0}
    task_counts = {}
    for task in TASKS:
        col = {'subtype4': 'subtype4', 'luma_lumb': 'luma_lumb', 'idc_ilc': 'idc_ilc', 'er_ihc': 'er'}[task]
        labeled = patients[patients[col].astype(str).str.len() > 0]
        eval_labeled = labeled[labeled['evaluable']]
        task_counts[task] = {'nominal': int(len(labeled)), 'evaluable': int(len(eval_labeled)), 'by_split_nominal': labeled.groupby('split').size().to_dict() if len(labeled) else {}, 'by_split_evaluable': eval_labeled.groupby('split').size().to_dict() if len(eval_labeled) else {}}
    payload = {'protocol_id': PROTOCOL_ID, 'status': 'PASS' if int((tiles['status'] != 'success').sum()) == 0 else 'PARTIAL', 'n_tiles': int(len(tiles)), 'n_success': int((tiles['status'] == 'success').sum()), 'n_patients': int(len(patients)), 'n_evaluable_patients': int(patients['evaluable'].sum()), 'splits': split_cov, 'tasks': task_counts}
    invalid_rate = None
    geom_files = list((base / 'features' / 'geometry_tokens').glob('*.npz'))
    if geom_files:
        invalid = 0
        nodes = 0
        for path in geom_files:
            with np.load(path, allow_pickle=False) as data:
                valid = np.asarray(data['valid3d'], bool)
                include = np.asarray(data['include'], bool)
                nodes += int(include.sum())
                invalid += int((include & ~valid).sum())
        invalid_rate = invalid / nodes if nodes else 0.0
        payload['invalid_3d_rate'] = invalid_rate
        payload['nodes_included'] = nodes
    atomic_json(base / 'qc' / 'summary.json', payload)
    return payload
