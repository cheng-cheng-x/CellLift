from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
from celllift.runtime import json
import os
from collections import Counter
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .utils import atomic_json, atomic_parquet, atomic_text, read_json, read_parquet_rows, sha256_file

def _read_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_parquet_rows(path))
    return rows

def build_pilot_contact_sheets(cfg: dict[str, Any]) -> list[str]:
    from PIL import Image
    data_root = Path(cfg['paths']['data_root'])
    rows = read_parquet_rows(data_root / '00_manifest/pilot_manifest.parquet')
    nucleus_only = cfg.get('input', {}).get('mode') == 'nucleus_only'
    destination = data_root / '05_qc' / ('nucleus_only_pilot_contact_sheets' if nucleus_only else 'pilot_contact_sheets')
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for page_start in range(0, len(rows), 16):
        page_rows = rows[page_start:page_start + 16]
        canvas = Image.new('RGB', (4 * 384, 4 * 224), 'white')
        for local_index, row in enumerate(page_rows):
            with Image.open(row['rgb_path']) as image:
                rgb = image.convert('RGB').resize((192, 192))
            nucleus = np.asarray(Image.fromarray(np.load(row['nucleus_mask_path']).astype(np.int32), mode='I').resize((192, 192), Image.Resampling.NEAREST))
            cell = None if nucleus_only else np.asarray(Image.fromarray(np.load(row['cell_mask_path']).astype(np.int32), mode='I').resize((192, 192), Image.Resampling.NEAREST))

            def boundary(mask: np.ndarray) -> np.ndarray:
                result = np.zeros(mask.shape, dtype=bool)
                result[1:, :] |= mask[1:, :] != mask[:-1, :]
                result[:, 1:] |= mask[:, 1:] != mask[:, :-1]
                return result & (mask > 0)
            overlay_array = np.asarray(rgb).copy()
            if cell is not None:
                overlay_array[boundary(cell)] = [0, 220, 0]
            overlay_array[boundary(nucleus)] = [255, 0, 0]
            overlay = Image.fromarray(overlay_array, mode='RGB')
            tile = Image.new('RGB', (384, 224), 'white')
            tile.paste(rgb, (0, 0))
            tile.paste(overlay, (192, 0))
            x = local_index % 4 * 384
            y = local_index // 4 * 224
            canvas.paste(tile, (x, y))
        output = destination / f'pilot_{page_start // 16:02d}.png'
        temporary = output.with_name(f'.{output.stem}.tmp.{os.getpid()}.png')
        canvas.save(temporary, format='PNG', compress_level=1)
        os.replace(temporary, output)
        outputs.append(str(output))
    atomic_parquet(destination / 'contact_sheet_manifest.parquet', [{'order': index, 'page': index // 16, 'cell': index % 16, 'patch_id': row['patch_id'], 'label_name': row['label_name'], 'official_split': row['official_split']} for index, row in enumerate(rows)])
    return outputs

def verify_pilot(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    data_root = Path(cfg['paths']['data_root'])
    statuses = _read_many(sorted((data_root / '00_manifest').glob('segment_pilot_shard_*.parquet')))
    expected = len(read_parquet_rows(data_root / '00_manifest/pilot_manifest.parquet'))
    complete = [row for row in statuses if row.get('status') == 'complete']
    if len(statuses) != expected:
        raise RuntimeError(f'pilot status rows {len(statuses)} != expected {expected}')
    nonempty_fraction = sum((int(row.get('nucleus_count', 0)) > 0 for row in complete)) / expected
    nucleus_only = cfg.get('input', {}).get('mode') == 'nucleus_only'
    matched_values = [float(row['nucleus_matched_fraction']) for row in complete if row.get('nucleus_matched_fraction') is not None]
    matched_median = float(np.median(matched_values)) if matched_values else None
    sheets = build_pilot_contact_sheets(cfg)
    pilot_graph_root = data_root / '05_qc' / ('nucleus_only_pilot_graph_cache' if nucleus_only else 'pilot_graph_cache')
    graph_rows = _read_many(sorted(pilot_graph_root.glob('graph_index_pilot_workshard_*.parquet')))
    graph_excluded = _read_many(sorted(pilot_graph_root.glob('excluded_pilot_workshard_*.parquet')))
    automatic_pass = len(complete) == expected and nonempty_fraction >= 0.99 and (len(graph_rows) == expected) and (not graph_excluded) and (nucleus_only or (matched_median is not None and matched_median >= 0.75))
    review_path = data_root / '05_qc' / ('nucleus_visual_review.json' if nucleus_only else 'visual_review.json')
    visual = read_json(review_path) if review_path.is_file() else {'status': 'PENDING'}
    visual_pass = visual.get('status') == 'PASS' and float(visual.get('usable_fraction', 0.0)) >= 0.9 and (float(visual.get('minimum_stratum_usable_fraction', 0.0)) >= 0.8)
    status = 'PASS' if automatic_pass and visual_pass else 'FAIL' if not automatic_pass or visual.get('status') == 'FAIL' else 'PENDING_VISUAL_REVIEW'
    report = {'status': status, 'automatic_gate': 'PASS' if automatic_pass else 'FAIL', 'visual_gate': visual.get('status', 'PENDING'), 'expected': expected, 'complete': len(complete), 'nonempty_fraction': nonempty_fraction, 'model_ready_graphs': len(graph_rows), 'graph_excluded': len(graph_excluded), 'median_nucleus_matched_fraction': matched_median, 'input_mode': 'nucleus_only' if nucleus_only else 'dual', 'contact_sheets': sheets}
    atomic_json(data_root / '05_qc' / ('nucleus_only_pilot_gate.status.json' if nucleus_only else 'pilot_gate.status.json'), report)
    return report

def _lmdb_manifest(graph_root: Path, shards: int=64) -> list[dict[str, Any]]:
    import lmdb
    rows: list[dict[str, Any]] = []
    for shard in range(shards):
        directory = graph_root / f'graph_cache_{shard:02d}.lmdb'
        data = directory / 'data.mdb'
        if not data.is_file():
            raise FileNotFoundError(data)
        env = lmdb.open(str(directory), subdir=True, readonly=True, lock=False, readahead=False)
        try:
            entries = int(env.stat()['entries'])
        finally:
            env.close()
        rows.append({'prefix': 'graph_cache', 'shard': shard, 'path': str(data), 'entries': entries, 'size_bytes': data.stat().st_size, 'sha256': sha256_file(data)})
    return rows

def verify_full(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    data_root = Path(cfg['paths']['data_root'])
    inventory = read_json(data_root / '00_manifest/inventory.status.json')
    if inventory.get('status') != 'PASS':
        raise RuntimeError('inventory gate is not PASS')
    manifest = read_parquet_rows(data_root / '00_manifest/patch_manifest.parquet')
    expected = len(manifest)
    standard = _read_many(sorted((data_root / '00_manifest').glob('standardize_full_shard_*.parquet')))
    segments = _read_many(sorted((data_root / '00_manifest').glob('segment_full_shard_*.parquet')))
    graph_root = data_root / '03_graph_cache'
    graph_rows = _read_many(sorted(graph_root.glob('graph_index_full_workshard_*.parquet')))
    excluded = _read_many(sorted(graph_root.glob('excluded_full_workshard_*.parquet')))
    if len(standard) != expected or any((row.get('status') != 'complete' for row in standard)):
        raise RuntimeError(f'standardization is incomplete: {len(standard)}/{expected}')
    if len(segments) != expected or any((row.get('status') != 'complete' for row in segments)):
        raise RuntimeError(f'segmentation is incomplete: {len(segments)}/{expected}')
    if len(graph_rows) + len(excluded) != expected:
        raise RuntimeError(f'graph accounting mismatch: {len(graph_rows)}+{len(excluded)} != {expected}')
    exclusion_fraction = len(excluded) / expected
    stratum_total = Counter((f"{row['official_split']}:{row['label_name']}" for row in manifest))
    stratum_excluded = Counter((f"{row['official_split']}:{row['label_name']}" for row in excluded))
    stratum_rates = {key: stratum_excluded[key] / value for key, value in stratum_total.items()}
    rate_pass = exclusion_fraction <= 0.005 and max(stratum_rates.values(), default=0) <= 0.01
    atomic_parquet(graph_root / 'graph_index.parquet', sorted(graph_rows, key=lambda row: int(row['layer_idx'])))
    atomic_parquet(graph_root / 'excluded_manifest.parquet', excluded)
    cache_manifest = _lmdb_manifest(graph_root)
    atomic_parquet(graph_root / 'cache_manifest.parquet', cache_manifest)
    report = {'status': 'PASS' if rate_pass else 'FAIL', 'dataset': dataset, 'input_mode': cfg.get('input', {}).get('mode', 'dual'), 'expected_patches': expected, 'model_ready_graphs': len(graph_rows), 'excluded': len(excluded), 'exclusion_fraction': exclusion_fraction, 'stratum_exclusion_rates': stratum_rates, 'lmdb_entries': sum((row['entries'] for row in cache_manifest)), 'feature_stats_sha256': read_json(graph_root / 'inference_feature_stats.json')['source_sha256']}
    if report['lmdb_entries'] != len(graph_rows):
        report['status'] = 'FAIL'
        report['lmdb_entry_mismatch'] = True
    atomic_json(data_root / '05_qc/final_gate.status.json', report)
    return report

def verify(cfg: dict[str, Any], dataset: str, *, pilot: bool=False) -> dict[str, Any]:
    return verify_pilot(cfg, dataset) if pilot else verify_full(cfg, dataset)
