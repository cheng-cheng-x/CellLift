from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import csv
import hashlib
from celllift.runtime import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
SPLIT_NAMES = ('train', 'val', 'test')
SPLIT_TO_CODE = {'train': 0, 'val': 1, 'test': 2}
CODE_TO_SPLIT = np.asarray(SPLIT_NAMES)

def sha256_file(path: Path, chunk: int=16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()

def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)

def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')

def atomic_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with tmp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)

def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))

def write_parquet_atomic(table: pa.Table, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(table, tmp, compression='zstd', compression_level=3, use_dictionary=True, row_group_size=131072)
    os.replace(tmp, path)
    return sha256_file(path)

def exact_subset(indices: np.ndarray, sizes: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray | None:
    if target_size < 0:
        return None
    order = rng.permutation(indices)
    reachable = np.zeros(target_size + 1, dtype=bool)
    parent_item = np.full(target_size + 1, -1, dtype=np.int32)
    parent_sum = np.full(target_size + 1, -1, dtype=np.int32)
    reachable[0] = True
    for item in order:
        size = int(sizes[item])
        if size > target_size:
            continue
        previous = reachable[:target_size - size + 1]
        destination = reachable[size:]
        new_previous_sums = np.flatnonzero(previous & ~destination)
        if new_previous_sums.size:
            new_sums = new_previous_sums + size
            reachable[new_sums] = True
            parent_item[new_sums] = int(item)
            parent_sum[new_sums] = new_previous_sums
        if reachable[target_size]:
            break
    if not reachable[target_size]:
        return None
    selected = []
    current = target_size
    while current:
        item = int(parent_item[current])
        if item < 0:
            raise RuntimeError('subset reconstruction failed')
        selected.append(item)
        current = int(parent_sum[current])
    return np.asarray(selected, dtype=np.int32)

def build_group_features(groups: list[dict[str, Any]], section_bin_width: int=10, quantiles: int=5) -> tuple[np.ndarray, list[str]]:
    sections = []
    for group in groups:
        sections.extend(group['sections'])
    max_section = max(sections) if sections else 1
    section_bin_count = int(math.ceil(max_section / section_bin_width))
    labels = [f'section_{i * section_bin_width + 1:03d}_{min((i + 1) * section_bin_width, max_section):03d}' for i in range(section_bin_count)]
    fields = ['nucleus_count', 'raw_cell_count', 'valid_cell_count', 'artifact_cell_count']
    values_by_field = {field: np.asarray([v for g in groups for v in g[field + '_values']], dtype=np.float64) for field in fields}
    edges_by_field = {field: np.quantile(values, np.linspace(0, 1, quantiles + 1)[1:-1]) if len(values) else np.asarray([]) for field, values in values_by_field.items()}
    for field in fields:
        for i in range(quantiles):
            labels.append(f'{field}_q{i + 1}')
    labels.extend(['total_nucleus_count', 'total_raw_cell_count', 'total_valid_cell_count', 'total_artifact_cell_count', 'layer_count'])
    features = np.zeros((len(groups), len(labels)), dtype=np.float64)
    for gi, group in enumerate(groups):
        cursor = 0
        for section in group['sections']:
            features[gi, (int(section) - 1) // section_bin_width] += 1
        cursor += section_bin_count
        for field in fields:
            edges = edges_by_field[field]
            for value in group[field + '_values']:
                bin_index = int(np.searchsorted(edges, float(value), side='right')) if len(edges) else 0
                features[gi, cursor + bin_index] += 1
            cursor += quantiles
        features[gi, cursor] = sum(group['nucleus_count_values'])
        features[gi, cursor + 1] = sum(group['raw_cell_count_values'])
        features[gi, cursor + 2] = sum(group['valid_cell_count_values'])
        features[gi, cursor + 3] = sum(group['artifact_cell_count_values'])
        features[gi, cursor + 4] = group['layer_count']
    return (features, labels)

def feature_score(split_features: np.ndarray, ratios: np.ndarray, global_features: np.ndarray) -> float:
    expected = ratios[:, None] * global_features[None, :]
    scale = np.maximum(global_features, 1.0)
    residual = (split_features - expected) / np.sqrt(scale)
    return float(np.square(residual).sum())

def optimize_same_size_swaps(assignment: np.ndarray, mutable: np.ndarray, sizes: np.ndarray, features: np.ndarray, ratios: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[np.ndarray, float, int]:
    global_features = features.sum(axis=0)
    split_features = np.stack([features[assignment == i].sum(axis=0) for i in range(3)])
    score = feature_score(split_features, ratios, global_features)
    buckets: dict[int, np.ndarray] = {}
    for size in sorted(set((int(x) for x in sizes[mutable]))):
        bucket = np.flatnonzero((sizes == size) & mutable)
        if len(bucket) >= 2:
            buckets[size] = bucket
    if not buckets:
        return (assignment, score, 0)
    eligible = np.asarray(list(buckets), dtype=np.int32)
    accepted = 0
    for _ in range(iterations):
        size = int(eligible[rng.integers(len(eligible))])
        left, right = rng.choice(buckets[size], size=2, replace=False)
        sl, sr = (int(assignment[left]), int(assignment[right]))
        if sl == sr:
            continue
        before = np.square((split_features[[sl, sr]] - ratios[[sl, sr], None] * global_features[None, :]) / np.sqrt(np.maximum(global_features, 1.0))[None, :]).sum()
        new_l = split_features[sl] - features[left] + features[right]
        new_r = split_features[sr] - features[right] + features[left]
        after = np.square((np.stack([new_l, new_r]) - ratios[[sl, sr], None] * global_features[None, :]) / np.sqrt(np.maximum(global_features, 1.0))[None, :]).sum()
        if after + 1e-12 < before:
            assignment[left], assignment[right] = (assignment[right], assignment[left])
            split_features[sl] = new_l
            split_features[sr] = new_r
            score += float(after - before)
            accepted += 1
    return (assignment, float(score), accepted)

def append_split_string_column(table: pa.Table, codes: np.ndarray, layer_column: str, split_name: str='split') -> pa.Table:
    layer_indices = table[layer_column].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    split_values = CODE_TO_SPLIT[codes[layer_indices]]
    return table.append_column(split_name, pa.array(split_values.tolist(), type=pa.dictionary(pa.int8(), pa.string())))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--old-split', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260720)
    parser.add_argument('--candidate-attempts', type=int, default=500)
    parser.add_argument('--swap-iterations', type=int, default=400000)
    args = parser.parse_args()
    started = time.time()
    dataset_root = args.new_root / '05_dataset'
    audit_path = args.new_root / '06_audit' / 'final_dataset_validation.json'
    if json.loads(audit_path.read_text(encoding='utf-8')).get('status') != 'PASS':
        raise RuntimeError(f'new dataset audit is not PASS: {audit_path}')
    old_rows = read_tsv(args.old_split)
    old_by_key = {(r['track_id'], int(r['section_id'])): r['split'] for r in old_rows}
    if len(old_by_key) != len(old_rows):
        raise RuntimeError('old split contains duplicate (track_id, section_id) keys')
    layers_table = pq.read_table(dataset_root / 'layers.parquet')
    layers = layers_table.to_pylist()
    n_layers = len(layers)
    if n_layers != len({int(r['layer_index']) for r in layers}):
        raise RuntimeError('layer_index is not unique')
    if sorted((int(r['layer_index']) for r in layers)) != list(range(n_layers)):
        raise RuntimeError('layer_index must be contiguous from 0')
    train_target = int(round(n_layers * 0.8))
    val_target = int(round(n_layers * 0.1))
    test_target = n_layers - train_target - val_target
    targets = {'train': train_target, 'val': val_target, 'test': test_target}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in layers:
        by_group[str(row['optimized_track_id'])].append(row)
    group_ids = sorted(by_group)
    groups: list[dict[str, Any]] = []
    forced: list[str | None] = []
    preserved_layer_rows = []
    for gid in group_ids:
        rows = by_group[gid]
        old_splits = []
        for row in rows:
            split = old_by_key.get((str(row['source_track_id']), int(row['section_id'])))
            if split:
                old_splits.append(split)
                preserved_layer_rows.append({'roi_layer_id': row['roi_layer_id'], 'optimized_track_id': row['optimized_track_id'], 'source_track_id': row['source_track_id'], 'section_id': int(row['section_id']), 'preserved_split': split})
        if len(set(old_splits)) > 1:
            raise RuntimeError(f'old split conflict inside optimized track {gid}: {Counter(old_splits)}')
        forced_split = old_splits[0] if old_splits else None
        forced.append(forced_split)
        groups.append({'optimized_track_id': gid, 'layer_count': len(rows), 'sections': [int(r['section_id']) for r in rows], 'source_track_ids': sorted({str(r['source_track_id']) for r in rows}), 'nucleus_count_values': [int(r['nucleus_count']) for r in rows], 'raw_cell_count_values': [int(r['raw_cell_count']) for r in rows], 'valid_cell_count_values': [int(r['valid_cell_count']) for r in rows], 'artifact_cell_count_values': [int(r['artifact_cell_count']) for r in rows]})
    sizes = np.asarray([g['layer_count'] for g in groups], dtype=np.int32)
    features, feature_labels = build_group_features(groups)
    assignment = np.full(len(groups), -1, dtype=np.int8)
    mutable = np.ones(len(groups), dtype=bool)
    forced_counts = Counter()
    for i, split in enumerate(forced):
        if split is not None:
            assignment[i] = SPLIT_TO_CODE[split]
            mutable[i] = False
            forced_counts[split] += int(sizes[i])
    remaining_targets = {name: targets[name] - forced_counts[name] for name in SPLIT_NAMES}
    if any((v < 0 for v in remaining_targets.values())):
        raise RuntimeError(f'forced old assignment exceeds targets: {remaining_targets}')
    if sum(remaining_targets.values()) != int(sizes[mutable].sum()):
        raise RuntimeError('remaining targets do not match mutable layer count')
    mutable_indices = np.flatnonzero(mutable).astype(np.int32)
    rng = np.random.default_rng(args.seed)
    ratios = np.asarray([targets[name] / n_layers for name in SPLIT_NAMES], dtype=np.float64)
    best_assignment = None
    best_score = math.inf
    feasible = 0
    for _ in range(args.candidate_attempts):
        test_idx = exact_subset(mutable_indices, sizes, remaining_targets['test'], rng)
        if test_idx is None:
            continue
        test_set = set((int(x) for x in test_idx))
        rest = np.asarray([x for x in mutable_indices if int(x) not in test_set], dtype=np.int32)
        val_idx = exact_subset(rest, sizes, remaining_targets['val'], rng)
        if val_idx is None:
            continue
        candidate = assignment.copy()
        candidate[mutable_indices] = SPLIT_TO_CODE['train']
        candidate[val_idx] = SPLIT_TO_CODE['val']
        candidate[test_idx] = SPLIT_TO_CODE['test']
        split_features = np.stack([features[candidate == i].sum(axis=0) for i in range(3)])
        score = feature_score(split_features, ratios, features.sum(axis=0))
        feasible += 1
        if score < best_score:
            best_score = score
            best_assignment = candidate
    if best_assignment is None:
        raise RuntimeError('could not find exact remaining assignment')
    final_assignment, final_score, accepted_swaps = optimize_same_size_swaps(best_assignment.copy(), mutable, sizes, features, ratios, args.swap_iterations, rng)
    group_to_split = {gid: SPLIT_NAMES[int(final_assignment[i])] for i, gid in enumerate(group_ids)}
    layer_split = np.empty(n_layers, dtype=np.int8)
    for row in layers:
        layer_split[int(row['layer_index'])] = SPLIT_TO_CODE[group_to_split[str(row['optimized_track_id'])]]
    counts = {name: int((layer_split == code).sum()) for name, code in SPLIT_TO_CODE.items()}
    if counts != targets:
        raise RuntimeError(f'exact final split mismatch: {counts} != {targets}')
    out = args.out_root
    if out.exists() and (not (out / '00_config' / 'split_lock.json').exists()):
        raise RuntimeError(f'output exists without lock: {out}')
    config = {'split_id': out.name, 'new_root': str(args.new_root), 'source_dataset_root': str(dataset_root), 'old_split_manifest': str(args.old_split), 'seed': args.seed, 'group_key': 'optimized_track_id', 'preserve_old_key': ['source_track_id', 'section_id'], 'targets': targets, 'candidate_attempts': args.candidate_attempts, 'swap_iterations': args.swap_iterations, 'feature_labels': feature_labels}
    lock = {'config': config, 'config_fingerprint': stable_hash(config), 'new_final_audit_sha256': sha256_file(audit_path), 'old_split_sha256': sha256_file(args.old_split), 'input_table_sha256': {'layers.parquet': sha256_file(dataset_root / 'layers.parquet'), 'edges.parquet': sha256_file(dataset_root / 'edges.parquet')}}
    lock_path = out / '00_config' / 'split_lock.json'
    if lock_path.is_file():
        existing = json.loads(lock_path.read_text(encoding='utf-8'))
        if existing != lock:
            raise RuntimeError('existing split_lock.json differs')
    atomic_json(lock_path, lock)
    atomic_json(out / '00_config' / 'split_config.json', config)
    split_arr = pa.array(CODE_TO_SPLIT[layer_split].tolist(), type=pa.dictionary(pa.int8(), pa.string()))
    layers_with_split = layers_table.append_column('split', split_arr)
    layer_sha = write_parquet_atomic(layers_with_split, out / '02_dataset' / 'layers.parquet')
    layer_rows = layers_with_split.to_pylist()
    layer_rows.sort(key=lambda r: (SPLIT_NAMES.index(r['split']), str(r['optimized_track_id']), int(r['section_id']), int(r['layer_index'])))
    layer_fields = ['split'] + [name for name in layers_table.column_names]
    atomic_tsv(out / '01_manifests' / 'all_layers_with_split.tsv', layer_rows, layer_fields)
    for name in SPLIT_NAMES:
        atomic_tsv(out / '01_manifests' / f'{name}_layers.tsv', [r for r in layer_rows if r['split'] == name], layer_fields)
    track_rows = []
    for i, gid in enumerate(group_ids):
        g = groups[i]
        split = group_to_split[gid]
        track_rows.append({'optimized_track_id': gid, 'split': split, 'forced_by_old_split': str(forced[i] is not None), 'preserved_old_split': forced[i] or '', 'layer_count': g['layer_count'], 'source_track_count': len(g['source_track_ids']), 'source_track_ids': ','.join(g['source_track_ids']), 'section_count': len(set(g['sections'])), 'section_min': min(g['sections']), 'section_max': max(g['sections']), 'total_nucleus_count': sum(g['nucleus_count_values']), 'total_raw_cell_count': sum(g['raw_cell_count_values']), 'total_valid_cell_count': sum(g['valid_cell_count_values']), 'total_artifact_cell_count': sum(g['artifact_cell_count_values']), 'stable_assignment_hash': hashlib.sha256(f'{args.seed}:{gid}'.encode('utf-8')).hexdigest()})
    atomic_tsv(out / '01_manifests' / 'track_assignment.tsv', track_rows, list(track_rows[0]))
    atomic_tsv(out / '01_manifests' / 'old_overlap_preservation.tsv', preserved_layer_rows, ['roi_layer_id', 'optimized_track_id', 'source_track_id', 'section_id', 'preserved_split'])
    summary_rows = []
    for name in SPLIT_NAMES:
        rows = [r for r in layer_rows if r['split'] == name]
        tracks = [r for r in track_rows if r['split'] == name]
        summary_rows.append({'split': name, 'target_roi_count': targets[name], 'roi_count': len(rows), 'actual_ratio': len(rows) / n_layers, 'optimized_track_count': len(tracks), 'forced_optimized_track_count': sum((t['forced_by_old_split'] == 'True' for t in tracks)), 'old_preserved_layer_count': sum((1 for r in preserved_layer_rows if r['preserved_split'] == name)), 'total_nucleus_count': sum((int(r['nucleus_count']) for r in rows)), 'total_raw_cell_count': sum((int(r['raw_cell_count']) for r in rows)), 'total_valid_cell_count': sum((int(r['valid_cell_count']) for r in rows)), 'total_artifact_cell_count': sum((int(r['artifact_cell_count']) for r in rows))})
    atomic_tsv(out / '01_manifests' / 'split_summary.tsv', summary_rows, list(summary_rows[0]))
    all_sections = sorted({int(r['section_id']) for r in layers})
    section_rows = []
    for section in all_sections:
        row = {'section_id': f'{section:03d}'}
        total = 0
        for name in SPLIT_NAMES:
            c = sum((int(r['section_id']) == section and r['split'] == name for r in layer_rows))
            row[f'{name}_count'] = c
            total += c
        row['total_count'] = total
        section_rows.append(row)
    atomic_tsv(out / '01_manifests' / 'section_distribution.tsv', section_rows, ['section_id', 'train_count', 'val_count', 'test_count', 'total_count'])
    edges_table = pq.read_table(dataset_root / 'edges.parquet')
    left = edges_table['left_layer_index'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    right = edges_table['right_layer_index'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(layer_split[left], layer_split[right]):
        bad = np.flatnonzero(layer_split[left] != layer_split[right])[:10]
        raise RuntimeError(f'edge crosses split boundary; examples={bad.tolist()}')
    edge_split_values = CODE_TO_SPLIT[layer_split[left]]
    edges_with_split = edges_table.append_column('split', pa.array(edge_split_values.tolist(), type=pa.dictionary(pa.int8(), pa.string())))
    edge_sha = write_parquet_atomic(edges_with_split, out / '02_dataset' / 'edges.parquet')
    edge_rows = edges_with_split.to_pylist()
    edge_counts = {}
    for name in SPLIT_NAMES:
        selected = [r for r in edge_rows if r['split'] == name]
        edge_counts[name] = {'edge_count': len(selected), 'gold_count': int(sum((int(r['gold_count']) for r in selected))), 'silver_nucleus_count': int(sum((int(r['silver_nucleus_count']) for r in selected))), 'unknown_count': int(sum((int(r['unknown_count']) for r in selected))), 'cell_only_count': int(sum((int(r['cell_only_count']) for r in selected))), 'cell_conflict_count': int(sum((int(r['cell_conflict_count']) for r in selected)))}
    links_table = pq.read_table(dataset_root / 'nucleus_links' / 'part-000.parquet')
    links_with_split = append_split_string_column(links_table, layer_split, 'left_layer_index')
    link_sha = write_parquet_atomic(links_with_split, out / '02_dataset' / 'nucleus_links' / 'part-000.parquet')
    link_split = np.asarray(links_with_split['split'].combine_chunks().to_pylist())
    quality = np.asarray(links_with_split['quality_class'].combine_chunks().to_pylist())
    cell_relation = np.asarray(links_with_split['cell_relation'].combine_chunks().to_pylist())
    link_counts = {}
    for name in SPLIT_NAMES:
        mask = link_split == name
        link_counts[name] = {'nucleus_link_count': int(mask.sum()), 'gold_count': int(((quality == 'gold') & mask).sum()), 'silver_nucleus_count': int(((quality == 'silver_nucleus') & mask).sum()), 'cell_conflict_count': int(((cell_relation == 'conflict') & mask).sum())}
    index_table = pq.read_table(dataset_root / 'nucleus_link_index' / 'part-000.parquet')
    index_with_split = append_split_string_column(index_table, layer_split, 'layer_index')
    index_sha = write_parquet_atomic(index_with_split, out / '02_dataset' / 'nucleus_link_index' / 'part-000.parquet')
    index_split = np.asarray(index_with_split['split'].combine_chunks().to_pylist())
    index_counts = {name: int((index_split == name).sum()) for name in SPLIT_NAMES}
    split_file_hashes: dict[str, dict[str, str]] = {}
    for name in SPLIT_NAMES:
        split_file_hashes[name] = {}
        for label, table, split_values, rel in [('layers', layers_with_split, np.asarray(layers_with_split['split'].combine_chunks().to_pylist()), f'{name}/layers.parquet'), ('edges', edges_with_split, np.asarray(edges_with_split['split'].combine_chunks().to_pylist()), f'{name}/edges.parquet'), ('nucleus_links', links_with_split, link_split, f'{name}/nucleus_links.parquet'), ('nucleus_link_index', index_with_split, index_split, f'{name}/nucleus_link_index.parquet')]:
            mask = pa.array((split_values == name).tolist())
            split_table = table.filter(mask)
            split_file_hashes[name][label] = write_parquet_atomic(split_table, out / '02_dataset_by_split' / rel)
    manifest_hashes = {p.name: sha256_file(p) for p in sorted((out / '01_manifests').glob('*.tsv'))}
    overlaps = {}
    split_track_sets = {name: {r['optimized_track_id'] for r in track_rows if r['split'] == name} for name in SPLIT_NAMES}
    split_layer_sets = {name: {r['roi_layer_id'] for r in layer_rows if r['split'] == name} for name in SPLIT_NAMES}
    for i, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[i + 1:]:
            overlaps[f'{left_name}_{right_name}_optimized_track'] = len(split_track_sets[left_name] & split_track_sets[right_name])
            overlaps[f'{left_name}_{right_name}_roi_layer'] = len(split_layer_sets[left_name] & split_layer_sets[right_name])
    if any(overlaps.values()):
        raise RuntimeError(f'split leakage detected: {overlaps}')
    final = {'status': 'PASS', 'split_id': out.name, 'group_key': 'optimized_track_id', 'preserve_old_key': ['source_track_id', 'section_id'], 'seed': args.seed, 'source_layers': n_layers, 'source_optimized_tracks': len(group_ids), 'source_old_overlap_layers': len(preserved_layer_rows), 'forced_layer_counts_all_layers_in_forced_groups': dict(forced_counts), 'targets': targets, 'layer_counts': counts, 'summary_by_split': summary_rows, 'edge_counts_by_split': edge_counts, 'nucleus_link_counts_by_split': link_counts, 'nucleus_link_index_rows_by_split': index_counts, 'total_edges': int(edges_with_split.num_rows), 'total_nucleus_links': int(links_with_split.num_rows), 'total_nucleus_link_index_rows': int(index_with_split.num_rows), 'old_preservation_conflict_count': 0, 'overlaps': overlaps, 'optimizer': {'candidate_attempts': args.candidate_attempts, 'feasible_candidates': feasible, 'initial_best_feature_score': best_score, 'optimized_feature_score': final_score, 'accepted_same_size_swaps': accepted_swaps, 'feature_labels': feature_labels}, 'parquet_sha256': {'layers': layer_sha, 'edges': edge_sha, 'nucleus_links_part_000': link_sha, 'nucleus_link_index_part_000': index_sha}, 'split_file_hashes': split_file_hashes, 'manifest_hashes': manifest_hashes, 'input_lock': lock, 'elapsed_seconds': time.time() - started}
    if sum((v['nucleus_link_count'] for v in link_counts.values())) != int(links_with_split.num_rows):
        final['status'] = 'FAIL'
    if sum((v['edge_count'] for v in edge_counts.values())) != int(edges_with_split.num_rows):
        final['status'] = 'FAIL'
    atomic_json(out / '03_audit' / 'final_audit.json', final)
    readme = f"# RSG-6 dataset split\n\nStatus: **{final['status']}**\n\n- Split ID: `{out.name}`\n- Grouping unit: complete `optimized_track_id`\n- Seed: `{args.seed}`\n- Source layers: {n_layers}\n- Old-overlap layers preserved by `(source_track_id, section_id)`: {len(preserved_layer_rows)}\n- Train/val/test layers: {counts['train']} / {counts['val']} / {counts['test']}\n- Train/val/test nucleus links: {link_counts['train']['nucleus_link_count']} / {link_counts['val']['nucleus_link_count']} / {link_counts['test']['nucleus_link_count']}\n- Split leakage on optimized tracks and ROI layers: 0\n- Edge crossing split boundary: 0\n\nThe split keeps every optimized RSG-6 track in exactly one split. If an RSG-6 layer corresponds to a layer in the previous 3,795-layer training dataset, its old train/val/test membership is preserved; all other optimized tracks are deterministically assigned to satisfy an exact 80/10/10 layer-count split and to balance section and segmentation-count features.\n"
    atomic_text(out / '03_audit' / 'README.md', readme)
    print(json.dumps(final, indent=2, sort_keys=True, allow_nan=False))
    return 0 if final['status'] == 'PASS' else 2
if __name__ == '__main__':
    raise SystemExit(main())
