from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
from celllift.runtime import json
import numpy as np
from PIL import Image
from .constants import ARVANITI_BOARDS, ARVANITI_DATA, ARVANITI_PATCH_NATIVE, LIZARD_DATA, SEED
from .io_utils import atomic_json, read_parquet, sha256_file, write_parquet

def _board_role(board: str) -> str | None:
    for role, boards in ARVANITI_BOARDS.items():
        if board in boards:
            return role
    return None

def author_patch_coords(width: int, height: int, patch_size: int=ARVANITI_PATCH_NATIVE) -> list[tuple[int, int, int]]:
    step = patch_size // 2
    coords = []
    index = 0
    for y in range(0, height - patch_size, step):
        for x in range(0, width - patch_size, step):
            coords.append((index, x, y))
            index += 1
    return coords

def author_patch_label(row0: int, col0: int, mask: np.ndarray, patch_size: int=ARVANITI_PATCH_NATIVE) -> int:
    window = patch_size // 3
    center = mask[row0 + window:row0 + 2 * window, col0 + window:col0 + 2 * window]
    found = np.unique(center)
    found = found[found < 4]
    return int(found[0]) if len(found) == 1 else 4

def _too_white(rgb: np.ndarray, limit: int=180) -> bool:
    return float(rgb.mean()) > limit

def freeze_arvaniti(data_root: Path | None=None) -> dict:
    from .audit import _decode_mask
    data_root = Path(data_root or ARVANITI_DATA)
    inv = data_root / '00_manifest' / 'corrected_set_encoding'
    rgb_rows = read_parquet(inv / 'rgb.parquet')
    mask_rows = read_parquet(inv / 'masks.parquet')
    masks_by = {(row['core_id'], row['reader']): row for row in mask_rows}
    core_rows = []
    for rgb in rgb_rows:
        role = _board_role(rgb['board'])
        if role is None:
            continue
        readers = ['train'] if role != 'TEST' else ['pathologist1', 'pathologist2']
        present = all(((rgb['core_id'], reader) in masks_by for reader in readers))
        core_rows.append({**rgb, 'role': role, 'readers': json.dumps(readers), 'masks_present': bool(present)})
    patch_rows = []
    dropped = Counter()
    for core_index, core in enumerate(core_rows):
        if core_index % 25 == 0:
            print(f'arvaniti freeze {core_index}/{len(core_rows)} windows={len(patch_rows)}', flush=True)
        if not core['masks_present']:
            dropped['missing_mask'] += 1
            continue
        rgb = np.array(Image.open(core['path']).convert('RGB'))
        height, width = rgb.shape[:2]
        readers = json.loads(core['readers'])
        decoded = {reader: _decode_mask(Path(masks_by[core['core_id'], reader]['path'])) for reader in readers}
        for patch_index, row0, col0 in author_patch_coords(width, height):
            crop = rgb[row0:row0 + ARVANITI_PATCH_NATIVE, col0:col0 + ARVANITI_PATCH_NATIVE]
            if crop.shape[0] != ARVANITI_PATCH_NATIVE or crop.shape[1] != ARVANITI_PATCH_NATIVE:
                dropped['partial'] += 1
                continue
            labels = {reader: author_patch_label(row0, col0, decoded[reader]) for reader in readers}
            if core['role'] != 'TEST':
                label = labels['train']
                if label == 4:
                    dropped['unlabelled_or_mixed'] += 1
                    continue
                if _too_white(crop):
                    dropped['too_white'] += 1
                    continue
                patch_rows.append(_arvaniti_patch(core, patch_index, row0, col0, label, None, None))
            else:
                if labels['pathologist1'] == 4 or labels['pathologist2'] == 4:
                    dropped['unlabelled_or_mixed'] += 1
                    continue
                if _too_white(crop):
                    dropped['too_white'] += 1
                    continue
                patch_rows.append(_arvaniti_patch(core, patch_index, row0, col0, labels['pathologist1'], labels['pathologist1'], labels['pathologist2']))
    out = data_root / '04_labels_splits'
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(out / 'cores.parquet', core_rows)
    write_parquet(out / 'windows.parquet', patch_rows)
    counts = {'cores': dict(Counter((row['role'] for row in core_rows))), 'windows': dict(Counter((row['role'] for row in patch_rows))), 'dropped': dict(dropped), 'label_hist': dict(Counter((int(row['label_id']) for row in patch_rows if row['role'] != 'TEST'))), 'test_p1_hist': dict(Counter((int(row['label_id_p1']) for row in patch_rows if row['role'] == 'TEST'))), 'test_p2_hist': dict(Counter((int(row['label_id_p2']) for row in patch_rows if row['role'] == 'TEST')))}
    sha = {'cores_parquet_sha256': sha256_file(out / 'cores.parquet'), 'windows_parquet_sha256': sha256_file(out / 'windows.parquet')}
    payload = {'status': 'FROZEN', 'seed': None, 'rule': 'author_boards_ZT111_199_204_fit_ZT76_val_ZT80_test', 'patch_script': 'eiriniar/gleason_CNN/utils/create_patches.py', 'white_limit': 180, 'test_requires_both_pathologists': True, **counts, **sha}
    atomic_json(out / 'split_lock.json', payload)
    atomic_json(data_root / '00_manifest' / 'corrected_set_encoding' / 'split_lock.json', payload)
    return payload

def _arvaniti_patch(core: dict, patch_index: int, row0: int, col0: int, label: int, p1, p2) -> dict:
    scale = 0.5
    graph_id = f"{core['core_id']}_patch_{patch_index:03d}"
    return {'graph_id': graph_id, 'core_id': core['core_id'], 'board': core['board'], 'role': core['role'], 'author_patch_index': int(patch_index), 'row0_native': int(row0), 'col0_native': int(col0), 'size_native': ARVANITI_PATCH_NATIVE, 'row0_target': int(round(row0 * scale)), 'col0_target': int(round(col0 * scale)), 'size_target': 375, 'rgb_path': core['path'], 'label_id': int(label), 'label_name': {0: 'benign', 1: 'G3', 2: 'G4', 3: 'G5'}.get(int(label), 'ignore'), 'label_id_p1': -1 if p1 is None else int(p1), 'label_id_p2': -1 if p2 is None else int(p2), 'patient_id': core['core_id'], 'official_split': {'FIT': 'train', 'VAL': 'val', 'TEST': 'test'}[core['role']]}

def freeze_lizard(data_root: Path | None=None, *, revision: str='conditional_geometry') -> dict:
    data_root = Path(data_root or LIZARD_DATA)
    inv = data_root / '00_manifest' / ('corrected_conditional_geometry' if revision == 'conditional_geometry' else 'corrected_set_encoding')
    rois = read_parquet(inv / 'rois.parquet')
    groups_payload = json.loads((inv / 'groups.json').read_text(encoding='utf-8'))
    group_members = {item['group_id']: item['members'] for item in groups_payload['groups']}
    roi_of = {row['roi_id']: row for row in rois}
    group_roles = {}
    moved = []
    for group_id, members in group_members.items():
        splits = {str(roi_of[m].get('split', '')).replace('.0', '') for m in members if m in roi_of}
        splits = {item.split('.')[0] for item in splits}
        if '1' in splits:
            group_roles[group_id] = 'TEST'
            if splits != {'1'}:
                moved.append({'group_id': group_id, 'members': members, 'from_splits': sorted(splits), 'to': 'TEST'})
        else:
            group_roles[group_id] = 'DEV'
    rng = np.random.RandomState(SEED)
    by_source: dict[str, list[str]] = defaultdict(list)
    for group_id, role in group_roles.items():
        if role != 'DEV':
            continue
        sources = sorted({str(roi_of[m].get('source', '')) for m in group_members[group_id] if m in roi_of})
        source = sources[0] if sources else ''
        by_source[source].append(group_id)
    for source, ids in by_source.items():
        ids.sort()
        rng.shuffle(ids)
        n_val = max(1, int(round(0.2 * len(ids)))) if len(ids) >= 2 else 0
        for index, group_id in enumerate(ids):
            group_roles[group_id] = 'VAL' if index < n_val else 'FIT'
    roi_rows = []
    for row in rois:
        group_id = row['group_id']
        role = group_roles.get(group_id, 'FIT')
        roi_rows.append({**row, 'role': role, 'official_split': {'FIT': 'train', 'VAL': 'val', 'TEST': 'test'}[role], 'patient_id': group_id})
    out = data_root / ('04_labels_splits_conditional_geometry' if revision == 'conditional_geometry' else '04_labels_splits')
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(out / 'rois.parquet', roi_rows)
    atomic_json(out / 'group_moves.json', {'moved_to_test': moved})
    counts = {'roi_roles': dict(Counter((row['role'] for row in roi_rows))), 'group_roles': dict(Counter(group_roles.values())), 'source_by_role': {role: dict(Counter((str(row.get('source', '')) for row in roi_rows if row['role'] == role))) for role in ('FIT', 'VAL', 'TEST')}, 'n_moved_groups': len(moved)}
    payload = {'status': 'FROZEN', 'seed': SEED, 'rule': 'split1_internal_test_plus_seed42_group_80_20_source_stratified', 'note': 'project split, not paper 3-fold, not withheld TCGA external test', **counts, 'rois_parquet_sha256': sha256_file(out / 'rois.parquet'), 'group_moves_sha256': sha256_file(out / 'group_moves.json')}
    atomic_json(out / 'split_lock.json', payload)
    atomic_json(inv / 'split_lock.json', payload)
    return payload

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['arvaniti', 'lizard', 'both'])
    args = parser.parse_args()
    if args.dataset in {'arvaniti', 'both'}:
        print(json.dumps(freeze_arvaniti(), indent=2))
    if args.dataset in {'lizard', 'both'}:
        print(json.dumps(freeze_lizard(), indent=2))
if __name__ == '__main__':
    main()
