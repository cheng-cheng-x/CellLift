from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
import csv
from celllift.runtime import json
import re
import numpy as np
from PIL import Image
from .constants import ARVANITI_BOARD_PREFIX, ARVANITI_DATA, ARVANITI_RAW, LIZARD_CLASS_MAP, LIZARD_DATA, LIZARD_PAPER_NUCLEI, LIZARD_RAW, MASK_VALUE_MAP, PALETTE_RGB
from .io_utils import atomic_json, sha256_file, skip_apple, unique_values, write_parquet

def _list_images(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    files = []
    for path in root.rglob('*'):
        if not path.is_file() or skip_apple(path):
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files)

def _board_of(stem: str) -> str | None:
    for board, prefix in ARVANITI_BOARD_PREFIX.items():
        if stem.startswith(prefix + '_') or stem == prefix:
            return board
    match = re.match('(ZT\\d+)', stem)
    return match.group(1) if match else None

def _decode_mask(path: Path) -> np.ndarray:
    image = np.array(Image.open(path))
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3]
        codes = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for color, value in PALETTE_RGB.items():
            codes[np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)] = value
        if int((codes == 255).sum()) > 0:
            unique = {tuple((int(v) for v in pixel)) for pixel in rgb.reshape(-1, 3)[codes.reshape(-1) == 255][:32]}
            raise RuntimeError(f'unmapped mask colors in {path}: {sorted(unique)[:12]}')
        return codes
    raise RuntimeError(f'unsupported mask rank {image.shape} at {path}')

def audit_arvaniti(raw_root: Path | None=None, out_root: Path | None=None) -> dict:
    raw_root = Path(raw_root or ARVANITI_RAW)
    out_root = Path(out_root or ARVANITI_DATA) / '00_manifest' / 'corrected_set_encoding'
    extracted = raw_root / 'extracted'
    rgb_files = [p for p in _list_images(extracted, ('.jpg', '.jpeg', '.png', '.tif', '.tiff')) if 'mask' not in p.name.lower()]
    mask_files = [p for p in _list_images(extracted, ('.png', '.tif', '.tiff')) if 'mask' in p.name.lower()]
    apple_skipped = sum((1 for p in extracted.rglob('*') if p.is_file() and skip_apple(p)))
    rgb_rows = []
    for path in rgb_files:
        stem = path.stem
        with Image.open(path) as image:
            width, height = image.size
        rgb_rows.append({'core_id': stem, 'board': _board_of(stem) or '', 'path': str(path), 'width': int(width), 'height': int(height), 'sha256': sha256_file(path), 'relpath': str(path.relative_to(extracted))})
    mask_rows = []
    pixel_counter: Counter[int] = Counter()
    for path in mask_files:
        name = path.name
        reader = 'pathologist1' if name.startswith('mask1_') else 'pathologist2' if name.startswith('mask2_') else 'train'
        stem = name
        for prefix in ('mask1_', 'mask2_', 'mask_'):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        stem = Path(stem).stem
        with Image.open(path) as image:
            width, height = image.size
            hist = image.histogram()
            values = [index for index, count in enumerate(hist[:256]) if count]
            for index, count in enumerate(hist[:256]):
                if count:
                    pixel_counter[index] += int(count)
        mask_rows.append({'core_id': stem, 'reader': reader, 'path': str(path), 'unique_values': json.dumps(values), 'height': int(height), 'width': int(width), 'sha256': sha256_file(path), 'relpath': str(path.relative_to(extracted))})
    rgb_by_id = {row['core_id']: row for row in rgb_rows}
    train_masks = {row['core_id']: row for row in mask_rows if row['reader'] == 'train'}
    test_m1 = {row['core_id']: row for row in mask_rows if row['reader'] == 'pathologist1'}
    test_m2 = {row['core_id']: row for row in mask_rows if row['reader'] == 'pathologist2'}
    boards = Counter((row['board'] for row in rgb_rows))
    missing_train = sorted(set(rgb_by_id) - set(train_masks) - set(test_m1))
    missing_test_pair = sorted(set(test_m1) ^ set(test_m2))
    unexpected_values = sorted((value for value in pixel_counter if value not in {0, 1, 2, 3, 4}))
    patient_ids = sorted({row['core_id'].rsplit('_', 2)[0] if row['core_id'].count('_') >= 3 else row['core_id'] for row in rgb_rows})
    board_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in rgb_rows:
        key = row['core_id']
        board_by_patient[key].add(row['board'])
    cross_board = {pid: sorted(boards_) for pid, boards_ in board_by_patient.items() if len(boards_) > 1}
    payload = {'status': 'PASS' if not unexpected_values else 'BLOCKER', 'rgb_count': len(rgb_rows), 'mask_count': len(mask_rows), 'train_mask_count': len(train_masks), 'test_pathologist1_count': len(test_m1), 'test_pathologist2_count': len(test_m2), 'apple_double_skip_count': apple_skipped, 'board_counts': dict(boards), 'unique_mask_values': sorted(pixel_counter), 'mask_value_hist': {str(k): int(v) for k, v in sorted(pixel_counter.items())}, 'mask_value_map': MASK_VALUE_MAP, 'missing_train_or_test_mask': missing_train[:50], 'unpaired_test_masks': missing_test_pair, 'unexpected_mask_values': unexpected_values, 'cross_board_core_ids': cross_board, 'patient_id_status': 'core_id_equals_filename_stem_no_donor_table', 'expected': {'rgb': 886, 'train_masks': 641, 'test_each': 245}}
    if payload['rgb_count'] != 886 or payload['train_mask_count'] != 641:
        payload['status'] = 'BLOCKER'
        payload['blocker'] = 'count_mismatch'
    if cross_board:
        payload['status'] = 'BLOCKER'
        payload['blocker'] = 'cross_board_patient_id'
    out_root.mkdir(parents=True, exist_ok=True)
    write_parquet(out_root / 'rgb.parquet', rgb_rows)
    write_parquet(out_root / 'masks.parquet', mask_rows)
    atomic_json(out_root / 'inventory.json', payload)
    atomic_json(out_root / 'mask_value_map.json', MASK_VALUE_MAP)
    atomic_json(out_root / 'pixel_hist.json', payload['mask_value_hist'])
    return payload

def _lizard_info(extracted: Path) -> list[dict]:
    info_path = next(extracted.rglob('info.csv'))
    rows = []
    with info_path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key.strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()})
    return rows

def _load_mat_ids(path: Path) -> dict:
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    inst = np.asarray(mat['inst_map'])
    nuc_ids = np.asarray(mat['id']).reshape(-1)
    classes = np.asarray(mat['class']).reshape(-1)
    centroids = np.asarray(mat['centroid'])
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)
    bboxes = np.asarray(mat['bbox'])
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(1, -1)
    return {'inst_map': inst.astype(np.int32), 'id': nuc_ids.astype(np.int32), 'class': classes.astype(np.int32), 'centroid': centroids.astype(np.float64), 'bbox': bboxes.astype(np.float64)}

def _filename_of(info: dict) -> str:
    for key in ('Filename', 'filename', 'Image', 'image', 'Name', 'name', 'file'):
        if info.get(key):
            return str(info[key]).strip()
    return str(next(iter(info.values()), '')).strip()

def _source_of(info: dict, stem: str) -> str:
    lowered = stem.lower()
    for name in ('consep', 'crag', 'dpath', 'glas', 'pannuke', 'tcga'):
        if lowered.startswith(name):
            return name
    for key in ('Source', 'source', 'Cohort', 'cohort'):
        if info.get(key):
            return str(info[key]).strip()
    return ''

def _normalize_source_text(text: str) -> str:
    value = str(text or '').strip()
    return re.sub('_1$', '', value)

def _group_key(filename: str, source: str, coord: str) -> tuple[str, str]:
    stem = Path(filename).stem
    cohort = source.lower()
    raw = str(coord or '').strip()
    if cohort == 'dpath' or stem.lower().startswith('dpath') or '-lv1-' in raw.lower():
        if '-lv1-' in raw:
            scan = raw.split('-lv1-')[0]
            return ('dpath_scan', f'dpath|{scan}')
        norm = _normalize_source_text(raw)
        if norm:
            return ('dpath_coord', f'dpath|{norm}')
        return ('unknown_patient', f'dpath|unknown|{stem}')
    norm = _normalize_source_text(raw)
    if norm and norm.lower() not in {'none', 'nan'}:
        return ('source_slide', f'{cohort}|{norm}')
    return ('unknown_patient', f'{cohort}|unknown|{stem}')

def _split_of(info: dict) -> str:
    for key in ('Split', 'split', 'Fold', 'fold'):
        if info.get(key) not in (None, ''):
            text = str(info[key]).strip()
            if text.endswith('.0'):
                text = text[:-2]
            return text
    return ''

def _coord_key(row: dict) -> str:
    fields = []
    for key in row:
        lowered = key.lower()
        if 'coord' in lowered or lowered in {'x', 'y', 'row', 'col'}:
            fields.append(f'{key}={row[key]}')
    return '|'.join(fields) if fields else ''

def audit_lizard(raw_root: Path | None=None, out_root: Path | None=None, *, revision: str='conditional_geometry') -> dict:
    raw_root = Path(raw_root or LIZARD_RAW)
    out_root = Path(out_root or LIZARD_DATA) / '00_manifest' / ('corrected_conditional_geometry' if revision == 'conditional_geometry' else 'corrected_set_encoding')
    extracted = raw_root / 'extracted'
    pngs = [p for p in _list_images(extracted, ('.png',)) if 'overlay' not in p.name.lower()]
    mats = _list_images(extracted, ('.mat',))
    png_by_stem = {p.stem: p for p in pngs}
    mat_by_stem = {p.stem: p for p in mats}
    info_rows = _lizard_info(extracted)
    if not info_rows:
        raise RuntimeError('Lizard info.csv is empty')
    class_hist: Counter[int] = Counter()
    nuclei_total = 0
    roi_rows = []
    pixel_hashes: dict[str, list[str]] = defaultdict(list)
    class_set = set()
    for info in info_rows:
        filename = _filename_of(info)
        stem = Path(filename).stem or filename
        png = png_by_stem.get(stem)
        mat = mat_by_stem.get(stem)
        if png is None or mat is None:
            roi_rows.append({'roi_id': stem, 'status': 'MISSING_FILE', 'png': str(png) if png else '', 'mat': str(mat) if mat else '', **{f'info_{k}': v for k, v in info.items()}})
            continue
        payload = _load_mat_ids(mat)
        n_nuc = int(payload['id'].size)
        nuclei_total += n_nuc
        for value in payload['class'].tolist():
            class_hist[int(value)] += 1
            class_set.add(int(value))
        with Image.open(png) as image:
            width, height = image.size
        pixel_hashes[sha256_file(png)].append(stem)
        source = _source_of(info, stem)
        split = _split_of(info)
        coord = str(info.get('Source') or info.get('source') or _coord_key(info))
        group_kind, group_id = _group_key(stem, source, coord)
        roi_rows.append({'roi_id': stem, 'status': 'OK', 'png': str(png), 'mat': str(mat), 'width': width, 'height': height, 'n_nuclei': n_nuc, 'source': source, 'split': split, 'coord_key': coord, 'group_kind': group_kind, 'group_seed': group_id, 'patient_status': 'unknown' if group_kind == 'unknown_patient' else 'slide_from_source', 'class_hist': json.dumps({str(k): int(v) for k, v in Counter(payload['class'].tolist()).items()}), 'inst_max': int(payload['inst_map'].max()) if payload['inst_map'].size else 0, 'png_sha256': sha256_file(png), 'mat_sha256': sha256_file(mat), **{f'info_{k}': v for k, v in info.items()}})
    seen_stems = {row['roi_id'] for row in roi_rows}
    for stem, png in png_by_stem.items():
        if stem in seen_stems:
            continue
        mat = mat_by_stem.get(stem)
        if mat is None:
            roi_rows.append({'roi_id': stem, 'status': 'MISSING_FILE', 'png': str(png), 'mat': ''})
            continue
        payload = _load_mat_ids(mat)
        n_nuc = int(payload['id'].size)
        nuclei_total += n_nuc
        for value in payload['class'].tolist():
            class_hist[int(value)] += 1
            class_set.add(int(value))
        with Image.open(png) as image:
            width, height = image.size
        pixel_hashes[sha256_file(png)].append(stem)
        source = _source_of({}, stem)
        coord = ''
        group_kind, group_id = _group_key(stem, source, coord)
        roi_rows.append({'roi_id': stem, 'status': 'OK', 'png': str(png), 'mat': str(mat), 'width': width, 'height': height, 'n_nuclei': n_nuc, 'source': source, 'split': '', 'coord_key': coord, 'group_kind': group_kind, 'group_seed': group_id, 'class_hist': json.dumps({str(k): int(v) for k, v in Counter(payload['class'].tolist()).items()}), 'inst_max': int(payload['inst_map'].max()) if payload['inst_map'].size else 0, 'png_sha256': sha256_file(png), 'mat_sha256': sha256_file(mat)})
    if class_set - set(LIZARD_CLASS_MAP):
        raise RuntimeError(f'Lizard class ids {sorted(class_set)} disagree with locked map {LIZARD_CLASS_MAP}')
    if 1 in class_set and 5 in class_set:
        pass
    adjacency: dict[str, set[str]] = defaultdict(set)
    by_hash = {key: values for key, values in pixel_hashes.items() if len(values) > 1}
    by_group_seed: dict[str, list[str]] = defaultdict(list)
    for row in roi_rows:
        if row['status'] != 'OK':
            continue
        by_group_seed[row['group_seed']].append(row['roi_id'])
    for members in by_hash.values():
        for left in members:
            for right in members:
                adjacency[left].add(right)
    for members in by_group_seed.values():
        if len(members) < 2:
            continue
        for left in members:
            for right in members:
                adjacency[left].add(right)
    seen: set[str] = set()
    groups = []
    for roi_id in sorted(adjacency):
        if roi_id in seen:
            continue
        stack = [roi_id]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency.get(node, ()))
        groups.append(sorted(component))
    for row in roi_rows:
        if row['roi_id'] in seen:
            continue
        groups.append([row['roi_id']])
        seen.add(row['roi_id'])
    group_of = {roi: f'g{index:03d}' for index, members in enumerate(sorted(groups, key=lambda xs: xs[0])) for roi in members}
    for row in roi_rows:
        row['group_id'] = group_of.get(row['roi_id'], '')
    png_count = len(png_by_stem)
    mat_count = len(mat_by_stem)
    payload = {'status': 'PASS' if png_count == 238 and nuclei_total == 431913 else 'BLOCKER', 'archive_status': 'UNOFFICIAL_MIRROR', 'dev_subset_238_pass': png_count == 238 and mat_count == 238 and (nuclei_total == 431913), 'full_291_incomplete_no_tcga': True, 'png_count': png_count, 'mat_count': mat_count, 'info_count': len(info_rows), 'nuclei_total': nuclei_total, 'class_hist': {str(k): int(v) for k, v in sorted(class_hist.items())}, 'class_map': LIZARD_CLASS_MAP, 'source_counts': dict(Counter((row.get('source', '') for row in roi_rows))), 'split_counts': dict(Counter((row.get('split', '') for row in roi_rows))), 'duplicate_pixel_groups': by_hash, 'n_groups': len(groups), 'group_size_hist': dict(Counter((len(g) for g in groups))), 'paper_nuclei_reference': LIZARD_PAPER_NUCLEI, 'readme_class_note': 'README listed both 1 and 5 as neutrophil; locked map is paper 1 neutrophil / 5 eosinophil'}
    out_root.mkdir(parents=True, exist_ok=True)
    write_parquet(out_root / 'rois.parquet', roi_rows)
    atomic_json(out_root / 'inventory.json', payload)
    atomic_json(out_root / 'class_map.json', {str(k): v for k, v in LIZARD_CLASS_MAP.items()})
    atomic_json(out_root / 'groups.json', {'groups': [{'group_id': group_of[g[0]], 'members': g} for g in groups]})
    atomic_json(out_root / 'scope.status.json', {'archive_pass': True, 'dev_subset_238_pass': payload['dev_subset_238_pass'], 'full_291_incomplete_no_tcga': True, 'zip_status': 'UNOFFICIAL_MIRROR', 'png_count': png_count, 'nuclei_total': nuclei_total})
    return payload

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['arvaniti', 'lizard', 'both'])
    args = parser.parse_args()
    if args.dataset in {'arvaniti', 'both'}:
        print(json.dumps(audit_arvaniti(), indent=2))
    if args.dataset in {'lizard', 'both'}:
        print(json.dumps(audit_lizard(), indent=2))
if __name__ == '__main__':
    main()
