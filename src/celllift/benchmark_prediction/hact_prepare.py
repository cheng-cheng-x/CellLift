from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from .paths import REMOTE_DATA, comparison_result_root, split_root
OURS_ID_TO_NAME = {0: 'N', 1: 'PB', 2: 'UDH', 3: 'FEA', 4: 'ADH', 5: 'DCIS', 6: 'IC'}
HACT_NAMES = {'N', 'PB', 'UDH', 'ADH', 'FEA', 'DCIS', 'IC'}
HACT_ROOT = Path(REMOTE_DATA) / 'hact_official_570'
IMAGE_ROOT = HACT_ROOT / 'images'
MANIFEST_CSV = Path(_resource_path('artifact_0005'))

def _code(row: dict, extra: dict | None=None) -> str | None:
    for raw in (row.get('label'), row.get('label_name'), (extra or {}).get('type_name'), (extra or {}).get('label')):
        name = str(raw or '').strip().upper()
        if name in HACT_NAMES:
            return name
    lid = row.get('label_id')
    if lid not in (None, ''):
        return OURS_ID_TO_NAME.get(int(lid))
    return None

def _roi_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not MANIFEST_CSV.is_file():
        return out
    with MANIFEST_CSV.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rec = {'type_name': row.get('type_name') or row.get('label_7'), 'paths': [p for p in (row.get('link_path'), row.get('src_path')) if p]}
            for key in (row.get('roi_id'), Path(str(row.get('file_name') or '')).stem):
                if key:
                    out[str(key)] = rec
    return out

def run() -> dict:
    lookup = _roi_manifest()
    by_id: dict[str, dict] = {}
    for split, name in (('fit', 'train_rows.json'), ('val', 'val_rows.json'), ('test', 'test_rows.json')):
        folder = 'train' if split == 'fit' else split
        path = split_root() / 'bracs' / name
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding='utf-8')):
            sid = str(row['sample_id'])
            extra = lookup.get(sid) or lookup.get(str(row.get('roi_id') or ''))
            paths = list((extra or {}).get('paths') or [])
            paths.extend((p for p in row.get('rgb_paths') or [] if p))
            by_id[sid] = {'paths': paths, 'code': _code(row, extra), 'label_id': None if row.get('label_id') in (None, '') else int(row['label_id']), 'split': folder, 'source': 'organized_roi' if extra else 'breast_roi_tile'}
    counts = {'train': 0, 'val': 0, 'test': 0, 'missing_path': 0, 'missing_label': 0, 'organized_roi': 0, 'breast_roi_tile': 0}
    manifest = []
    for split in ('train', 'val', 'test'):
        dest = IMAGE_ROOT / split
        dest.mkdir(parents=True, exist_ok=True)
        for leftover in dest.glob('*'):
            leftover.unlink()
    index = 0
    for sid, rec in sorted(by_id.items()):
        if rec['code'] is None:
            counts['missing_label'] += 1
            continue
        existing = [p for p in rec['paths'] if Path(p).is_file()]
        if not existing:
            counts['missing_path'] += 1
            continue
        index += 1
        name = f"{index:05d}_roi_{rec['code']}.png"
        target = IMAGE_ROOT / rec['split'] / name
        src = Path(existing[0])
        if target.is_symlink() or target.is_file():
            target.unlink()
        target.symlink_to(src)
        counts[rec['split']] += 1
        src_kind = 'organized_roi' if rec['source'] == 'organized_roi' and src == Path(rec['paths'][0]) else 'breast_roi_tile'
        counts[src_kind] += 1
        manifest.append({'sample_id': sid, 'split': rec['split'], 'label_id': rec['label_id'], 'hact_code': rec['code'], 'hact_name': name, 'src': str(src), 'src_kind': src_kind})
    out = comparison_result_root() / 'analysis' / 'hact_prepare.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'counts': counts, 'expected': {'train': 3163, 'val': 494, 'test': 570}, 'image_root': str(IMAGE_ROOT), 'n_mapped': len(manifest)}
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    (HACT_ROOT / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    payload['written'] = str(out)
    payload['test_complete'] = counts['test'] == 570
    return payload
if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
