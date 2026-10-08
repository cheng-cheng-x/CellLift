from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Callable, Mapping, Sequence
try:
    from .common import BracsDataError, EXPECTED_ROI_COUNT, EXPECTED_ROI_SPLIT_COUNTS, EXPECTED_WSI_SPLIT_COUNTS, T3_LABELS, T7_LABELS, atomic_write_json
except ImportError:
    from celllift.breast_roi_baseline.common import BracsDataError, EXPECTED_ROI_COUNT, EXPECTED_ROI_SPLIT_COUNTS, EXPECTED_WSI_SPLIT_COUNTS, T3_LABELS, T7_LABELS, atomic_write_json
OPENSLIDE_MPP_X_KEYS = ('openslide.mpp-x', 'mpp-x', 'mpp_x')
OPENSLIDE_MPP_Y_KEYS = ('openslide.mpp-y', 'mpp-y', 'mpp_y')
APERIO_MPP_KEYS = ('aperio.MPP', 'aperio.mpp')
_SPLITS = {'train', 'val', 'test'}
_SPLIT_ALIASES = {'train': 'train', 'training': 'train', 'tr': 'train', 'val': 'val', 'valid': 'val', 'validation': 'val', 'test': 'test', 'testing': 'test', 'te': 'test'}
_TRUE_VALUES = {'1', 'true', 'yes', 'y'}

@dataclass(frozen=True)
class WsiRecord:
    wsi_id: str
    split: str
    split_orig: str
    group: str
    type_name: str
    label_7: str
    label_3: str
    source_path: Path
    link_path: Path | None
    image_path: Path
    discarded_official_val: bool

@dataclass(frozen=True)
class RoiRecord:
    roi_id: str
    file_name: str
    split: str
    split_orig: str
    class_dir: str
    type_name: str
    label_7: str
    label_3: str
    wsi_id: str
    width: int
    height: int
    source_path: Path
    link_path: Path | None
    image_path: Path

@dataclass(frozen=True)
class BracsManifest:
    wsis: tuple[WsiRecord, ...]
    rois: tuple[RoiRecord, ...]

    @property
    def wsi_by_id(self) -> dict[str, WsiRecord]:
        return {record.wsi_id: record for record in self.wsis}

    @property
    def roi_by_id(self) -> dict[str, RoiRecord]:
        return {record.roi_id: record for record in self.rois}

@dataclass(frozen=True)
class SlideMpp:
    mpp_x: float
    mpp_y: float
    mpp: float
    source: str
    aperio_mpp: float | None

def _clean(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, '')
    return '' if value is None else str(value).strip()

def _required(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = _clean(row, key)
    if not value:
        raise BracsDataError(f'{context}: required column {key!r} is empty')
    return value

def normalize_split(value: str) -> str:
    normalized = _SPLIT_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise BracsDataError(f'unsupported BRACS split {value!r}')
    return normalized

def normalize_label(value: str, labels: Sequence[str], *, context: str) -> str:
    raw = str(value).strip().upper()
    if raw in labels:
        return raw
    try:
        index = int(raw)
    except ValueError as error:
        raise BracsDataError(f'{context}: unsupported label {value!r}') from error
    by_id = dict(enumerate(labels))
    if index not in by_id:
        raise BracsDataError(f'{context}: unsupported label ID {index}')
    return by_id[index]

def _path(value: str, *, base: Path) -> Path:
    if not value:
        raise BracsDataError('empty image path')
    expanded = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded)
    if candidate.is_absolute() or expanded.startswith('/'):
        return candidate
    return (base / candidate).resolve()

def _optional_path(value: str, *, base: Path) -> Path | None:
    return _path(value, base=base) if value else None

def _select_image_path(source: Path, link: Path | None) -> Path:
    return link if link is not None else source

def _load_csv(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(set(required_columns) - columns)
        if missing:
            raise BracsDataError(f'{path}: missing columns {missing}')
        return [dict(row) for row in reader]

def load_wsi_manifest(path: str | Path) -> tuple[WsiRecord, ...]:
    manifest_path = Path(path)
    rows = _load_csv(manifest_path, ('wsi_id', 'split_new', 'src_path', 'link_path', 'discarded_official_val'))
    records: list[WsiRecord] = []
    for index, row in enumerate(rows, start=2):
        discarded = _clean(row, 'discarded_official_val').lower() in _TRUE_VALUES
        split_raw = _clean(row, 'split_new')
        if discarded or not split_raw:
            continue
        context = f'{manifest_path}:{index}'
        source = _path(_required(row, 'src_path', context=context), base=manifest_path.parent)
        link = _optional_path(_clean(row, 'link_path'), base=manifest_path.parent)
        records.append(WsiRecord(wsi_id=_required(row, 'wsi_id', context=context), split=normalize_split(split_raw), split_orig=_clean(row, 'split_orig'), group=_clean(row, 'group'), type_name=_clean(row, 'type_name'), label_7=_clean(row, 'label_7'), label_3=_clean(row, 'label_3'), source_path=source, link_path=link, image_path=_select_image_path(source, link), discarded_official_val=discarded))
    return tuple(records)

def load_roi_manifest(path: str | Path) -> tuple[RoiRecord, ...]:
    manifest_path = Path(path)
    rows = _load_csv(manifest_path, ('roi_id', 'file_name', 'split_new', 'label_7', 'label_3', 'wsi_id', 'width', 'height', 'src_path', 'link_path'))
    records: list[RoiRecord] = []
    for index, row in enumerate(rows, start=2):
        split_raw = _clean(row, 'split_new')
        if not split_raw:
            continue
        context = f'{manifest_path}:{index}'
        try:
            width = int(_required(row, 'width', context=context))
            height = int(_required(row, 'height', context=context))
        except ValueError as error:
            raise BracsDataError(f'{context}: width/height must be integers') from error
        if width <= 0 or height <= 0:
            raise BracsDataError(f'{context}: width/height must be positive')
        source = _path(_required(row, 'src_path', context=context), base=manifest_path.parent)
        link = _optional_path(_clean(row, 'link_path'), base=manifest_path.parent)
        records.append(RoiRecord(roi_id=_required(row, 'roi_id', context=context), file_name=_required(row, 'file_name', context=context), split=normalize_split(split_raw), split_orig=_clean(row, 'split_orig'), class_dir=_clean(row, 'class_dir'), type_name=_clean(row, 'type_name'), label_7=normalize_label(_required(row, 'label_7', context=context), T7_LABELS, context=context), label_3=normalize_label(_required(row, 'label_3', context=context), T3_LABELS, context=context), wsi_id=_required(row, 'wsi_id', context=context), width=width, height=height, source_path=source, link_path=link, image_path=_select_image_path(source, link)))
    return tuple(records)

def load_bracs_manifest(wsi_manifest: str | Path, roi_manifest: str | Path) -> BracsManifest:
    return BracsManifest(wsis=load_wsi_manifest(wsi_manifest), rois=load_roi_manifest(roi_manifest))

def _duplicates(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted((key for key, count in counts.items() if count > 1))

def validate_bracs_manifest(manifest: BracsManifest, *, expected_roi_count: int | None=EXPECTED_ROI_COUNT, expected_roi_split_counts: Mapping[str, int] | None=EXPECTED_ROI_SPLIT_COUNTS, expected_wsi_split_counts: Mapping[str, int] | None=EXPECTED_WSI_SPLIT_COUNTS, check_paths: bool=False) -> dict[str, Any]:
    errors: list[str] = []
    duplicate_wsis = _duplicates([record.wsi_id for record in manifest.wsis])
    duplicate_rois = _duplicates([record.roi_id for record in manifest.rois])
    if duplicate_wsis:
        errors.append(f'duplicate wsi_id values: {duplicate_wsis[:10]}')
    if duplicate_rois:
        errors.append(f'duplicate roi_id values: {duplicate_rois[:10]}')
    wsi_by_id = manifest.wsi_by_id
    missing_parent = sorted({record.wsi_id for record in manifest.rois if record.wsi_id not in wsi_by_id})
    if missing_parent:
        errors.append(f'ROI rows reference missing WSI IDs: {missing_parent[:10]}')
    referenced_wsis = {record.wsi_id for record in manifest.rois}
    unreferenced_wsis = sorted(set(wsi_by_id) - referenced_wsis)
    wsi_splits: dict[str, set[str]] = defaultdict(set)
    for record in manifest.wsis:
        wsi_splits[record.wsi_id].add(record.split)
    for record in manifest.rois:
        wsi_splits[record.wsi_id].add(record.split)
        parent = wsi_by_id.get(record.wsi_id)
        if parent is not None and parent.split != record.split:
            errors.append(f'ROI {record.roi_id} split {record.split} != WSI {record.wsi_id} split {parent.split}')
    split_leaks = sorted((wsi_id for wsi_id, splits in wsi_splits.items() if len(splits) > 1))
    if split_leaks:
        errors.append(f'WSI IDs cross logical splits: {split_leaks[:10]}')
    roi_split_counts = Counter((record.split for record in manifest.rois))
    wsi_split_counts_all = Counter((record.split for record in manifest.wsis))
    wsi_split_counts = Counter((record.split for record in manifest.wsis if record.wsi_id in referenced_wsis))
    if expected_roi_count is not None and len(manifest.rois) != expected_roi_count:
        errors.append(f'expected {expected_roi_count} ROI rows, found {len(manifest.rois)}')
    if expected_roi_split_counts is not None and dict(roi_split_counts) != dict(expected_roi_split_counts):
        errors.append(f'ROI split counts {dict(roi_split_counts)} != {dict(expected_roi_split_counts)}')
    if expected_wsi_split_counts is not None and dict(wsi_split_counts) != dict(expected_wsi_split_counts):
        errors.append(f'WSI split counts {dict(wsi_split_counts)} != {dict(expected_wsi_split_counts)}')
    invalid_t7 = sorted({record.label_7 for record in manifest.rois} - set(T7_LABELS))
    invalid_t3 = sorted({record.label_3 for record in manifest.rois} - set(T3_LABELS))
    if invalid_t7:
        errors.append(f'unsupported T7 labels: {invalid_t7}')
    if invalid_t3:
        errors.append(f'unsupported T3 labels: {invalid_t3}')
    missing_paths: list[str] = []
    if check_paths:
        missing_paths.extend((str(record.image_path) for record in manifest.wsis if not record.image_path.is_file()))
        missing_paths.extend((str(record.image_path) for record in manifest.rois if not record.image_path.is_file()))
        if missing_paths:
            errors.append(f'missing image paths: {missing_paths[:10]}')
    if errors:
        raise BracsDataError('BRACS preflight failed:\n- ' + '\n- '.join(errors))
    return {'status': 'PASS', 'roi_count': len(manifest.rois), 'wsi_count': len(referenced_wsis), 'wsi_inventory_count': len(manifest.wsis), 'unreferenced_wsi_count': len(unreferenced_wsis), 'unreferenced_wsi_ids': unreferenced_wsis, 'roi_split_counts': dict(sorted(roi_split_counts.items())), 'wsi_split_counts': dict(sorted(wsi_split_counts.items())), 'wsi_inventory_split_counts': dict(sorted(wsi_split_counts_all.items())), 'wsi_disjoint_splits': True, 't7_counts': dict(sorted(Counter((record.label_7 for record in manifest.rois)).items())), 't3_counts': dict(sorted(Counter((record.label_3 for record in manifest.rois)).items())), 'paths_checked': bool(check_paths)}

def _first_float(properties: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        raw = properties.get(key)
        if raw is None or str(raw).strip() == '':
            continue
        try:
            value = float(str(raw).strip())
        except ValueError as error:
            raise BracsDataError(f'slide property {key}={raw!r} is not numeric') from error
        if not math.isfinite(value) or value <= 0:
            raise BracsDataError(f'slide property {key}={raw!r} must be finite and positive')
        return value
    return None

def extract_slide_mpp(properties: Mapping[str, Any], *, xy_relative_tolerance: float=0.001, aperio_relative_tolerance: float=0.01) -> SlideMpp:
    mpp_x = _first_float(properties, OPENSLIDE_MPP_X_KEYS)
    mpp_y = _first_float(properties, OPENSLIDE_MPP_Y_KEYS)
    aperio = _first_float(properties, APERIO_MPP_KEYS)
    if mpp_x is None and aperio is not None:
        mpp_x = aperio
    if mpp_y is None and aperio is not None:
        mpp_y = aperio
    if mpp_x is None or mpp_y is None:
        raise BracsDataError('missing MPP: require openslide.mpp-x/y or aperio.MPP; fixed-MPP fallback is forbidden')
    mean = (mpp_x + mpp_y) / 2.0
    if abs(mpp_x - mpp_y) / mean > xy_relative_tolerance:
        raise BracsDataError(f'anisotropic MPP is outside tolerance: x={mpp_x}, y={mpp_y}')
    if aperio is not None and abs(mean - aperio) / mean > aperio_relative_tolerance:
        raise BracsDataError(f'OpenSlide MPP ({mean}) disagrees with aperio.MPP ({aperio})')
    source = 'openslide.mpp-x/y' if any((properties.get(key) not in (None, '') for key in OPENSLIDE_MPP_X_KEYS)) else 'aperio.MPP'
    return SlideMpp(mpp_x=mpp_x, mpp_y=mpp_y, mpp=mean, source=source, aperio_mpp=aperio)

def read_slide_mpp(slide_path: str | Path, *, properties: Mapping[str, Any] | None=None, slide_factory: Callable[[str], Any] | None=None, xy_relative_tolerance: float=0.001, aperio_relative_tolerance: float=0.01) -> SlideMpp:
    if properties is not None:
        return extract_slide_mpp(properties, xy_relative_tolerance=xy_relative_tolerance, aperio_relative_tolerance=aperio_relative_tolerance)
    if slide_factory is None:
        try:
            import openslide
        except ImportError as error:
            raise BracsDataError('OpenSlide is required when properties are not injected') from error
        slide_factory = openslide.OpenSlide
    slide = slide_factory(str(slide_path))
    try:
        slide_properties = dict(slide.properties)
    finally:
        close = getattr(slide, 'close', None)
        if callable(close):
            close()
    return extract_slide_mpp(slide_properties, xy_relative_tolerance=xy_relative_tolerance, aperio_relative_tolerance=aperio_relative_tolerance)

def collect_wsi_mpp(wsis: Sequence[WsiRecord], *, properties_by_wsi: Mapping[str, Mapping[str, Any]] | None=None, slide_factory: Callable[[str], Any] | None=None, xy_relative_tolerance: float=0.001, aperio_relative_tolerance: float=0.01) -> dict[str, SlideMpp]:
    result: dict[str, SlideMpp] = {}
    for record in wsis:
        injected = None if properties_by_wsi is None else properties_by_wsi.get(record.wsi_id)
        if properties_by_wsi is not None and injected is None:
            raise BracsDataError(f'no injected slide properties for WSI {record.wsi_id}')
        result[record.wsi_id] = read_slide_mpp(record.image_path, properties=injected, slide_factory=slide_factory, xy_relative_tolerance=xy_relative_tolerance, aperio_relative_tolerance=aperio_relative_tolerance)
    return result

def preflight_bracs_data(wsi_manifest: str | Path, roi_manifest: str | Path, *, expected_roi_count: int | None=EXPECTED_ROI_COUNT, expected_roi_split_counts: Mapping[str, int] | None=EXPECTED_ROI_SPLIT_COUNTS, expected_wsi_split_counts: Mapping[str, int] | None=EXPECTED_WSI_SPLIT_COUNTS, check_paths: bool=True, read_mpp: bool=True, properties_by_wsi: Mapping[str, Mapping[str, Any]] | None=None, slide_factory: Callable[[str], Any] | None=None, xy_relative_tolerance: float=0.001, aperio_relative_tolerance: float=0.01, output: str | Path | None=None) -> dict[str, Any]:
    manifest = load_bracs_manifest(wsi_manifest, roi_manifest)
    report = validate_bracs_manifest(manifest, expected_roi_count=expected_roi_count, expected_roi_split_counts=expected_roi_split_counts, expected_wsi_split_counts=expected_wsi_split_counts, check_paths=check_paths)
    if read_mpp:
        referenced = {record.wsi_id for record in manifest.rois}
        mpp_by_wsi = collect_wsi_mpp([record for record in manifest.wsis if record.wsi_id in referenced], properties_by_wsi=properties_by_wsi, slide_factory=slide_factory, xy_relative_tolerance=xy_relative_tolerance, aperio_relative_tolerance=aperio_relative_tolerance)
        values = [entry.mpp for entry in mpp_by_wsi.values()]
        report['mpp'] = {'wsi_count': len(values), 'minimum': min(values), 'maximum': max(values), 'mean': sum(values) / len(values), 'records': {key: asdict(value) for key, value in sorted(mpp_by_wsi.items())}}
    if output is not None:
        atomic_write_json(output, report)
    return report
