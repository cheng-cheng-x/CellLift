from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import io
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping
from PIL import Image
try:
    from .common import BracsDataError, TARGET_MPP_UM_PER_PX, TILE_SIZE_PX, T3_LABELS, T7_LABELS, atomic_write_bytes, atomic_write_json, safe_component, sha256_bytes, sha256_file, stable_tile_id
    from .data import BracsManifest, RoiRecord, SlideMpp
except ImportError:
    from celllift.breast_roi_baseline.common import BracsDataError, TARGET_MPP_UM_PER_PX, TILE_SIZE_PX, T3_LABELS, T7_LABELS, atomic_write_bytes, atomic_write_json, safe_component, sha256_bytes, sha256_file, stable_tile_id
    from celllift.breast_roi_baseline.data import BracsManifest, RoiRecord, SlideMpp

@dataclass(frozen=True)
class TilePlan:
    roi_id: str
    tile_id: str
    row: int
    column: int
    target_x0: int
    target_y0: int
    target_x1: int
    target_y1: int
    native_x0: float
    native_y0: float
    native_x1: float
    native_y1: float
    valid_width: int
    valid_height: int
    valid_fraction: float
    pad_right: int
    pad_bottom: int
TILE_MANIFEST_FIELDS = ('roi_id', 'tile_id', 'graph_id', 'wsi_id', 'split_new', 'split', 'label_7', 'label_3', 'label_7_name', 'label_3_name', 'validation_fold', 'final_validation_fold', 'roi_path', 'tile_path', 'rgb_path', 'tile_sha256', 'native_mpp_x', 'native_mpp_y', 'native_mpp', 'mpp_source', 'target_mpp', 'resize_scale', 'native_width', 'native_height', 'target_width', 'target_height', 'tile_row', 'tile_column', 'target_x0', 'target_y0', 'target_x1', 'target_y1', 'native_x0', 'native_y0', 'native_x1', 'native_y1', 'valid_width', 'valid_height', 'valid_fraction', 'pad_right', 'pad_bottom')

def _arrow_tile_schema(pa: Any) -> Any:
    strings = {'roi_id', 'tile_id', 'graph_id', 'wsi_id', 'split_new', 'split', 'label_7_name', 'label_3_name', 'roi_path', 'tile_path', 'rgb_path', 'tile_sha256', 'mpp_source'}
    int8 = {'label_7', 'label_3', 'validation_fold', 'final_validation_fold'}
    integers = {'native_width', 'native_height', 'target_width', 'target_height', 'tile_row', 'tile_column', 'target_x0', 'target_y0', 'target_x1', 'target_y1', 'valid_width', 'valid_height', 'pad_right', 'pad_bottom'}
    fields = []
    for name in TILE_MANIFEST_FIELDS:
        if name in strings:
            dtype = pa.string()
        elif name in int8:
            dtype = pa.int8()
        elif name in integers:
            dtype = pa.int32()
        else:
            dtype = pa.float64()
        fields.append(pa.field(name, dtype, nullable=True))
    return pa.schema(fields)

def resize_scale(native_mpp: float, target_mpp: float=TARGET_MPP_UM_PER_PX) -> float:
    if not math.isfinite(native_mpp) or native_mpp <= 0:
        raise BracsDataError('native MPP must be finite and positive')
    if not math.isfinite(target_mpp) or target_mpp <= 0:
        raise BracsDataError('target MPP must be finite and positive')
    return native_mpp / target_mpp

def scaled_dimension(native_pixels: int, scale: float) -> int:
    if native_pixels <= 0 or not math.isfinite(scale) or scale <= 0:
        raise BracsDataError('native dimension and scale must be positive')
    return max(1, int(math.floor(native_pixels * scale + 0.5)))

def plan_roi_tiles(roi_id: str, native_width: int, native_height: int, native_mpp: float, *, target_mpp: float=TARGET_MPP_UM_PER_PX, tile_size: int=TILE_SIZE_PX) -> tuple[tuple[TilePlan, ...], int, int, float]:
    if tile_size <= 0:
        raise BracsDataError('tile_size must be positive')
    scale = resize_scale(native_mpp, target_mpp)
    target_width = scaled_dimension(native_width, scale)
    target_height = scaled_dimension(native_height, scale)
    rows = math.ceil(target_height / tile_size)
    columns = math.ceil(target_width / tile_size)
    plans: list[TilePlan] = []
    for row in range(rows):
        y0 = row * tile_size
        valid_height = min(tile_size, target_height - y0)
        for column in range(columns):
            x0 = column * tile_size
            valid_width = min(tile_size, target_width - x0)
            x1 = x0 + valid_width
            y1 = y0 + valid_height
            plans.append(TilePlan(roi_id=roi_id, tile_id=stable_tile_id(roi_id, row, column), row=row, column=column, target_x0=x0, target_y0=y0, target_x1=x1, target_y1=y1, native_x0=x0 / scale, native_y0=y0 / scale, native_x1=min(float(native_width), x1 / scale), native_y1=min(float(native_height), y1 / scale), valid_width=valid_width, valid_height=valid_height, valid_fraction=valid_width * valid_height / float(tile_size * tile_size), pad_right=tile_size - valid_width, pad_bottom=tile_size - valid_height))
    return (tuple(plans), target_width, target_height, scale)

def _rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode == 'RGB':
        return image
    if image.mode in {'RGBA', 'LA'} or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert('RGB')
    return image.convert('RGB')

def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', compress_level=1)
    return buffer.getvalue()

def _write_or_validate_tile(image: Image.Image, path: Path, *, overwrite: bool) -> str:
    if path.exists() and (not overwrite):
        with Image.open(path) as existing:
            if existing.size != image.size or existing.mode != 'RGB':
                raise BracsDataError(f'existing tile has incompatible shape/mode: {path}')
            if existing.tobytes() != image.tobytes():
                raise BracsDataError(f'existing tile content is stale: {path}')
        return sha256_file(path)
    encoded = _png_bytes(image)
    atomic_write_bytes(path, encoded)
    return sha256_bytes(encoded)

def prepare_roi_tiles(roi: RoiRecord, mpp: SlideMpp, output_root: str | Path, *, target_mpp: float=TARGET_MPP_UM_PER_PX, tile_size: int=TILE_SIZE_PX, overwrite: bool=False, validation_fold: int | None=None, final_validation_fold: int | None=None) -> list[dict[str, Any]]:
    plans, target_width, target_height, scale = plan_roi_tiles(roi.roi_id, roi.width, roi.height, mpp.mpp, target_mpp=target_mpp, tile_size=tile_size)
    with Image.open(roi.image_path) as opened:
        image = _rgb_on_white(opened)
        if image.size != (roi.width, roi.height):
            raise BracsDataError(f'ROI {roi.roi_id} manifest size {(roi.width, roi.height)} != image size {image.size}')
        resized = image.resize((target_width, target_height), resample=Image.Resampling.BOX)
    roi_directory = Path(output_root) / roi.split / safe_component(roi.roi_id)
    rows: list[dict[str, Any]] = []
    for plan in plans:
        crop = resized.crop((plan.target_x0, plan.target_y0, plan.target_x1, plan.target_y1))
        canvas = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
        canvas.paste(crop, (0, 0))
        tile_path = roi_directory / f'{plan.tile_id}.png'
        checksum = _write_or_validate_tile(canvas, tile_path, overwrite=overwrite)
        rows.append({'roi_id': roi.roi_id, 'tile_id': plan.tile_id, 'graph_id': plan.tile_id, 'wsi_id': roi.wsi_id, 'split_new': roi.split, 'split': roi.split, 'label_7': T7_LABELS.index(roi.label_7), 'label_3': T3_LABELS.index(roi.label_3), 'label_7_name': roi.type_name or roi.label_7, 'label_3_name': roi.label_3, 'validation_fold': validation_fold, 'final_validation_fold': final_validation_fold, 'roi_path': str(roi.image_path), 'tile_path': str(tile_path), 'rgb_path': str(tile_path), 'tile_sha256': checksum, 'native_mpp_x': mpp.mpp_x, 'native_mpp_y': mpp.mpp_y, 'native_mpp': mpp.mpp, 'mpp_source': mpp.source, 'target_mpp': target_mpp, 'resize_scale': scale, 'native_width': roi.width, 'native_height': roi.height, 'target_width': target_width, 'target_height': target_height, 'tile_row': plan.row, 'tile_column': plan.column, 'target_x0': plan.target_x0, 'target_y0': plan.target_y0, 'target_x1': plan.target_x1, 'target_y1': plan.target_y1, 'native_x0': plan.native_x0, 'native_y0': plan.native_y0, 'native_x1': plan.native_x1, 'native_y1': plan.native_y1, 'valid_width': plan.valid_width, 'valid_height': plan.valid_height, 'valid_fraction': plan.valid_fraction, 'pad_right': plan.pad_right, 'pad_bottom': plan.pad_bottom})
    return rows

def prepare_bracs_tiles(manifest: BracsManifest, mpp_by_wsi: Mapping[str, SlideMpp], output_root: str | Path, manifest_output: str | Path, *, target_mpp: float=TARGET_MPP_UM_PER_PX, tile_size: int=TILE_SIZE_PX, overwrite: bool=False, summary_output: str | Path | None=None, folds_by_roi: Mapping[str, Mapping[str, int | None]] | None=None, workers: int=1) -> dict[str, Any]:
    destination = Path(manifest_output)
    if destination.suffix.lower() not in {'.csv', '.parquet'}:
        raise BracsDataError('tile manifest must use .csv or .parquet')
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temporary = tempfile.mkstemp(prefix=f'.{destination.name}.tmp.', dir=str(destination.parent))
    os.close(handle)
    temporary = Path(raw_temporary)
    tile_count = 0
    split_counts: Counter[str] = Counter()
    roi_counts: Counter[str] = Counter()
    ordered_rois = sorted(manifest.rois, key=lambda item: (item.split, item.roi_id))

    def prepare_one(roi: RoiRecord) -> tuple[RoiRecord, list[dict[str, Any]]]:
        mpp = mpp_by_wsi.get(roi.wsi_id)
        if mpp is None:
            raise BracsDataError(f'missing MPP for WSI {roi.wsi_id}')
        folds = {} if folds_by_roi is None else folds_by_roi.get(roi.roi_id, {})
        return (roi, prepare_roi_tiles(roi, mpp, output_root, target_mpp=target_mpp, tile_size=tile_size, overwrite=overwrite, validation_fold=folds.get('validation_fold'), final_validation_fold=folds.get('final_validation_fold')))

    def prepared() -> Iterable[tuple[RoiRecord, list[dict[str, Any]]]]:
        if workers <= 1:
            yield from map(prepare_one, ordered_rois)
            return
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            yield from executor.map(prepare_one, ordered_rois)
    try:
        if destination.suffix.lower() == '.csv':
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(TILE_MANIFEST_FIELDS))
                writer.writeheader()
                for roi, rows in prepared():
                    writer.writerows(rows)
                    tile_count += len(rows)
                    split_counts[roi.split] += len(rows)
                    roi_counts[roi.split] += 1
                stream.flush()
                os.fsync(stream.fileno())
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as error:
                raise BracsDataError('pyarrow is required for a Parquet tile manifest') from error
            parquet_writer = None
            schema = _arrow_tile_schema(pa)
            try:
                for roi, rows in prepared():
                    table = pa.Table.from_pylist(rows, schema=schema)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(temporary, schema, compression='zstd')
                    parquet_writer.write_table(table)
                    tile_count += len(rows)
                    split_counts[roi.split] += len(rows)
                    roi_counts[roi.split] += 1
            finally:
                if parquet_writer is not None:
                    parquet_writer.close()
            if parquet_writer is None:
                raise BracsDataError('cannot publish an empty BRACS tile manifest')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {'status': 'PASS', 'roi_count': len(manifest.rois), 'tile_count': tile_count, 'roi_split_counts': dict(sorted(roi_counts.items())), 'tile_split_counts': dict(sorted(split_counts.items())), 'target_mpp': target_mpp, 'tile_size': tile_size, 'workers': int(workers), 'manifest': str(destination), 'manifest_sha256': sha256_file(destination)}
    if summary_output is not None:
        atomic_write_json(summary_output, summary)
    return summary
