from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
import math
import stat
from concurrent.futures import ThreadPoolExecutor
from celllift.runtime import ResourcePath as Path
from typing import Any
import cv2
import numpy as np
import tifffile
from PIL import Image
from .core import result_root, run_logged, sha256_file, stage_status, valis_shell, write_json, write_tsv

def _fiducials(path: Path, observer: str) -> list[dict[str, Any]]:
    with path.open(newline='', encoding='utf-8', errors='replace') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if len(rows) != 259:
        raise RuntimeError(f'{path} has {len(rows)} data rows; expected 259')
    output = []
    for index, row in enumerate(rows, start=1):
        filename = str(row['Filename']).strip()
        expected = f'{index:03d}.tif'
        if filename != expected:
            raise RuntimeError(f'{path}: expected {expected}, got {filename}')
        coordinates = [value for key, value in row.items() if key and key != 'Filename' and value]
        if len(coordinates) != 16:
            raise RuntimeError(f'{path}: {filename} has {len(coordinates)} coordinates; expected 16')
        output.append({'observer': observer, 'left_section': f'{index:03d}', 'right_section': f'{index + 1:03d}', **{f'coordinate_{i + 1:02d}': float(value) for i, value in enumerate(coordinates)}})
    return output

def _probe(path: Path) -> tuple[int, int, int, int, str]:
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise RuntimeError(f'{path} is not a single-page TIFF')
        page = tif.pages[0]
        height, width = page.shape[:2]
        samples = int(page.samplesperpixel or (page.shape[2] if len(page.shape) == 3 else 1))
        bits = int(page.bitspersample)
        compression = str(page.compression.name)
    return (width, height, samples, bits, compression)

def write_fiducial_source_identity(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for observer in ('observer1', 'observer2'):
        path = Path(cfg[observer])
        if not path.is_file():
            raise RuntimeError(f'missing official fiducial source for {observer}: {path}')
        read_only = not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        if not read_only:
            raise RuntimeError(f'official fiducial source must be read-only: {path}')
        rows.append({'observer': observer, 'source_path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path), 'source_read_only': read_only, 'role': 'independent_final_audit_only'})
    write_tsv(root / '00_manifests/official_fiducial_source_identity.tsv', rows)
    return rows

def run_preflight(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    raw_root = Path(cfg['raw_root'])
    paths = sorted(raw_root.glob('*.tif'))
    expected = [raw_root / f'{index:03d}.tif' for index in range(1, 261)]
    if paths != expected:
        missing = [str(path) for path in expected if path not in paths]
        extra = [str(path) for path in paths if path not in expected]
        raise RuntimeError(f'prostate TIFF identity mismatch; missing={missing}, extra={extra}')
    physical = cfg['physical']
    probes = [_probe(path) for path in paths]
    expected_probe = (int(physical['width_px']), int(physical['height_px']), 3, 8, 'PACKBITS')
    bad = [(path.name, probe) for path, probe in zip(paths, probes) if probe != expected_probe]
    if bad:
        raise RuntimeError(f'unexpected TIFF metadata: {bad[:10]}')
    writable_sources = [path for path in paths if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)]
    if writable_sources:
        raise RuntimeError(f'source TIFFs must be filesystem read-only before preflight: {[str(path) for path in writable_sources[:10]]}')
    workers = min(int(cfg['runtime']['cpu_qc_workers']), 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(sha256_file, paths))
    slide_rows = []
    for index, (path, digest, probe) in enumerate(zip(paths, hashes, probes), start=1):
        slide_rows.append({'section_id': f'{index:03d}', 'sequence_position': index, 'source_path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest, 'width': probe[0], 'height': probe[1], 'channels': probe[2], 'bits': probe[3], 'compression': probe[4], 'mpp_x': physical['mpp_um_per_px'], 'mpp_y': physical['mpp_um_per_px'], 'z_um_numeric': (index - int(cfg['global']['reference_section'])) * float(physical['section_spacing_um']), 'is_global_reference': index == int(cfg['global']['reference_section']), 'source_read_only': not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))})
    manifest = root / '00_manifests/slide_identity.tsv'
    write_tsv(manifest, slide_rows)
    fiducial_source_rows = write_fiducial_source_identity(cfg, root)
    observer_rows = [*_fiducials(Path(cfg['observer1']), 'observer1'), *_fiducials(Path(cfg['observer2']), 'observer2')]
    write_tsv(root / '00_manifests/official_fiducials_audit_only.tsv', observer_rows)
    write_json(root / '01_preflight/preflight_summary.json', {'status': 'PASS', 'slide_count': len(slide_rows), 'adjacent_edge_count': 259, 'observer_rows': len(observer_rows), 'source_read_only': all((bool(row['source_read_only']) for row in slide_rows)), 'fiducial_source_count': len(fiducial_source_rows), 'fiducial_sources_read_only': all((bool(row['source_read_only']) for row in fiducial_source_rows)), 'official_fiducials_role': 'independent final audit only'})
    stage_status(cfg, 'preflight', 'PASS', slide_count=260, adjacent_edges=259)

def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tissue = ((hsv[..., 1] > 12) & (gray < 247) | (gray < 205)).astype(np.uint8)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(tissue, 8)
    output = np.zeros_like(tissue)
    min_area = max(16, int(tissue.size * 0.0002))
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == label] = 255
    return output

def run_thumbnail(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    manifest = root / '00_manifests/slide_identity.tsv'
    if not manifest.exists():
        raise RuntimeError('preflight must run before thumbnail')
    helper = Path(cfg['code_root']) / 'scripts/vips_batch.py'
    physical = cfg['physical']
    jobs = [(float(physical['global_mpp_um_per_px']), root / '02_thumbnails_order_qc/thumbnails_10um', 88), (float(physical['diagnostic_mpp_um_per_px']), root / '02_thumbnails_order_qc/thumbnails_3p68um', 90)]
    for target_mpp, output, quality in jobs:
        command = valis_shell(cfg, helper, ['thumbnails', '--manifest', str(manifest), '--output-dir', str(output), '--source-mpp', str(physical['mpp_um_per_px']), '--target-mpp', str(target_mpp), '--quality', str(quality)])
        run_logged(command, root / f'logs/thumbnail_{target_mpp:g}um.log')
    ten_root = root / '02_thumbnails_order_qc/thumbnails_10um'
    mask_root = root / '02_thumbnails_order_qc/masks_10um'
    mask_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(1, 261):
        section = f'{index:03d}'
        image = np.asarray(Image.open(ten_root / f'{section}.jpg').convert('RGB'))
        mask = tissue_mask(image)
        Image.fromarray(mask).save(mask_root / f'{section}.png', compress_level=3)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rows.append({'section_id': section, 'thumbnail_width': image.shape[1], 'thumbnail_height': image.shape[0], 'tissue_fraction': float((mask > 0).mean()), 'dark_fraction': float((gray < 35).mean()), 'mean_gray': float(gray.mean()), 'std_gray': float(gray.std())})
    write_tsv(root / '00_manifests/thumbnail_qc.tsv', rows)
    stage_status(cfg, 'thumbnail', 'PASS', thumbnail_count=260, mask_count=260)

def _histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    hist_left = cv2.calcHist([left], [0], None, [64], [0, 256])
    hist_right = cv2.calcHist([right], [0], None, [64], [0, 256])
    cv2.normalize(hist_left, hist_left)
    cv2.normalize(hist_right, hist_right)
    return float(cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL))

def run_order_qc(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    thumb_root = root / '02_thumbnails_order_qc/thumbnails_10um'
    mask_root = root / '02_thumbnails_order_qc/masks_10um'
    rows = []
    for index in range(1, 260):
        left_id, right_id = (f'{index:03d}', f'{index + 1:03d}')
        left = np.asarray(Image.open(thumb_root / f'{left_id}.jpg').convert('L'))
        right = np.asarray(Image.open(thumb_root / f'{right_id}.jpg').convert('L'))
        left_mask = np.asarray(Image.open(mask_root / f'{left_id}.png')) > 0
        right_mask = np.asarray(Image.open(mask_root / f'{right_id}.png')) > 0
        intersection = np.logical_and(left_mask, right_mask).sum()
        dice = float(2 * intersection / max(1, left_mask.sum() + right_mask.sum()))
        hist = _histogram_similarity(left, right)
        rows.append({'edge_index': index, 'left_id': left_id, 'right_id': right_id, 'prealigned_mask_dice': dice, 'histogram_similarity': hist, 'similarity': 0.7 * dice + 0.3 * max(0.0, hist), 'ordering_status': 'fixed_numeric_order'})
    write_tsv(root / '02_thumbnails_order_qc/adjacent_similarity.tsv', rows)
    write_tsv(root / '00_manifests/ordering_manifest.tsv', [{'section_id': f'{index:03d}', 'sequence_position': index, 'segment_id': 'segment_001', 'global_reference': '130' if index == 130 else ''} for index in range(1, 261)])
    stage_status(cfg, 'order_qc', 'PASS', edge_count=259, weakest_edge=min(rows, key=lambda row: row['similarity'])['edge_index'], reordered=False)
