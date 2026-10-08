from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping
import numpy as np
from PIL import Image, ImageDraw
from .common import atomic_write_json, atomic_write_tsv, parse_bool, parse_float, parse_int, read_json, read_tsv, roi_layer_id, sha256_file, stable_bucket

def is_accepted_registration_edge(row: Mapping[str, Any]) -> bool:
    final_status = str(row.get('final_status', '') or row.get('profile_status', '')).strip().lower()
    if final_status:
        return final_status == 'accepted'
    status = str(row.get('status', '')).strip().lower()
    if status in {'accepted', 'pass'}:
        return True
    return False

def _make_layer_maps(layer_rows: Iterable[Mapping[str, Any]]) -> tuple[dict[tuple[str, int], Mapping[str, Any]], dict[tuple[str, int], Mapping[str, Any]]]:
    by_opt: dict[tuple[str, int], Mapping[str, Any]] = {}
    by_src: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in layer_rows:
        section = parse_int(row.get('section_id'))
        by_opt[str(row.get('optimized_track_id', '')), section] = row
        by_src[str(row.get('source_track_id', row.get('optimized_track_id', ''))), section] = row
    return (by_opt, by_src)

def resolve_accepted_registration_edges(rsg_root: str | Path, layer_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_opt, by_src = _make_layer_maps(layer_rows)
    raw_edges = read_tsv(Path(rsg_root) / '00_manifests/edge_transform_manifest.tsv')
    accepted_edges = [row for row in raw_edges if is_accepted_registration_edge(row)]
    resolved: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_valid: set[tuple[str, int, int]] = set()
    seen_raw: set[tuple[str, int, int]] = set()
    duplicate_raw = 0
    duplicate_valid = 0
    non_adjacent = 0
    for raw in accepted_edges:
        track = str(raw.get('track_id') or raw.get('optimized_track_id') or '')
        left = parse_int(raw.get('left_id', raw.get('left_section_id')))
        right = parse_int(raw.get('right_id', raw.get('right_section_id')))
        raw_key = (track, left, right)
        raw_edge_id = raw.get('edge_id') or f'{track}_{left:03d}__{right:03d}'
        if raw_key in seen_raw:
            duplicate_raw += 1
            excluded.append({'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'exclusion_reason': 'duplicate_raw_accepted_edge', 'left_layer_resolved': '', 'right_layer_resolved': '', 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'final_status': raw.get('final_status', ''), 'profile_status': raw.get('profile_status', '')})
            continue
        seen_raw.add(raw_key)
        if abs(right - left) != 1:
            non_adjacent += 1
            excluded.append({'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'exclusion_reason': 'non_adjacent_raw_accepted_edge', 'left_layer_resolved': '', 'right_layer_resolved': '', 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'final_status': raw.get('final_status', ''), 'profile_status': raw.get('profile_status', '')})
            continue
        left_row = by_opt.get((track, left)) or by_src.get((track, left))
        right_row = by_opt.get((track, right)) or by_src.get((track, right))
        left_ok = left_row is not None
        right_ok = right_row is not None
        if not left_ok or not right_ok:
            excluded.append({'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'exclusion_reason': 'missing_delivered_registered_layer_endpoint', 'left_layer_resolved': bool(left_ok), 'right_layer_resolved': bool(right_ok), 'left_roi_layer_id': left_row.get('roi_layer_id', '') if left_row else '', 'right_roi_layer_id': right_row.get('roi_layer_id', '') if right_row else '', 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'final_status': raw.get('final_status', ''), 'profile_status': raw.get('profile_status', '')})
            continue
        left_opt = str(left_row.get('optimized_track_id', ''))
        right_opt = str(right_row.get('optimized_track_id', ''))
        if left_opt != right_opt:
            excluded.append({'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'exclusion_reason': 'endpoints_cross_optimized_tracks', 'left_layer_resolved': True, 'right_layer_resolved': True, 'left_roi_layer_id': left_row.get('roi_layer_id', ''), 'right_roi_layer_id': right_row.get('roi_layer_id', ''), 'left_optimized_track_id': left_opt, 'right_optimized_track_id': right_opt, 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'final_status': raw.get('final_status', ''), 'profile_status': raw.get('profile_status', '')})
            continue
        valid_key = (left_opt, left, right)
        if valid_key in seen_valid:
            duplicate_valid += 1
            excluded.append({'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'exclusion_reason': 'duplicate_delivered_edge', 'left_layer_resolved': True, 'right_layer_resolved': True, 'left_roi_layer_id': left_row.get('roi_layer_id', ''), 'right_roi_layer_id': right_row.get('roi_layer_id', ''), 'optimized_track_id': left_opt, 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'final_status': raw.get('final_status', ''), 'profile_status': raw.get('profile_status', '')})
            continue
        seen_valid.add(valid_key)
        resolved.append({'raw': raw, 'track_id': track, 'left_section_id': left, 'right_section_id': right, 'left_row': left_row, 'right_row': right_row, 'optimized_track_id': left_opt, 'registration_source_edge_id': f'{track}__{left:03d}_{right:03d}', 'raw_edge_id': raw_edge_id})
    summary = {'raw_edge_rows': len(raw_edges), 'raw_accepted_edge_rows': len(accepted_edges), 'valid_delivered_accepted_edge_rows': len(resolved), 'excluded_raw_accepted_edge_rows': len(excluded), 'duplicate_raw_accepted_edge_count': duplicate_raw, 'duplicate_delivered_edge_count': duplicate_valid, 'non_adjacent_raw_accepted_edge_count': non_adjacent, 'exclusion_reason_counts': dict(Counter((str(r.get('exclusion_reason', '')) for r in excluded)))}
    return (resolved, excluded, summary)
REQUIRED_RSG_MANIFESTS = ['00_manifests/registered_layer_manifest.tsv', '00_manifests/edge_transform_manifest.tsv', '00_manifests/triplet_manifest.tsv', 'Serial_DELIVERY_SUMMARY.json']

def freeze_snapshot(rsg_root: str | Path, result_root: str | Path, *, code_version: str='serial_cross_layer_dataset_set_encoding') -> dict[str, Any]:
    rsg = Path(rsg_root)
    out = Path(result_root) / '00_snapshot'
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for rel in REQUIRED_RSG_MANIFESTS:
        path = rsg / rel
        artifacts[rel] = {'path': str(path), 'exists': path.is_file(), 'sha256': sha256_file(path) if path.is_file() else None}
    summary = read_json(rsg / 'Serial_DELIVERY_SUMMARY.json', {}) or {}
    snapshot = {'dataset_version': 'prostate_serial_roi1024_native_cellpose3_dual_cross_layer_set_encoding', 'rsg_root': str(rsg), 'code_version': code_version, 'artifacts': artifacts, 'rsg_delivery_summary': summary}
    atomic_write_json(out / 'input_snapshot.json', snapshot)
    return snapshot

def validate_rsg_manifests(rsg_root: str | Path, *, check_png_pixels: bool=False, max_rows: int | None=None) -> dict[str, Any]:
    rsg = Path(rsg_root)
    layer_path = rsg / '00_manifests/registered_layer_manifest.tsv'
    edge_path = rsg / '00_manifests/edge_transform_manifest.tsv'
    layers = read_tsv(layer_path)
    edges = read_tsv(edge_path)
    if max_rows is not None:
        layers_to_check = layers[:max_rows]
    else:
        layers_to_check = layers
    errors: list[str] = []
    warnings: list[str] = []
    layer_keys: set[tuple[str, str, str]] = set()
    roi_ids: set[str] = set()
    by_track_section: dict[tuple[str, int], dict[str, str]] = {}
    by_source_track_section: dict[tuple[str, int], dict[str, str]] = {}
    for i, row in enumerate(layers_to_check):
        key = (row.get('optimized_track_id', ''), row.get('source_track_id', ''), row.get('section_id', ''))
        if key in layer_keys:
            errors.append(f'duplicate layer key {key}')
        layer_keys.add(key)
        rid = roi_layer_id(row['optimized_track_id'], row['section_id'])
        if rid in roi_ids:
            errors.append(f'duplicate roi_layer_id {rid}')
        roi_ids.add(rid)
        by_track_section[row['optimized_track_id'], parse_int(row['section_id'])] = row
        by_source_track_section[row.get('source_track_id', row['optimized_track_id']), parse_int(row['section_id'])] = row
        if parse_float(row.get('support_fraction'), -1.0) != 1.0:
            errors.append(f'support_fraction != 1 at layer row {i}: {rid}')
        if not parse_bool(row.get('single_resample_from_raw')):
            errors.append(f'single_resample_from_raw false at layer row {i}: {rid}')
        if parse_int(row.get('width')) != 1024 or parse_int(row.get('height')) != 1024:
            errors.append(f'bad dimensions at layer row {i}: {rid}')
        if abs(parse_float(row.get('mpp_um_per_px')) - 0.46) > 1e-06:
            errors.append(f'bad mpp at layer row {i}: {rid}')
        p = Path(row['path'])
        if not p.is_file():
            errors.append(f'missing png at layer row {i}: {p}')
        elif check_png_pixels:
            try:
                with Image.open(p) as img:
                    if img.format != 'PNG' or img.mode != 'RGB' or img.size != (1024, 1024):
                        errors.append(f'not RGB 1024 PNG at layer row {i}: {p} got {img.format}/{img.mode}/{img.size}')
            except Exception as exc:
                errors.append(f'cannot read png at layer row {i}: {p}: {exc}')
    resolved_edges, excluded_edges, edge_resolution_summary = resolve_accepted_registration_edges(rsg_root, layers)
    valid_edge_keys: set[tuple[str, int, int]] = set()
    reverse_seen: set[tuple[str, int, int]] = set()
    for resolved in resolved_edges:
        opt = str(resolved['optimized_track_id'])
        left = int(resolved['left_section_id'])
        right = int(resolved['right_section_id'])
        key = (opt, left, right)
        if key in valid_edge_keys:
            errors.append(f'duplicate delivered accepted edge {key}')
        valid_edge_keys.add(key)
        if (opt, right, left) in reverse_seen:
            errors.append(f'reverse duplicate delivered accepted edge {key}')
        reverse_seen.add(key)
        if abs(right - left) != 1:
            errors.append(f'non-adjacent delivered accepted edge {key}')
        raw = resolved['raw']
        if raw.get('support_fraction') not in {None, ''} and parse_float(raw.get('support_fraction'), 1.0) != 1.0:
            errors.append(f'accepted edge support_fraction != 1 {key}')
    if excluded_edges:
        warnings.append(f'excluded {len(excluded_edges)} raw accepted edge(s) without usable delivered registered-layer endpoints; see excluded_registration_edges.tsv')
    report = {'status': 'PASS' if not errors else 'FAIL', 'layer_rows': len(layers), 'checked_layer_rows': len(layers_to_check), 'edge_rows': len(edges), 'accepted_edge_rows': edge_resolution_summary['raw_accepted_edge_rows'], 'valid_delivered_accepted_edge_rows': edge_resolution_summary['valid_delivered_accepted_edge_rows'], 'excluded_raw_accepted_edge_rows': edge_resolution_summary['excluded_raw_accepted_edge_rows'], 'edge_resolution_summary': edge_resolution_summary, 'excluded_registration_edges_preview': excluded_edges[:20], '_excluded_registration_edges_full': excluded_edges, 'errors': errors[:1000], 'error_count': len(errors), 'warnings': warnings}
    return report

def seam_score_rgb_png(path: str | Path, *, grid_periods: Iterable[int]=(128, 256), neighbor_offset: int=4) -> dict[str, Any]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert('RGB'), dtype=np.float32)
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    h, w = gray.shape
    scores: dict[str, float] = {}
    max_ratio = 0.0
    max_period = None
    for period in grid_periods:
        xs = [x for x in range(period, w, period) if 2 <= x < w - 2]
        ys = [y for y in range(period, h, period) if 2 <= y < h - 2]
        vals: list[float] = []
        refs: list[float] = []
        for x in xs:
            vals.append(float(gx[:, x - 1].mean()))
            for off in (-neighbor_offset, neighbor_offset):
                xx = x + off
                if 1 <= xx < w:
                    refs.append(float(gx[:, xx - 1].mean()))
        for y in ys:
            vals.append(float(gy[y - 1, :].mean()))
            for off in (-neighbor_offset, neighbor_offset):
                yy = y + off
                if 1 <= yy < h:
                    refs.append(float(gy[yy - 1, :].mean()))
        candidate = float(np.mean(vals)) if vals else 0.0
        reference = float(np.mean(refs)) if refs else 1e-06
        ratio = candidate / max(reference, 1e-06)
        scores[f'period_{period}_candidate_gradient'] = candidate
        scores[f'period_{period}_neighbor_gradient'] = reference
        scores[f'period_{period}_ratio'] = ratio
        if ratio > max_ratio:
            max_ratio = ratio
            max_period = period
    return {'path': str(path), 'max_periodic_seam_ratio': float(max_ratio), 'max_period_px': max_period, **scores}

def _seam_scan_one(row: Mapping[str, Any]) -> dict[str, Any]:
    score = seam_score_rgb_png(row['path'])
    return {'roi_layer_id': roi_layer_id(row['optimized_track_id'], row['section_id']), 'optimized_track_id': row['optimized_track_id'], 'source_track_id': row.get('source_track_id', ''), 'section_id': int(row['section_id']), 'dense_edge_composition_count': int(row.get('dense_edge_composition_count', 0) or 0), **score}

def build_registration_qc_tables(rsg_root: str | Path, result_root: str | Path, *, scan_limit: int | None=None, workers: int=8) -> dict[str, Any]:
    out = Path(result_root) / '01_registration_qc'
    out.mkdir(parents=True, exist_ok=True)
    validation = validate_rsg_manifests(rsg_root, check_png_pixels=True, max_rows=scan_limit)
    excluded_edges = validation.pop('_excluded_registration_edges_full', []) or []
    if excluded_edges:
        atomic_write_tsv(out / 'excluded_registration_edges.tsv', excluded_edges)
        atomic_write_tsv(Path(result_root) / '00_snapshot' / 'excluded_registration_edges.tsv', excluded_edges)
    atomic_write_json(out / 'manifest_integrity.json', validation)
    if validation['status'] != 'PASS':
        return {'status': 'FAIL', 'stage': 'manifest_integrity', 'validation': validation}
    layers = read_tsv(Path(rsg_root) / '00_manifests/registered_layer_manifest.tsv')
    if scan_limit is not None:
        layers = layers[:scan_limit]
    if workers and workers > 1 and (len(layers) > 1):
        with ProcessPoolExecutor(max_workers=workers) as ex:
            seam_rows = list(ex.map(_seam_scan_one, layers, chunksize=32))
    else:
        seam_rows = [_seam_scan_one(row) for row in layers]
    seam_rows = sorted(seam_rows, key=lambda r: float(r['max_periodic_seam_ratio']), reverse=True)
    atomic_write_tsv(out / 'registered_layer_seam_scan.tsv', seam_rows)
    summary = {'status': 'PASS_PENDING_VISUAL_REVIEW', 'scanned_layers': len(seam_rows), 'top_seam_layers': seam_rows[:32], 'note': 'Automatic seam scan ranks suspicious layers only; human contact-sheet review is required before Cellpose production.'}
    atomic_write_json(out / 'registration_qc_summary.json', summary)
    return summary

def _resize_for_panel(img: Image.Image, size: int=256) -> Image.Image:
    return img.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)

def _seam_heatmap_panel(img: Image.Image, size: int=256) -> Image.Image:
    arr = np.asarray(img.convert('RGB'), dtype=np.float32).mean(axis=2)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:] = np.abs(arr[:, 1:] - arr[:, :-1])
    gy[1:, :] = np.abs(arr[1:, :] - arr[:-1, :])
    grad = np.maximum(gx, gy)
    p99 = float(np.percentile(grad, 99.5)) if grad.size else 1.0
    norm = np.clip(grad / max(p99, 1e-06), 0, 1)
    heat = np.zeros((*norm.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255 * norm).astype(np.uint8)
    heat[..., 1] = (180 * np.maximum(0, norm - 0.4) / 0.6).astype(np.uint8)
    for period in (128, 256):
        for x in range(period, norm.shape[1], period):
            heat[:, max(0, x - 1):min(norm.shape[1], x + 1), 2] = 255
        for y in range(period, norm.shape[0], period):
            heat[max(0, y - 1):min(norm.shape[0], y + 1), :, 2] = 255
    return Image.fromarray(heat, mode='RGB').resize((size, size), Image.Resampling.BILINEAR)

def _label_panel(panel: Image.Image, title: str, footer: str='') -> Image.Image:
    w, h = panel.size
    out = Image.new('RGB', (w, h + 34), 'white')
    out.paste(panel, (0, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4, 3), title[:48], fill=(0, 0, 0))
    if footer:
        draw.text((4, h + 22), footer[:64], fill=(0, 0, 0))
    return out

def _make_layer_contact_sheet(rows: list[Mapping[str, Any]], output: Path, *, panel: int=256) -> None:
    if not rows:
        return
    cols = 4
    cell_w, cell_h = (panel, panel + 34)
    sheet = Image.new('RGB', (cols * cell_w, math.ceil(len(rows) / cols) * cell_h), 'white')
    for i, row in enumerate(rows):
        try:
            img = Image.open(row['path'])
            base = _resize_for_panel(img, panel)
            heat = _seam_heatmap_panel(img, panel)
            blend = Image.blend(base, heat, 0.35)
            title = f"{row.get('roi_layer_id', '')}"
            footer = f"seam={float(row.get('max_periodic_seam_ratio', 0)):.3f}; period={row.get('max_period_px', '')}"
            tile = _label_panel(blend, title, footer)
        except Exception as exc:
            tile = _label_panel(Image.new('RGB', (panel, panel), (255, 240, 240)), 'ERROR', str(exc))
        sheet.paste(tile, (i % cols * cell_w, i // cols * cell_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f'.{output.name}.tmp')
    sheet.save(tmp, format='PNG')
    tmp.replace(output)

def _make_edge_contact_sheet(edge_rows: list[Mapping[str, Any]], layer_lookup: Mapping[tuple[str, int], Mapping[str, str]], output: Path, *, panel: int=192) -> None:
    if not edge_rows:
        return
    cols = 5
    cell_w, cell_h = (panel, panel + 34)
    sheet = Image.new('RGB', (cols * cell_w, len(edge_rows) * cell_h), 'white')
    for r, edge in enumerate(edge_rows):
        track = str(edge.get('track_id') or edge.get('optimized_track_id') or '')
        left_id = parse_int(edge.get('left_id', edge.get('left_section_id')))
        right_id = parse_int(edge.get('right_id', edge.get('right_section_id')))
        left_row = layer_lookup.get((track, left_id))
        right_row = layer_lookup.get((track, right_id))
        panels: list[tuple[str, Image.Image, str]] = []
        try:
            if left_row is None or right_row is None:
                raise KeyError(f'missing layer for {track}:{left_id}-{right_id}')
            left = Image.open(left_row['path']).convert('RGB')
            right = Image.open(right_row['path']).convert('RGB')
            lsmall = _resize_for_panel(left, panel)
            rsmall = _resize_for_panel(right, panel)
            la = np.asarray(left, dtype=np.float32)
            ra = np.asarray(right, dtype=np.float32)
            overlay = np.zeros_like(la, dtype=np.uint8)
            overlay[..., 0] = np.clip(la.mean(axis=2), 0, 255).astype(np.uint8)
            overlay[..., 1] = np.clip(ra.mean(axis=2), 0, 255).astype(np.uint8)
            overlay[..., 2] = np.clip(ra.mean(axis=2), 0, 255).astype(np.uint8)
            blend = Image.blend(left, right, 0.5)
            seam = _seam_heatmap_panel(blend, panel)
            panels = [('left registered', lsmall, f'sec={left_id}'), ('right registered', rsmall, f'sec={right_id}'), ('red/cyan overlay', Image.fromarray(overlay).resize((panel, panel), Image.Resampling.BILINEAR), ''), ('alpha blend', _resize_for_panel(blend, panel), ''), ('seam heatmap', seam, f"origin={edge.get('selection_origin', edge.get('attempt_origin', ''))}")]
        except Exception as exc:
            panels = [('ERROR', Image.new('RGB', (panel, panel), (255, 235, 235)), str(exc))] * cols
        for c, (title, img, footer) in enumerate(panels):
            tile = _label_panel(img, f"{edge.get('edge_id', track)} {title}", footer)
            sheet.paste(tile, (c * cell_w, r * cell_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f'.{output.name}.tmp')
    sheet.save(tmp, format='PNG')
    tmp.replace(output)

def build_registration_visual_qc_contact_sheets(rsg_root: str | Path, result_root: str | Path, *, top_layer_count: int=32, edge_count: int=48) -> dict[str, Any]:
    qc_root = Path(result_root) / '01_registration_qc'
    seam_path = qc_root / 'registered_layer_seam_scan.tsv'
    if not seam_path.is_file():
        build_registration_qc_tables(rsg_root, result_root)
    seam_rows = read_tsv(seam_path)
    top_rows = seam_rows[:top_layer_count]
    layer_sheet = qc_root / 'visual_qc_top_seam_layers.png'
    _make_layer_contact_sheet(top_rows, layer_sheet)
    layers = read_tsv(Path(rsg_root) / '00_manifests/registered_layer_manifest.tsv')
    layer_lookup = {(str(r['optimized_track_id']), parse_int(r['section_id'])): r for r in layers}
    layer_lookup.update({(str(r.get('source_track_id', r['optimized_track_id'])), parse_int(r['section_id'])): r for r in layers})
    layer_rows_for_resolution = []
    for idx, r in enumerate(layers):
        rr = dict(r)
        rr['layer_index'] = idx
        rr['roi_layer_id'] = roi_layer_id(r['optimized_track_id'], r['section_id'])
        layer_rows_for_resolution.append(rr)
    resolved_edges, excluded_edges, edge_resolution_summary = resolve_accepted_registration_edges(rsg_root, layer_rows_for_resolution)
    edges = []
    for re in resolved_edges:
        e = dict(re['raw'])
        e['track_id'] = re['optimized_track_id']
        e['optimized_track_id'] = re['optimized_track_id']
        e['source_track_id'] = re['left_row'].get('source_track_id', '')
        e['left_id'] = f"{int(re['left_section_id']):03d}"
        e['right_id'] = f"{int(re['right_section_id']):03d}"
        e['edge_id'] = f"{re['optimized_track_id']}__{int(re['left_section_id']):03d}_{int(re['right_section_id']):03d}"
        edges.append(e)
    selected: list[Mapping[str, Any]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    origins: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for e in edges:
        origins[str(e.get('selection_origin', e.get('attempt_origin', 'unknown')))].append(e)
    for origin in sorted(origins):
        if origins[origin]:
            e = origins[origin][len(origins[origin]) // 2]
            k = (str(e.get('track_id') or e.get('optimized_track_id') or ''), parse_int(e.get('left_id', e.get('left_section_id'))), parse_int(e.get('right_id', e.get('right_section_id'))))
            if k not in seen_keys:
                selected.append(e)
                seen_keys.add(k)
    if edge_count > len(selected) and edges:
        step = max(1, len(edges) // (edge_count - len(selected)))
        for e in edges[::step]:
            k = (str(e.get('track_id') or e.get('optimized_track_id') or ''), parse_int(e.get('left_id', e.get('left_section_id'))), parse_int(e.get('right_id', e.get('right_section_id'))))
            if k in seen_keys:
                continue
            selected.append(e)
            seen_keys.add(k)
            if len(selected) >= edge_count:
                break
    for metric in ('median_tre_um', 'p90_tre_um', 'jacobian_min', 'inverse_consistency_p90_um'):
        ranked = sorted(edges, key=lambda e: parse_float(e.get(metric), -1000000000.0), reverse=True)
        for e in ranked[:4]:
            k = (str(e.get('track_id') or e.get('optimized_track_id') or ''), parse_int(e.get('left_id', e.get('left_section_id'))), parse_int(e.get('right_id', e.get('right_section_id'))))
            if k not in seen_keys and len(selected) < edge_count:
                selected.append(e)
                seen_keys.add(k)
    edge_sheet = qc_root / 'visual_qc_accepted_edges_real_registered.png'
    _make_edge_contact_sheet(selected[:edge_count], layer_lookup, edge_sheet)
    manifest = {'status': 'PASS_PENDING_VISUAL_REVIEW_DECISION', 'top_layer_count': len(top_rows), 'edge_count': len(selected[:edge_count]), 'layer_contact_sheet': str(layer_sheet), 'edge_contact_sheet': str(edge_sheet), 'note': 'Review these real registered PNG sheets before writing registration_visual_qc_decision.json with status PASS.'}
    atomic_write_json(qc_root / 'visual_qc_contact_sheets.json', manifest)
    return manifest

def build_segmentation_source_manifest(rsg_root: str | Path, result_root: str | Path, *, bucket_count: int=64, compute_sha: bool=True) -> list[dict[str, Any]]:
    layers = read_tsv(Path(rsg_root) / '00_manifests/registered_layer_manifest.tsv')
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(layers):
        rid = roi_layer_id(row['optimized_track_id'], row['section_id'])
        png = Path(row['path'])
        sha = sha256_file(png) if compute_sha else ''
        rows.append({'layer_index': idx, 'roi_layer_id': rid, 'optimized_track_id': row['optimized_track_id'], 'source_track_id': row['source_track_id'], 'section_id': int(row['section_id']), 'registered_png_path': str(png), 'registered_png_sha256': sha, 'png_path': str(png), 'png_sha256': sha, 'source_png_path': str(png), 'source_png_sha256': sha, 'render_status': 'complete', 'track_id': row['optimized_track_id'], 'component_id': row.get('source_track_id', row['optimized_track_id']), 'width': int(row['width']), 'height': int(row['height']), 'mode': 'RGB', 'mpp_um_per_px': float(row['mpp_um_per_px']), 'support_fraction': float(row['support_fraction']), 'single_resample_from_raw': row['single_resample_from_raw'], 'dense_edge_composition_count': int(row.get('dense_edge_composition_count', 0) or 0), 'registration_visual_qc': 'PASS_PENDING_VISUAL_REVIEW', 'segmentation_bucket': stable_bucket(rid, bucket_count)})
    out = Path(result_root) / '02_segmentation' / '00_manifests'
    atomic_write_tsv(out / 'serial_segmentation_source_manifest.tsv', rows)
    atomic_write_json(out / 'source_manifest_summary.json', {'status': 'PASS', 'layer_count': len(rows), 'bucket_count': bucket_count})
    return rows

def build_edge_job_manifest(rsg_root: str | Path, result_root: str | Path, layer_rows: Iterable[Mapping[str, Any]] | None=None, *, bucket_count: int=64) -> list[dict[str, Any]]:
    if layer_rows is None:
        layer_rows = build_segmentation_source_manifest(rsg_root, result_root, compute_sha=False)
    resolved_edges, excluded_edges, resolution_summary = resolve_accepted_registration_edges(rsg_root, layer_rows)
    output: list[dict[str, Any]] = []
    for resolved in resolved_edges:
        raw = resolved['raw']
        left = int(resolved['left_section_id'])
        right = int(resolved['right_section_id'])
        left_row = resolved['left_row']
        right_row = resolved['right_row']
        optimized_track = str(resolved['optimized_track_id'])
        eid = f'{optimized_track}__{left:03d}_{right:03d}'
        output.append({'edge_id': eid, 'registration_source_edge_id': resolved['registration_source_edge_id'], 'raw_edge_id': resolved['raw_edge_id'], 'optimized_track_id': optimized_track, 'source_track_id': left_row.get('source_track_id', ''), 'left_layer_index': int(left_row['layer_index']), 'right_layer_index': int(right_row['layer_index']), 'left_roi_layer_id': left_row['roi_layer_id'], 'right_roi_layer_id': right_row['roi_layer_id'], 'left_section_id': left, 'right_section_id': right, 'selection_origin': raw.get('selection_origin', raw.get('attempt_origin', raw.get('selected_mode', ''))), 'acceptance_profile': raw.get('acceptance_profile', ''), 'profile_status': raw.get('profile_status', ''), 'final_status': raw.get('final_status', ''), 'median_tre_um': raw.get('median_tre_um', ''), 'p90_tre_um': raw.get('p90_tre_um', ''), 'jacobian_min': raw.get('jacobian_min', ''), 'jacobian_p1': raw.get('jacobian_p1', ''), 'jacobian_p99': raw.get('jacobian_p99', ''), 'inverse_consistency_p90_um': raw.get('inverse_consistency_p90_um', ''), 'support_fraction': raw.get('support_fraction', ''), 'transform_path': raw.get('transform_path', ''), 'transform_sha256': raw.get('transform_sha256', ''), 'matching_bucket': stable_bucket(eid, bucket_count)})
    out = Path(result_root) / '04_cross_layer_matches' / '00_manifests'
    atomic_write_tsv(out / 'edge_job_manifest.tsv', output)
    if excluded_edges:
        atomic_write_tsv(Path(result_root) / '00_snapshot' / 'excluded_registration_edges.tsv', excluded_edges)
        atomic_write_tsv(out / 'excluded_registration_edges.tsv', excluded_edges)
    summary = dict(resolution_summary)
    summary.update({'status': 'PASS', 'edge_job_count': len(output), 'bucket_count': bucket_count})
    atomic_write_json(out / 'edge_job_manifest_summary.json', summary)
    return output
