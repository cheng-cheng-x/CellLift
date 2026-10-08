from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
from collections import Counter
from celllift.runtime import ResourcePath as Path
from typing import Any
from .barcodes import keep_primary_dx, missing, patient_id
from . import paths
from .io_utils import write_json
from .protocol import ER_IDS, HISTO_EXACT, HISTO_IDS, LUMAB_IDS, PROTOCOL_ID, SUBTYPE4, SUBTYPE4_IDS, SUBTYPE_STRATUM

def parse_bcr_clinical(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    if len(lines) < 4:
        raise ValueError(f'clinical file too short: {path}')
    header = next(csv.reader([lines[0]], delimiter='\t'))
    index = {name: i for i, name in enumerate(header)}
    required = ('bcr_patient_barcode', 'history_neoadjuvant_treatment', 'histological_type', 'er_status_by_ihc', 'pr_status_by_ihc', 'her2_status_by_ihc', 'her2_fish_status')
    for name in required:
        if name not in index:
            raise KeyError(name)
    rows: dict[str, dict[str, str]] = {}
    for line in lines[3:]:
        if not line.strip():
            continue
        parts = next(csv.reader([line], delimiter='\t'))
        barcode = patient_id(parts[index['bcr_patient_barcode']] if index['bcr_patient_barcode'] < len(parts) else '')
        if not barcode:
            continue
        rows[barcode] = {name: (parts[index[name]] if index[name] < len(parts) else '').strip() for name in required}
    return rows

def parse_cbio_subtype(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    header_index = None
    header: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith('#') or not line.strip():
            continue
        parts = next(csv.reader([line], delimiter='\t'))
        if parts and parts[0] == 'PATIENT_ID':
            header = parts
            header_index = i
            break
    if header_index is None or 'SUBTYPE' not in header:
        raise ValueError('cBioPortal PATIENT_ID/SUBTYPE header not found')
    subtype_col = header.index('SUBTYPE')
    mapping: dict[str, str] = {}
    for line in lines[header_index + 1:]:
        if not line.strip() or line.startswith('#'):
            continue
        parts = next(csv.reader([line], delimiter='\t'))
        pid = patient_id(parts[0] if parts else '')
        if not pid:
            continue
        mapping[pid] = parts[subtype_col].strip() if subtype_col < len(parts) else ''
    return mapping

def project_inventory(slides) -> list[dict[str, Any]]:
    rows = []
    for record in slides:
        if str(record.get('cohort') or '') != 'BRCA':
            continue
        slide_id = str(record.get('slide_id') or '')
        if not keep_primary_dx(slide_id):
            continue
        if not bool(record.get('is_eligible')):
            continue
        pid = patient_id(record.get('patient_id') or slide_id)
        if not pid:
            continue
        rows.append({'patient_id': pid, 'slide_id': slide_id, 'source_path': str(record.get('source_path') or ''), 'link_path': str(record.get('link_path') or ''), 'mpp_x': float(record['mpp_x']) if record.get('mpp_x') not in (None, '') else None, 'mpp_y': float(record['mpp_y']) if record.get('mpp_y') not in (None, '') else None, 'width0': int(record['width0']) if record.get('width0') not in (None, '') else None, 'height0': int(record['height0']) if record.get('height0') not in (None, '') else None})
    rows.sort(key=lambda row: (row['patient_id'], row['slide_id']))
    return rows

def _histo_task(raw: str) -> str | None:
    if missing(raw):
        return None
    mapped = HISTO_EXACT.get(raw.strip())
    if mapped in {'IDC', 'ILC'}:
        return mapped
    return None

def _er_task(raw: str) -> str | None:
    text = raw.strip()
    if text in {'Positive', 'Negative'}:
        return text
    return None

def build_patient_rows(inventory: list[dict[str, Any]], clinical: dict[str, dict[str, str]], subtypes: dict[str, str]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for slide in inventory:
        by_patient.setdefault(slide['patient_id'], []).append(slide)
    rows = []
    for pid in sorted(by_patient):
        slides = by_patient[pid]
        clin = clinical.get(pid)
        excluded = ''
        neoadjuvant = ''
        if clin is None:
            excluded = 'no_clinical'
        else:
            neoadjuvant = clin.get('history_neoadjuvant_treatment', '')
            if neoadjuvant != 'No':
                excluded = 'neoadjuvant_not_no'
        subtype_raw = subtypes.get(pid, '')
        if missing(subtype_raw):
            subtype_raw = ''
        histo_raw = clin.get('histological_type', '') if clin else ''
        er_raw = clin.get('er_status_by_ihc', '') if clin else ''
        pr_raw = clin.get('pr_status_by_ihc', '') if clin else ''
        her2_ihc = clin.get('her2_status_by_ihc', '') if clin else ''
        her2_fish = clin.get('her2_fish_status', '') if clin else ''
        subtype4 = subtype_raw if subtype_raw in SUBTYPE4 else ''
        is_normal = subtype_raw == 'BRCA_Normal'
        histo = _histo_task(histo_raw)
        er = _er_task(er_raw)
        in_master = not excluded and bool(subtype4 or histo or er or is_normal)
        rows.append({'patient_id': pid, 'n_eligible_slides': len(slides), 'eligible_slide_ids': [slide['slide_id'] for slide in slides], 'neoadjuvant': neoadjuvant, 'excluded_reason': excluded, 'in_master': in_master, 'subtype_raw': subtype_raw, 'subtype4': subtype4, 'subtype4_id': SUBTYPE4_IDS.get(subtype4), 'luma_lumb': subtype_raw if subtype_raw in LUMAB_IDS else '', 'luma_lumb_id': LUMAB_IDS.get(subtype_raw), 'histo_raw': histo_raw, 'idc_ilc': histo or '', 'idc_ilc_id': HISTO_IDS.get(histo) if histo else None, 'er_raw': er_raw, 'er': er or '', 'er_id': ER_IDS.get(er) if er else None, 'pr_raw': pr_raw, 'her2_ihc_raw': her2_ihc, 'her2_fish_raw': her2_fish, 'is_normal_like': is_normal, 'subtype_stratum': SUBTYPE_STRATUM.get(subtype_raw, 'Missing')})
    return rows

def label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    master = [row for row in rows if row['in_master']]
    return {'protocol_id': PROTOCOL_ID, 'n_patients_with_eligible_slides': len(rows), 'n_master': len(master), 'excluded': dict(Counter((row['excluded_reason'] or 'kept' for row in rows))), 'subtype_raw': dict(Counter((row['subtype_raw'] or 'empty' for row in master))), 'subtype4': dict(Counter((row['subtype4'] or 'empty' for row in master))), 'idc_ilc': dict(Counter((row['idc_ilc'] or 'empty' for row in master))), 'er': dict(Counter((row['er'] or 'empty' for row in master))), 'n_subtype4': sum((1 for row in master if row['subtype4'])), 'n_luma_lumb': sum((1 for row in master if row['luma_lumb'])), 'n_idc_ilc': sum((1 for row in master if row['idc_ilc'])), 'n_er': sum((1 for row in master if row['er'])), 'n_normal_like': sum((1 for row in master if row['is_normal_like']))}

def write_labels(rows: list[dict[str, Any]], inventory: list[dict[str, Any]], root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    label_path = paths.label_dir(base) / 'patient_labels.parquet'
    inv_path = paths.inventory_dir(base) / 'eligible_slides.parquet'
    pd.DataFrame(rows).to_parquet(label_path, index=False)
    pd.DataFrame(inventory).to_parquet(inv_path, index=False)
    summary = label_summary(rows)
    summary['paths'] = {'labels': str(label_path), 'inventory': str(inv_path)}
    write_json(paths.label_dir(base) / 'summary.json', summary)
    return summary

def build_labels(root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    cbio = paths.source_dir(base) / 'data_clinical_patient.txt'
    slides = pd.read_parquet(paths.STAGE1_SLIDES).to_dict('records')
    inventory = project_inventory(slides)
    clinical = parse_bcr_clinical(paths.CLINICAL_PATIENT)
    subtypes = parse_cbio_subtype(cbio)
    rows = build_patient_rows(inventory, clinical, subtypes)
    return write_labels(rows, inventory, base)
