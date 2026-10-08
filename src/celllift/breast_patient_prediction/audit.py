from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import csv
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
from .barcodes import patient_id
from . import paths
from .io_utils import write_json
from .protocol import PROTOCOL_ID

def _ids_from_values(values: Iterable[object]) -> set[str]:
    return {pid for value in values if (pid := patient_id(str(value or '')))}

def _read_table_ids(path: Path, columns: tuple[str, ...]) -> set[str]:
    if not path.is_file():
        return set()
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        ids: set[str] = set()
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    ids |= _ids_from_values((row.get(col) for col in columns if col in row))
                    ids |= _ids_from_values([row.get('patient_id'), row.get('sample_id'), row.get('wsi_id')])
                elif isinstance(row, str):
                    ids |= _ids_from_values([row])
        return ids
    if path.suffix.lower() == '.csv':
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            ids = set()
            for row in reader:
                ids |= _ids_from_values((row.get(col) for col in columns))
                ids |= _ids_from_values([row.get('patient_id'), row.get('source_image'), row.get('patch_id')])
            return ids
    if path.suffix.lower() == '.parquet':
        import pandas as pd
        table = pd.read_parquet(path)
        ids = set()
        for col in columns:
            if col in table.columns:
                ids |= _ids_from_values(table[col].astype(str))
        for col in ('patient_id', 'wsi_id', 'sample_id'):
            if col in table.columns:
                ids |= _ids_from_values(table[col].astype(str))
        return ids
    return set()

def audit_overlap(root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = paths.ensure_tree(root)
    labels = pd.read_parquet(paths.label_dir(base) / 'patient_labels.parquet')
    master = set(labels.loc[labels['in_master'], 'patient_id'].astype(str))
    pretrain = pd.read_parquet(paths.STAGE1_SLIDES)
    pretrain_ids = set(pretrain.loc[(pretrain['cohort'] == 'BRCA') & pretrain['is_dx'], 'patient_id'].map(lambda value: patient_id(str(value))))
    pretrain_ids.discard('')
    conditional_geometry_ids = _read_table_ids(paths.BRCA_conditional_geometry_MANIFEST, ('source_image', 'patch_id'))
    bracs_ids: set[str] = set()
    bracs_used = []
    for candidate in paths.BRACS_CANDIDATES:
        found = _read_table_ids(candidate, ('patient_id', 'wsi_id', 'sample_id'))
        if found or candidate.is_file():
            bracs_used.append({'path': str(candidate), 'n_tcga_like': len(found), 'exists': candidate.is_file()})
            bracs_ids |= found
    report = {'protocol_id': PROTOCOL_ID, 'policy': 'audit_only_keep', 'n_master': len(master), 'overlap': {'brca_conditional_geometry': {'n_source_patients': len(conditional_geometry_ids), 'n_overlap_master': len(master & conditional_geometry_ids)}, 'tcga_pretrain_dx': {'n_source_patients': len(pretrain_ids), 'n_overlap_master': len(master & pretrain_ids)}, 'bracs_tcga_like': {'n_source_patients': len(bracs_ids), 'n_overlap_master': len(master & bracs_ids), 'sources': bracs_used}}, 'claim': 'Prostate-trained ProjectionScene transfer to BRCA endpoints. Not an in-domain validation of breast reconstruction on these slides.'}
    write_json(paths.audit_dir(base) / 'overlap.json', report)
    return report
