from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
from . import paths
from .io_utils import read_json, write_text
from .protocol import PROTOCOL_ID, TASKS

def _md_counts(title: str, mapping: dict[str, Any]) -> list[str]:
    lines = [f'### {title}', '']
    for key in sorted(mapping):
        lines.append(f'- `{key}`: {mapping[key]}')
    lines.append('')
    return lines

def render_summary(root: Path | None=None) -> str:
    base = paths.data_root(root)
    labels = read_json(paths.label_dir(base) / 'summary.json')
    splits = read_json(paths.split_dir(base) / 'split_meta.json')
    tiles = read_json(paths.tile_dir(base) / 'summary.json')
    audit = read_json(paths.audit_dir(base) / 'overlap.json')
    lines = ['# Cohort summary', '', f'Protocol `{PROTOCOL_ID}`. Aggregate counts only.', '', '## Eligibility', '', f"- Patients with eligible primary DX slides: {labels['n_patients_with_eligible_slides']}", f"- Master universe: {labels['n_master']}", f"- subtype4: {labels['n_subtype4']}", f"- luma_lumb: {labels['n_luma_lumb']}", f"- idc_ilc: {labels['n_idc_ilc']}", f"- er_ihc: {labels['n_er']}", f"- Normal-like in master: {labels['n_normal_like']}", '']
    lines += _md_counts('Exclusion reasons', labels['excluded'])
    lines += _md_counts('Master subtype4', labels['subtype4'])
    lines += _md_counts('Master IDC/ILC', labels['idc_ilc'])
    lines += _md_counts('Master ER', labels['er'])
    lines.extend(['## Split', '', f"- Accepted seed: `{splits['seed']}` (base 42 + {splits['offset']})", f"- Floors: HER2 {splits['floors']['her2']}, ILC {splits['floors']['ilc']}, ER- {splits['floors']['er_neg']}", f"- Lowered floors: {splits['lowered_floors']}", ''])
    for part in ('fit', 'val', 'test'):
        counts = splits['counts'][part]
        lines.append(f"- {part}: n={counts['n']}, HER2={counts['her2']}, ILC={counts['ilc']}, ER-={counts['er_neg']}")
    lines.append('')
    lines.append('Task membership after inherited split:')
    lines.append('')
    for task in TASKS:
        item = splits['task_counts'][task]
        lines.append(f"- `{task}`: FIT {item['fit']} / VAL {item['val']} / TEST {item['test']}")
    lines.extend(['', '## Tiles', '', f"- Selected tiles: {tiles['n_tiles']}", f"- Patients in merge: {tiles['n_patients']}", f"- Unevaluable (<{tiles['min_tiles']} tiles): {tiles['n_unevaluable']}", f"- Slide read errors: {tiles['n_slide_errors']}", '', '## Overlap audit (keep all master patients)', '', f"- BRCA_conditional_geometry overlap: {audit['overlap']['brca_conditional_geometry']['n_overlap_master']} / source {audit['overlap']['brca_conditional_geometry']['n_source_patients']}", f"- tcga-pretrain DX overlap: {audit['overlap']['tcga_pretrain_dx']['n_overlap_master']} / source {audit['overlap']['tcga_pretrain_dx']['n_source_patients']}", f"- BRACS TCGA-like overlap: {audit['overlap']['bracs_tcga_like']['n_overlap_master']}", '', audit['claim'], ''])
    return '\n'.join(lines)

def write_docs_summary(docs_path: Path, root: Path | None=None) -> Path:
    text = render_summary(root)
    write_text(docs_path, text)
    write_text(paths.audit_dir(root) / 'cohort_summary.md', text)
    return docs_path
