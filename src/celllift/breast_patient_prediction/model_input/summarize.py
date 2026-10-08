from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any
from .. import paths
from ..io_utils import atomic_json
from . import layout
from .pack import coverage_summary, write_indices
DEFAULT_DOCS = Path(_resource_path('artifact_0020'))
from celllift.runtime import output_root
LOCAL_DOCS = output_root() / 'breast_patient_tasks' / 'report.md'

def _md(summary: dict[str, Any], pilot: dict[str, Any] | None) -> str:
    splits = summary.get('splits') or {}
    tasks = summary.get('tasks') or {}
    lines = ['# Model-input set_encoding', '', f"Protocol `{summary.get('protocol_id', '')}`. Shared RGB / nuclei / DINO / selected-scene cache.", 'No training. Splits and labels unchanged.', '', f"Status: **{summary.get('status', 'UNKNOWN')}**", '', '## Coverage', '', f"- Tiles: {summary.get('n_success')} / {summary.get('n_tiles')} success", f"- Patients: {summary.get('n_evaluable_patients')} evaluable / {summary.get('n_patients')} master", f"- Included 3D-invalid rate: {summary.get('invalid_3d_rate')}", '', '### By split', '', '| split | tiles | success | empty-nuclei | empty rate |', '|---|---:|---:|---:|---:|']
    for name, row in sorted(splits.items()):
        lines.append(f"| {name} | {row.get('tiles')} | {row.get('success')} | {row.get('empty_nuclei')} | {row.get('empty_rate')} |")
    lines.extend(['', '### Tasks (nominal vs evaluable)', '', '| task | nominal | evaluable |', '|---|---:|---:|'])
    for name, row in tasks.items():
        lines.append(f"| {name} | {row.get('nominal')} | {row.get('evaluable')} |")
    if pilot:
        sec = pilot.get('seconds') or {}
        extra = pilot.get('extrapolated_hours_123406') or {}
        lines.extend(['', '## 32-tile pilot', '', f"- Status: {pilot.get('status')}", f"- Tiles: {pilot.get('n_tiles')}", f"- Seconds RGB/mask/graph/infer: {sec.get('rgb')}, {sec.get('mask')}, {sec.get('graph')}, {sec.get('infer')}", f"- Extrapolated GPU-hours (mask, one GPU): {extra.get('mask_one_gpu')}", f"- Extrapolated GPU-hours (DINO+ProjectionScene, one GPU): {extra.get('infer_one_gpu')}", '', 'Overlays are evidence only; Cellpose hyperparameters were not changed.'])
    lines.extend(['', '## Claim boundary', '', 'Prostate-trained ProjectionScene transfer cache on these BRCA tiles.', 'Not in-domain validation of breast reconstruction.', '', 'Patient-level paths remain under the Data root; this page stores aggregates only.', ''])
    return '\n'.join(lines) + '\n'

def write_results_page(docs: Path | None=None, data_root: Path | None=None) -> dict[str, Any]:
    write_indices(data_root)
    summary = coverage_summary(data_root)
    base = layout.root(data_root)
    pilot_path = base / 'qc' / 'pilot_summary.json'
    pilot = json.loads(pilot_path.read_text(encoding='utf-8')) if pilot_path.is_file() else None
    dest = docs or (DEFAULT_DOCS if DEFAULT_DOCS.parent.is_dir() else LOCAL_DOCS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(_md(summary, pilot), encoding='utf-8')
    result = paths.result_root()
    result.mkdir(parents=True, exist_ok=True)
    atomic_json(result / 'qc_summary.json', summary)
    return {'docs': str(dest), 'summary': summary.get('status')}
