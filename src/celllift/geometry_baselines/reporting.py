from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from .io_utils import atomic_json, sha256
from .protocol import PROTOCOL_ID

def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _number(value: Any) -> str:
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.6f}'
    return str(value)

def _metric_table(dataset: str, metrics: Sequence[Mapping[str, Any]]) -> list[str]:
    primary = 'QWK' if dataset == 'sicapv2' else 'patient_AUROC' if any(('patient_AUROC' in row for row in metrics)) else 'mean_fold_patient_AUROC'
    sample_name = 'Patches' if dataset == 'sicapv2' else 'Patients'
    sample_key = 'patches' if dataset == 'sicapv2' else 'patients'
    lines = [f'| Encoder | Arm | Route | {sample_name} | {primary} |', '|---|---|---|---:|---:|']
    for row in metrics:
        lines.append(f"| {row.get('encoder')} | {row.get('arm_id')} | {row.get('route', 'primary')} | {_number(row.get(sample_key))} | {_number(row.get(primary))} |")
    return lines

def _seed_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ['| Encoder | Arm | Metric | Five seeds | Mean | SD |', '|---|---|---|---|---:|---:|']
    for row in rows:
        seeds = ', '.join((_number(value) for value in row.get('seed_values', [])))
        lines.append(f"| {row.get('encoder')} | {row.get('arm_id')} | {row.get('metric')} | {seeds} | {_number(row.get('mean'))} | {_number(row.get('std'))} |")
    return lines

def _comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ['| Encoder | Family | Comparison | Delta | 95% CI | raw p | Holm p | Positive |', '|---|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['encoder']} | {row['family']} | {row['comparison']} | {_number(row['delta'])} | [{_number(row['ci_low'])}, {_number(row['ci_high'])}] | {_number(row['p_one_sided'])} | {_number(row['holm_p'])} | {bool(row['positive'])} |")
    return lines

def render_report(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(cfg['paths']['result_root'])
    dataset = str(cfg['dataset'])
    validation = _load(root / 'metrics' / 'validation' / 'summary.json')
    validation_tile = _load(root / 'metrics' / 'validation_tile' / 'summary.json')
    official = _load(root / 'official_test' / 'evaluation' / 'summary.json')
    official_tile_metrics = _rows(root / 'official_test' / 'evaluation' / 'tile_metrics.parquet')
    seed_summary = _rows(root / 'official_test' / 'evaluation' / 'seed_mean_std.parquet')
    fidelity = _load(root / 'paper_baseline' / 'fidelity.json')
    lines = [f'# SetEncoder geometry_baselines report: {dataset}', '', f'Protocol: `{PROTOCOL_ID}`', '']
    if fidelity:
        lines.extend(['## Paper fidelity', '', '```json', json.dumps(fidelity, indent=2, sort_keys=True), '```', ''])
    if validation:
        lines.extend(['## Validation', '', *_metric_table(dataset, validation['metrics']), ''])
        comparisons = validation.get('statistics', {}).get('comparisons', [])
        if comparisons:
            lines.extend(['### Registered validation comparisons', '', *_comparison_table(comparisons), ''])
        if validation.get('ncr_qc'):
            lines.extend(['### NCR3D QC', '', '```json', json.dumps(validation['ncr_qc'], indent=2, sort_keys=True), '```', ''])
        if validation_tile:
            lines.extend(['### CRC tile-level validation route', '', *_metric_table(dataset, validation_tile['metrics']), '', '#### Registered tile-route comparisons', '', *_comparison_table(validation_tile.get('statistics', {}).get('comparisons', [])), ''])
    else:
        lines.extend(['## Validation', '', 'Incomplete; no partial scientific comparison is rendered.', ''])
    if official:
        lines.extend(['## Official TEST ensemble', '', *_metric_table(dataset, official['metrics']), ''])
        comparisons = official.get('statistics', {}).get('comparisons', [])
        if comparisons:
            lines.extend(['### Registered TEST comparisons', '', *_comparison_table(comparisons), ''])
        if seed_summary:
            lines.extend(['### Five single seeds and mean ± SD', '', *_seed_table(seed_summary), ''])
        if official_tile_metrics:
            lines.extend(['### CRC tile-level official TEST route', '', *_metric_table(dataset, official_tile_metrics), ''])
            tile_comparisons = official.get('tile_statistics', {}).get('comparisons', [])
            if tile_comparisons:
                lines.extend(['#### Registered tile-route TEST comparisons', '', *_comparison_table(tile_comparisons), ''])
    else:
        lines.extend(['## Official TEST', '', 'Locked or incomplete.', ''])
    destination = root / 'figures' / 'report.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(destination)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'validation_present': validation is not None, 'official_test_present': official is not None, 'report': str(destination), 'report_sha256': sha256(destination)}
    atomic_json(root / 'runtime' / 'report.json', payload)
    return payload
