from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from .artifact_filter import exclusion_manifest_rows, mark_artifact_cells
from .common import atomic_write_gzip_json, atomic_write_json, atomic_write_tsv, parse_bool, read_gzip_json, read_json, read_tsv, sha256_file, sha256_text, stable_bucket
from .index import assert_endpoint_degree_at_most_one, build_nucleus_link_index
from .production_io import compact_partitioned_rows, write_table
from .schemas import CELL_RELATIONS, QUALITY_CLASSES, pyarrow_schemas
from .segmentation_context import layer_summary_from_segmentation_row

def build_artifact_filtered_layer_manifest(*, result_root: str | Path, source_manifest: str | Path | None=None, cellpose_manifest: str | Path | None=None, prefer_parquet: bool=True) -> dict[str, Any]:
    root = Path(result_root)
    source_manifest = Path(source_manifest) if source_manifest else root / '02_segmentation/00_manifests/serial_segmentation_source_manifest.tsv'
    cellpose_manifest = Path(cellpose_manifest) if cellpose_manifest else root / '02_segmentation/cellpose3_dual/05_manifests/segmentation_manifest.tsv'
    source_rows = read_tsv(source_manifest)
    cp_rows = read_tsv(cellpose_manifest)
    source_by_id = {r['roi_layer_id']: r for r in source_rows}
    final_visual_qc: str | None = None
    visual_qc_decision_path = root / '01_registration_qc/registration_visual_qc_decision.json'
    if visual_qc_decision_path.exists():
        visual_qc_decision = read_json(visual_qc_decision_path)
        if str(visual_qc_decision.get('status', '')) == 'PASS' and str(visual_qc_decision.get('registration_visual_qc', '')) == 'PASS':
            final_visual_qc = 'PASS'
    out_root = root / '03_instance_relations'
    filtered_root = out_root / 'filtered_pairs'
    layer_rows: list[dict[str, Any]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    missing_source: list[str] = []
    for cp in cp_rows:
        rid = cp['roi_layer_id']
        src = source_by_id.get(rid)
        if src is None:
            missing_source.append(rid)
            continue
        marked = mark_artifact_cells(read_gzip_json(cp['pairs_path']))
        bucket = stable_bucket(rid, 64)
        filtered_path = filtered_root / f'bucket_{bucket:02d}' / f'{rid}.json.gz'
        atomic_write_gzip_json(filtered_path, marked)
        payloads[rid] = marked
        summary = marked.get('summary', {}) or {}
        row = {**src, 'nucleus_mask_path': cp['nucleus_mask_path'], 'nucleus_mask_sha256': cp['nucleus_mask_sha256'], 'nucleus_instances_path': cp['nucleus_instances_path'], 'nucleus_instances_sha256': cp['nucleus_instances_sha256'], 'cell_mask_path': cp['cell_mask_path'], 'cell_mask_sha256': cp['cell_mask_sha256'], 'cell_instances_path': cp['cell_instances_path'], 'cell_instances_sha256': cp['cell_instances_sha256'], 'raw_pairs_path': cp['pairs_path'], 'raw_pairs_sha256': cp['pairs_sha256'], 'pairs_path': str(filtered_path), 'pairs_sha256': sha256_file(filtered_path), 'nucleus_count': int(cp.get('nucleus_count', 0) or 0), 'cell_count': int(cp.get('cell_count', 0) or 0), 'raw_cell_count': int(summary.get('raw_cell_count', cp.get('cell_count', 0)) or 0), 'valid_cell_count': int(summary.get('valid_cell_count', cp.get('cell_count', 0)) or 0), 'artifact_cell_count': int(summary.get('artifact_cell_count', 0) or 0), 'segmentation_status': 'complete', 'algorithm_fingerprint': cp.get('algorithm_fingerprint', ''), 'registration_visual_qc': final_visual_qc or src.get('registration_visual_qc', 'PASS')}
        layer_rows.append(row)
    layer_rows.sort(key=lambda r: int(r['layer_index']))
    manifest_path = out_root / 'layer_matching_manifest.tsv'
    atomic_write_tsv(manifest_path, layer_rows)
    artifact_rows = exclusion_manifest_rows(layer_rows, payloads)
    artifact_path = out_root / 'artifact_exclusion_manifest.tsv'
    atomic_write_tsv(artifact_path, artifact_rows)
    dataset_root = root / '05_dataset'
    try:
        schemas = pyarrow_schemas()
    except ModuleNotFoundError:
        schemas = {}
    layers_for_dataset = [layer_summary_from_segmentation_row(r) for r in layer_rows]
    layers_out = dataset_root / 'layers.parquet'
    mode = write_table(layers_out, layers_for_dataset, schema=schemas.get('layers'), prefer_parquet=prefer_parquet)
    summary_out = {'status': 'PASS' if len(layer_rows) == len(source_rows) and (not missing_source) else 'PARTIAL', 'source_rows': len(source_rows), 'cellpose_rows': len(cp_rows), 'joined_rows': len(layer_rows), 'missing_source_count': len(missing_source), 'artifact_cell_count': sum((int(r['artifact_cell_count']) for r in layer_rows)), 'raw_cell_count': sum((int(r['raw_cell_count']) for r in layer_rows)), 'valid_cell_count': sum((int(r['valid_cell_count']) for r in layer_rows)), 'layer_matching_manifest_path': str(manifest_path), 'artifact_exclusion_manifest_path': str(artifact_path), 'layers_dataset_path': str(layers_out if mode == 'parquet' else layers_out.with_suffix('.tsv')), 'write_mode': mode}
    atomic_write_json(out_root / 'artifact_filter_summary.json', summary_out)
    return summary_out

def _read_nucleus_ids_for_index(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in layer_rows:
        item = dict(row)
        path = item.get('nucleus_instances_path')
        if path:
            try:
                payload = read_gzip_json(path)
                ids = []
                for inst in payload.get('instances', []) or []:
                    raw = inst.get('instance_id', inst.get('nucleus_id'))
                    if raw not in {None, ''}:
                        ids.append(int(raw))
                item['nucleus_ids'] = sorted(set(ids))
            except Exception:
                pass
        enriched.append(item)
    return enriched

def write_code_lookups(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root)
    rows_quality = [{'code': i, 'quality_class': v} for i, v in enumerate(QUALITY_CLASSES)]
    rows_cell = [{'code': i, 'cell_relation': v} for i, v in enumerate(CELL_RELATIONS)]
    out = root / '05_dataset' / 'lookups'
    atomic_write_tsv(out / 'quality_class_lookup.tsv', rows_quality)
    atomic_write_tsv(out / 'cell_relation_lookup.tsv', rows_cell)
    atomic_write_json(out / 'dataset_code_lookups.json', {'quality_classes': QUALITY_CLASSES, 'cell_relations': CELL_RELATIONS})
    return {'status': 'PASS', 'lookup_dir': str(out)}

def write_sha256_inventory(result_root: str | Path, *, include_suffixes: tuple[str, ...]=('.json', '.tsv', '.parquet', '.md', '.png')) -> dict[str, Any]:
    root = Path(result_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.name.startswith('.'):
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        try:
            digest = sha256_file(path)
        except OSError:
            continue
        rows.append({'relative_path': str(path.relative_to(root)).replace('\\', '/'), 'size_bytes': path.stat().st_size, 'sha256': digest})
    out = root / '06_audit' / 'sha256_inventory.tsv'
    atomic_write_tsv(out, rows, fields=['relative_path', 'size_bytes', 'sha256'])
    return {'status': 'PASS', 'path': str(out), 'file_count': len(rows), 'sha256': sha256_file(out)}

def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return read_tsv(path)

def compact_match_outputs(*, result_root: str | Path, prefer_parquet: bool=True) -> dict[str, Any]:
    root = Path(result_root)
    match_root = root / '04_cross_layer_matches'
    dataset_root = root / '05_dataset'
    link_files = sorted((match_root / 'nucleus_links').glob('bucket_*/*.parquet')) or sorted((match_root / 'nucleus_links').glob('bucket_*/*.tsv'))
    audit_files = sorted((match_root / 'audit_links').glob('bucket_*/*.parquet')) or sorted((match_root / 'audit_links').glob('bucket_*/*.tsv'))
    try:
        schemas = pyarrow_schemas()
    except ModuleNotFoundError:
        schemas = {}
    compact_summary: dict[str, Any] = {}
    links_compact = dataset_root / 'nucleus_links' / 'part-000.parquet'
    if prefer_parquet:
        compact_summary['nucleus_links'] = compact_partitioned_rows(link_files, links_compact, schema=schemas.get('nucleus_links'))
        audit_compact = root / '06_audit' / 'audit_links.parquet'
        compact_summary['audit_links'] = compact_partitioned_rows(audit_files, audit_compact, schema=None)
        link_rows = _read_table_rows(links_compact)
    else:
        link_rows = []
        for f in link_files:
            link_rows.extend(_read_table_rows(f))
        write_table(links_compact.with_suffix('.tsv'), link_rows, prefer_parquet=False)
        compact_summary['nucleus_links'] = {'status': 'PASS', 'rows': len(link_rows), 'output_path': str(links_compact.with_suffix('.tsv'))}
    accepted_link_rows = [r for r in link_rows if parse_bool(r.get('accepted'))]
    assert_endpoint_degree_at_most_one(accepted_link_rows)
    layer_rows = read_tsv(root / '03_instance_relations/layer_matching_manifest.tsv')
    index_rows = build_nucleus_link_index(_read_nucleus_ids_for_index(layer_rows), accepted_link_rows)
    index_out = dataset_root / 'nucleus_link_index' / 'part-000.parquet'
    index_mode = write_table(index_out, index_rows, schema=None, prefer_parquet=prefer_parquet)
    edge_manifest = read_tsv(root / '04_cross_layer_matches/00_manifests/edge_job_manifest.tsv')
    counts_by_edge: dict[str, Counter[str]] = defaultdict(Counter)
    for row in accepted_link_rows:
        eid = str(row['edge_id'])
        counts_by_edge[eid][str(row.get('quality_class', ''))] += 1
        if row.get('cell_relation') == 'conflict':
            counts_by_edge[eid]['cell_conflict'] += 1
    audit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for f in audit_files:
        for row in _read_table_rows(f):
            audit_counts[str(row.get('edge_id', ''))][str(row.get('quality_class', ''))] += 1
    edge_file_sha: dict[str, str] = {}
    for f in link_files + audit_files:
        edge_file_sha.setdefault(f.stem, '')
        try:
            edge_file_sha[f.stem] = sha256_text(edge_file_sha[f.stem] + sha256_file(f))
        except OSError:
            pass
    edge_rows = []
    for e in edge_manifest:
        eid = e['edge_id']
        c = counts_by_edge.get(eid, Counter())
        ac = audit_counts.get(eid, Counter())
        edge_rows.append({'edge_id': eid, 'optimized_track_id': e.get('optimized_track_id', ''), 'source_track_id': e.get('source_track_id', ''), 'left_layer_index': int(e['left_layer_index']), 'right_layer_index': int(e['right_layer_index']), 'left_section_id': int(e['left_section_id']), 'right_section_id': int(e['right_section_id']), 'selection_origin': e.get('selection_origin', ''), 'median_tre_um': float(e.get('median_tre_um') or 0.0), 'p90_tre_um': float(e.get('p90_tre_um') or 0.0), 'jacobian_min': float(e.get('jacobian_min') or 0.0), 'jacobian_p1': float(e.get('jacobian_p1') or 0.0), 'jacobian_p99': float(e.get('jacobian_p99') or 0.0), 'inverse_consistency_p90_um': float(e.get('inverse_consistency_p90_um') or 0.0), 'matching_status': 'complete' if eid in counts_by_edge or eid in audit_counts else 'missing', 'gold_count': int(c['gold']), 'silver_nucleus_count': int(c['silver_nucleus']), 'unknown_count': int(ac['unknown']), 'cell_only_count': int(ac['cell_only']), 'cell_conflict_count': int(c['cell_conflict']), 'edge_output_sha256': edge_file_sha.get(eid, '')})
    edges_out = dataset_root / 'edges.parquet'
    edges_mode = write_table(edges_out, edge_rows, schema=schemas.get('edges'), prefer_parquet=prefer_parquet)
    lookups = write_code_lookups(root)
    inventory = write_sha256_inventory(root)
    final = {'status': 'PASS' if all((r['matching_status'] == 'complete' for r in edge_rows)) else 'PARTIAL', 'edge_rows': len(edge_rows), 'complete_edges': sum((r['matching_status'] == 'complete' for r in edge_rows)), 'link_rows': len(link_rows), 'index_rows': len(index_rows), 'gold_count': sum((int(r['gold_count']) for r in edge_rows)), 'silver_nucleus_count': sum((int(r['silver_nucleus_count']) for r in edge_rows)), 'cell_conflict_count': sum((int(r['cell_conflict_count']) for r in edge_rows)), 'compact': compact_summary, 'index_output': str(index_out if index_mode == 'parquet' else index_out.with_suffix('.tsv')), 'edges_output': str(edges_out if edges_mode == 'parquet' else edges_out.with_suffix('.tsv')), 'lookups': lookups, 'sha256_inventory': inventory}
    atomic_write_json(root / '06_audit/final_dataset_summary.json', final)
    return final

def validate_final_dataset(*, result_root: str | Path, expected_layers: int=11016, expected_edges: int=8972) -> dict[str, Any]:
    root = Path(result_root)
    dataset_root = root / '05_dataset'
    errors: list[str] = []
    warnings: list[str] = []

    def require(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)
    summary_path = root / '06_audit/final_dataset_summary.json'
    summary = read_json(summary_path) if summary_path.exists() else {}
    layers_path = dataset_root / 'layers.parquet'
    edges_path = dataset_root / 'edges.parquet'
    links_path = dataset_root / 'nucleus_links' / 'part-000.parquet'
    index_path = dataset_root / 'nucleus_link_index' / 'part-000.parquet'
    audit_path = root / '06_audit' / 'audit_links.parquet'
    for required in [layers_path, edges_path, links_path, index_path, audit_path]:
        require(required.exists(), f'missing required dataset artifact: {required}')
    layers = _read_table_rows(layers_path) if layers_path.exists() else []
    edges = _read_table_rows(edges_path) if edges_path.exists() else []
    links = _read_table_rows(links_path) if links_path.exists() else []
    index_rows = _read_table_rows(index_path) if index_path.exists() else []
    audit_rows = _read_table_rows(audit_path) if audit_path.exists() else []
    require(len(layers) == expected_layers, f'layers row count {len(layers)} != expected {expected_layers}')
    require(len(edges) == expected_edges, f'edges row count {len(edges)} != expected {expected_edges}')
    require(all((str(r.get('registration_visual_qc', '')) == 'PASS' for r in layers)), 'not all layers have registration_visual_qc=PASS')
    require(all((str(r.get('segmentation_status', '')) == 'complete' for r in layers)), 'not all layers have segmentation_status=complete')
    require(all((str(r.get('matching_status', '')) == 'complete' for r in edges)), 'not all edges have matching_status=complete')
    link_quality = Counter((str(r.get('quality_class', '')) for r in links))
    audit_quality = Counter((str(r.get('quality_class', '')) for r in audit_rows))
    bad_quality = set(link_quality) - {'gold', 'silver_nucleus'}
    require(not bad_quality, f'nucleus_links contain non-positive quality classes: {sorted(bad_quality)}')
    require('cell_only' not in link_quality, 'cell_only found in accepted nucleus_links')
    require('unknown' not in link_quality, 'unknown found in accepted nucleus_links')
    require(all((parse_bool(r.get('accepted')) for r in links)), 'nucleus_links contains accepted=false rows')
    require(all((parse_bool(r.get('nucleus_supervision_valid')) for r in links)), 'nucleus_links contains nucleus_supervision_valid=false rows')
    bad_cell_supervision = 0
    bad_conflict = 0
    for r in links:
        rel = str(r.get('cell_relation', ''))
        valid = parse_bool(r.get('cell_supervision_valid'))
        if valid != (rel == 'agree'):
            bad_cell_supervision += 1
        if rel == 'conflict' and valid:
            bad_conflict += 1
    require(bad_cell_supervision == 0, f'cell_supervision_valid is inconsistent with cell_relation for {bad_cell_supervision} links')
    require(bad_conflict == 0, f'cell_relation=conflict has cell_supervision_valid=true for {bad_conflict} links')
    try:
        assert_endpoint_degree_at_most_one(links)
    except Exception as exc:
        errors.append(f'endpoint degree violation: {exc}')
    expected_index = sum((int(r.get('nucleus_count', 0) or 0) for r in layers))
    require(len(index_rows) == expected_index, f'nucleus_link_index rows {len(index_rows)} != total nucleus_count {expected_index}')
    lookup_dir = dataset_root / 'lookups'
    for required in [lookup_dir / 'quality_class_lookup.tsv', lookup_dir / 'cell_relation_lookup.tsv', lookup_dir / 'dataset_code_lookups.json']:
        require(required.exists(), f'missing lookup file: {required}')
    require((root / '06_audit' / 'sha256_inventory.tsv').exists(), 'missing sha256 inventory')
    if not audit_rows:
        warnings.append('audit_links is empty; this is unusual but not a hard failure')
    result = {'status': 'PASS' if not errors else 'FAIL', 'error_count': len(errors), 'warning_count': len(warnings), 'errors': errors, 'warnings': warnings, 'layers': len(layers), 'edges': len(edges), 'complete_edges': sum((str(r.get('matching_status', '')) == 'complete' for r in edges)), 'nucleus_links': len(links), 'nucleus_link_index_rows': len(index_rows), 'audit_rows': len(audit_rows), 'quality_counts': dict(link_quality), 'audit_quality_counts': dict(audit_quality), 'cell_conflict_links': sum((1 for r in links if str(r.get('cell_relation', '')) == 'conflict')), 'artifact_cells': sum((int(r.get('artifact_cell_count', 0) or 0) for r in layers)), 'summary_status': summary.get('status', '')}
    atomic_write_json(root / '06_audit' / 'final_dataset_validation.json', result)
    return result

def write_dataset_readme(*, result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root)
    readme = root / '05_dataset/README_Serial_CROSS_LAYER_DATASET.md'
    text = '# RSG-6 cross-layer nucleus matching dataset\n\n' + f'Dataset root:\n\n```text\n{root}\n```\n\n' + 'Core tables:\n\n- `05_dataset/layers.parquet`: one row per registered layer, including segmentation paths and artifact-filter counts.\n- `05_dataset/edges.parquet`: one row per delivered accepted adjacent edge.\n- `05_dataset/nucleus_links/`: accepted `gold` and `silver_nucleus` nucleus links.\n- `05_dataset/nucleus_link_index/`: per-nucleus lower/upper neighbor lookup.\n\nSemantics:\n\n- Artifact filter: `assigned_nucleus_count == 0 AND cell_area_px > 816`.\n- Cell evidence never vetoes nucleus matches.\n- `cell_relation=conflict` keeps `nucleus_supervision_valid=true` and sets `cell_supervision_valid=false`.\n- `cell_only` and `unknown` are audit-only, not nucleus positives or negatives.\n\nExample:\n\n```python\nimport pyarrow.parquet as pq\nroot = r"' + str(root) + '"\nlayers = pq.read_table(root + "/05_dataset/layers.parquet")\nedges = pq.read_table(root + "/05_dataset/edges.parquet")\nlinks = pq.read_table(root + "/05_dataset/nucleus_links")\nindex = pq.read_table(root + "/05_dataset/nucleus_link_index")\n```\n'
    readme.parent.mkdir(parents=True, exist_ok=True)
    tmp = readme.with_name(f'.{readme.name}.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(readme)
    return {'status': 'PASS', 'path': str(readme), 'sha256': sha256_file(readme)}
