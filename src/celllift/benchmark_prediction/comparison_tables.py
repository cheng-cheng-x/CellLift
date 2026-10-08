from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from .evaluate import _lookup_test_metrics, _score
from .io import job_dir
from .paths import baseline_result_root, comparison_result_root, result_root

def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))

def write_tables() -> dict:
    stats = _json(comparison_result_root() / 'analysis' / 'stats.json')
    fusion = _json(comparison_result_root() / 'analysis' / 'sicap_fusion.json')
    audit = _json(comparison_result_root() / 'analysis' / 'sicap_audit.json')
    hact = _json(comparison_result_root() / 'analysis' / 'hact_metrics.json')
    geom = (stats.get('geometry') or {}).get('deltas') or {}
    p2 = (stats.get('crc_p2') or {}).get('deltas') or {}

    def g(dataset, pair):
        return geom.get(f'{dataset}:{pair}') or {}
    table_a = [{'dataset': 'sicapv2', 'metric': 'qwk', 'G2': g('sicapv2', 'G3-G2').get('a'), 'G3': g('sicapv2', 'G3-G2').get('b'), 'G3_minus_G2': g('sicapv2', 'G3-G2'), 'GR_minus_G2': g('sicapv2', 'GR-G2'), 'cluster': 'patient'}, {'dataset': 'bracs', 'metric': 'macro_f1', 'G2': g('bracs', 'G3-G2').get('a'), 'G3': g('bracs', 'G3-G2').get('b'), 'G3_minus_G2': g('bracs', 'G3-G2'), 'GR_minus_G2': g('bracs', 'GR-G2'), 'cluster': 'wsi'}, {'dataset': 'tcga_crc_msi', 'metric': 'patient_auroc', 'G2': g('tcga_crc_msi', 'G3-G2').get('a'), 'G3': g('tcga_crc_msi', 'G3-G2').get('b'), 'G3_minus_G2': g('tcga_crc_msi', 'G3-G2'), 'GR_minus_G2': g('tcga_crc_msi', 'GR-G2'), 'cluster': 'patient'}]
    paper = _json(baseline_result_root() / 'sicapv2/baseline/B_paper200/full_train/seed_42/fold_00/test_metrics.json')
    a2 = _json(result_root() / 'sicapv2/route/A2/full_train/seed_42/fold_00/test_metrics.json')
    as_ = _json(result_root() / 'sicapv2/route/AS/full_train/seed_42/fold_00/test_metrics.json')
    primary = fusion.get('primary_3d') or 'B+G3'
    sicap_f2 = (fusion.get('test') or {}).get('B+G2') or {}
    sicap_f3 = (fusion.get('test') or {}).get(primary) or {}
    bracs_b = _json(result_root() / 'bracs/baseline/B_rgb/official/seed_42/fold_00/test_metrics.json')
    bracs_f2 = _json(result_root() / 'bracs/fusion/B_rgb+E2/official/seed_42/fold_00/test_metrics.json')
    bracs_f3 = _json(result_root() / 'bracs/fusion/B_rgb+ER/official/seed_42/fold_00/test_metrics.json')
    bracs_h2 = _json(result_root() / 'bracs/feature/H2/official/seed_42/fold_00/test_metrics.json')
    table_b = [{'dataset': 'sicapv2', 'B': {'display': 'B_paper200', 'metrics': paper, 'score': paper.get('qwk')}, 'F2': {'display': 'B+G2', 'metrics': sicap_f2.get('metrics'), 'score': (sicap_f2.get('metrics') or {}).get('qwk')}, 'F3': {'display': primary, 'metrics': sicap_f3.get('metrics'), 'score': (sicap_f3.get('metrics') or {}).get('qwk')}, 'strong_2d': {'display': 'A2', 'score': a2.get('qwk')}, 'AS': as_.get('qwk'), 'primary_3d': primary, 'audit': audit.get('disposition')}, {'dataset': 'bracs', 'B': {'display': 'B_rgb (local RGB)', 'metrics': bracs_b, 'score': bracs_b.get('macro_f1')}, 'F2': {'display': 'B_rgb+Geo2D', 'metrics': bracs_f2, 'score': bracs_f2.get('macro_f1')}, 'F3': {'display': 'B_rgb+Geo2D+res3D', 'metrics': bracs_f3, 'score': bracs_f3.get('macro_f1')}, 'strong_2d': {'display': 'H2', 'score': bracs_h2.get('macro_f1')}, 'official_hact': hact or {'status': 'Incomplete'}}, {'dataset': 'tcga_crc_msi', 'B': {'display': 'P2 (B2)', 'score': (p2.get('P2+G3-P2') or {}).get('a')}, 'F2': {'display': 'P2+G2', 'score': (p2.get('P2+G3-P2+G2') or {}).get('a')}, 'F3': {'display': 'P2+G3', 'delta_vs_P2': p2.get('P2+G3-P2'), 'delta_vs_F2': p2.get('P2+G3-P2+G2')}, 'P2+GR': {'delta_vs_P2': p2.get('P2+GR-P2'), 'delta_vs_F2': p2.get('P2+GR-P2+G2')}, 'exploratory_after_seeing_test': True, 'isolation': stats.get('crc_isolation')}]
    payload = {'table_a': table_a, 'table_b': table_b, 'seed': 42, 'protocol': 'paired_analysis_seed42'}
    dest = comparison_result_root() / 'tables' / 'main_tables.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')
    payload['written'] = str(dest)
    return payload
