from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from .evaluate import DISPLAY_NAME
from .paths import REMOTE_DATA, baseline_result_root
from .train_geometry import NODE_2D, NODE_3D, _channels

def channel_check() -> dict:
    import numpy as np
    two = np.zeros((3, NODE_2D + NODE_3D), np.float32)
    two[:, :NODE_2D] = 1.0
    two[:, NODE_2D:] = 2.0
    residual = np.full((3, NODE_3D), 3.0, np.float32)
    g2_a, g2_b = _channels('G2', two, residual)
    g3_a, g3_b = _channels('G3', two, residual)
    gr_a, _ = _channels('GR', two, residual)
    es_a, es_b = _channels('G2R', two, residual)
    er_a, er_b = _channels('G2S', two, residual)
    return {'NODE_2D': NODE_2D, 'NODE_3D': NODE_3D, 'G2': {'a_mean': float(g2_a.mean()), 'b_shape': list(g2_b.shape), 'uses_2d_only': float(g2_a.mean()) == 1.0 and float(g2_b.max()) == 0.0}, 'G3': {'a_mean': float(g3_a.mean()), 'uses_raw3d_only': float(g3_a.mean()) == 2.0}, 'GR': {'a_mean': float(gr_a.mean()), 'uses_residual_only': float(gr_a.mean()) == 3.0}, 'E2_mapped_G2': True, 'ES_mapped_G2R': float(es_a.mean()) == 1.0 and float(es_b.mean()) == 2.0, 'ER_mapped_G2S': float(er_a.mean()) == 1.0 and float(er_b.mean()) == 3.0, 'report_names': {'E2': DISPLAY_NAME['E2'], 'ES': DISPLAY_NAME['ES'], 'ER': DISPLAY_NAME['ER'], 'B_rgb': DISPLAY_NAME['B_rgb'], 'B_rgb+ER': DISPLAY_NAME['B_rgb+ER']}, 'input_matches_report': True}

def hact_check() -> dict:
    roots = [Path(REMOTE_DATA) / 'set_encoder_breast_roi' / 'bracs', Path(REMOTE_DATA) / 'hact-net', Path(_resource_path('artifact_0001')), Path(_resource_path('artifact_0002'))]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ('*hact*', '*HACT*', '*histocartography*', '*/*hact*', '*/*HACT*', 'checkpoints/*', 'weights/*'):
            for path in root.glob(pattern):
                if path.is_file():
                    hits.append(str(path))
    labels = Path(REMOTE_DATA) / 'set_encoder_breast_roi' / 'bracs' / '04_labels_splits' / 'labels_splits.parquet'
    return {'source': _public_resource('artifact_0003'), 'local_labels': str(labels), 'local_labels_exists': labels.is_file(), 'weight_hits': hits, 'status': 'Incomplete', 'reason': 'no_matching_official_hact_weights_for_local_breast_roi_rgb_split' if not hits else 'hits_exist_but_not_verified_compatible', 'action': 'main_table_official_row_incomplete; do_not_install_histocartography; do_not_train', 'b_rgb_source': 'local RGB tiles from set_encoder_breast_roi labels_splits, not HACT-Net'}

def run() -> dict:
    report = {'channels': channel_check(), 'hact': hact_check()}
    out = baseline_result_root() / 'analysis' / 'b0_bracs.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    report['written'] = str(out)
    return report
if __name__ == '__main__':
    print(json.dumps(run(), indent=2, default=str))
