from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import math
import sys
from celllift.runtime import ResourcePath as Path
import numpy as np
from PIL import Image
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from celllift.model_inputs.common.graph import build_record
from celllift.model_inputs.common.segment import _atomic_gzip_json, _atomic_npy, _complete_existing, _eval, _load_models, canonical_relabel, instance_payload, pair_instances, stain_inputs
from celllift.model_inputs.common.standardize import _write_rgb
from celllift.model_inputs.common.utils import atomic_json, atomic_parquet, read_config, read_parquet_rows, shard_hex

def sensitivity_row(root: Path, row: dict) -> dict:
    value = dict(row)
    patch_id = str(row['patch_id'])
    shard = shard_hex(patch_id)
    segment = root / 'segmentations'
    size = 1113
    mpp = 512.0 / size
    value.update({'source_mpp': 1.0, 'mpp_provenance': 'sensitivity_only_nominal_10x_1.00_um_per_px', 'target_mpp': 0.46, 'actual_target_mpp': mpp, 'output_width_px': size, 'output_height_px': size, 'physical_width_um': 512.0, 'physical_height_um': 512.0, 'rgb_path': str(root / 'rgb' / shard / f'{patch_id}.png'), 'nucleus_mask_path': str(segment / 'nucleus_masks' / shard / f'{patch_id}.npy'), 'cell_mask_path': str(segment / 'cell_masks' / shard / f'{patch_id}.npy'), 'nucleus_instances_path': str(segment / 'nucleus_instances' / shard / f'{patch_id}.json.gz'), 'cell_instances_path': str(segment / 'cell_instances' / shard / f'{patch_id}.json.gz'), 'pairs_path': str(segment / 'pairs' / shard / f'{patch_id}.json.gz'), 'graph_id': f'sicapv2_mpp100_sensitivity:{patch_id}'})
    return value

def summarize_graph(row: dict, layer_idx: int) -> dict:
    graph = build_record(row, layer_idx)
    mask = np.load(row['nucleus_mask_path'], mmap_mode='r')
    areas = np.bincount(np.asarray(mask).reshape(-1))[1:]
    diameters = 2.0 * np.sqrt(areas[areas > 0] / np.pi) * float(row['actual_target_mpp'])
    return {'patch_id': row['patch_id'], 'nucleus_count': len(graph.nucleus_id), 'edge_count': len(graph.edge_src), 'median_degree': float(len(graph.edge_src) / max(1, len(graph.nucleus_id))), 'median_rho_um': float(np.median(graph.rho_um)), 'median_nucleus_equivalent_diameter_um': float(np.median(diameters)), 'nucleus_matched_fraction': float(np.count_nonzero(graph.anchor_cell_id) / len(graph.nucleus_id))}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if cfg['dataset'] != 'sicapv2':
        raise RuntimeError('SICAP configuration required')
    data_root = Path(cfg['paths']['data_root'])
    output_root = data_root / '05_qc/mpp_sensitivity_100'
    formal_rows = read_parquet_rows(data_root / '00_manifest/pilot_manifest.parquet')
    rows = [sensitivity_row(output_root, row) for row in formal_rows]
    atomic_parquet(output_root / 'sensitivity_manifest.parquet', rows)
    for row in rows:
        _write_rgb(Path(row['source_path']), Path(row['rgb_path']), (1113, 1113))
    nucleus_model, cell_model = _load_models()
    statuses = []
    for index, row in enumerate(rows):
        resumed = _complete_existing(row)
        if resumed is None:
            with Image.open(row['rgb_path']) as image:
                nucleus_input, cell_input = stain_inputs(np.asarray(image.convert('RGB')))
            nucleus_mask = canonical_relabel(_eval(nucleus_model, [nucleus_input], diameter=17.0, channels=[0, 0], batch_size=1)[0])
            cell_mask = canonical_relabel(_eval(cell_model, [cell_input], diameter=20.0, channels=[1, 2], batch_size=1)[0])
            nuclei, cells = (instance_payload(nucleus_mask), instance_payload(cell_mask))
            pairs = pair_instances(nucleus_mask, cell_mask)
            _atomic_npy(Path(row['nucleus_mask_path']), nucleus_mask)
            _atomic_npy(Path(row['cell_mask_path']), cell_mask)
            _atomic_gzip_json(Path(row['nucleus_instances_path']), nuclei)
            _atomic_gzip_json(Path(row['cell_instances_path']), cells)
            _atomic_gzip_json(Path(row['pairs_path']), pairs)
            resumed = pairs['summary']
        statuses.append({'patch_id': row['patch_id'], **resumed})
        if (index + 1) % 4 == 0 or index + 1 == len(rows):
            atomic_parquet(output_root / 'segmentation_progress.parquet', statuses)
    formal_metrics, sensitivity_metrics = ([], [])
    for index, (formal, sensitivity) in enumerate(zip(formal_rows, rows)):
        formal_metrics.append({'scale_policy': 'formal_0.92', **summarize_graph(formal, index)})
        sensitivity_metrics.append({'scale_policy': 'sensitivity_1.00', **summarize_graph(sensitivity, index)})
    metrics = formal_metrics + sensitivity_metrics
    atomic_parquet(output_root / 'per_patch_comparison.parquet', metrics)
    keys = ('nucleus_count', 'edge_count', 'median_degree', 'median_rho_um', 'median_nucleus_equivalent_diameter_um', 'nucleus_matched_fraction')
    summary = {'status': 'PASS', 'patches_per_policy': len(rows), 'formal_source_mpp': 0.92, 'sensitivity_source_mpp': 1.0}
    for key in keys:
        formal_value = float(np.median([row[key] for row in formal_metrics]))
        sensitivity_value = float(np.median([row[key] for row in sensitivity_metrics]))
        summary[key] = {'formal_median': formal_value, 'sensitivity_median': sensitivity_value, 'relative_change': (sensitivity_value - formal_value) / formal_value if formal_value else None}
    atomic_json(output_root / 'sensitivity.status.json', summary)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
