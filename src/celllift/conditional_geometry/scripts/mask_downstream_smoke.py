from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData
from celllift.conditional_geometry.common.mask_token_store import MaskFoldTokenStore, configure
from celllift.conditional_geometry.mask_experiment import ARMS
from celllift.conditional_geometry.mask_protocol import MASK_RESIDUAL_TOKEN_DIM, RAY_DIM

def mask_downstream_store_smoke(cfg: Mapping[str, Any], fold: int, graph_limit: int=64, seed: int=42) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rgb_mode = str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays'
    if rgb_mode:
        from celllift.conditional_geometry.common.rgb_mask_probe_data import RGBMaskFoldProbeData
        from celllift.conditional_geometry.rgb_mask_experiment import ARMS as selected_arms
        data = RGBMaskFoldProbeData(cfg, fold, graph_limit=graph_limit)
    else:
        if str(cfg.get('protocol_id')) == 'matched_mask_full3d_vs_residual3d_raw_residual_comparison':
            from celllift.conditional_geometry.full3d_experiment import ARMS as selected_arms
        else:
            selected_arms = ARMS
        data = MaskFoldProbeData(cfg, fold, graph_limit=graph_limit)
    graph_ids = list(data.graph_ids)
    graph_index = Path(cfg['paths']['model_input_root']) / '03_graph_cache' / 'graph_index.parquet'
    patch_manifest = Path(cfg['paths']['model_input_root']) / '00_manifest' / 'patch_manifest.parquet'
    filters = [('graph_id', 'in', graph_ids)]
    graph_rows = pq.read_table(graph_index, filters=filters, partitioning=None).to_pylist()
    patch_rows = {str(row['graph_id']): row for row in pq.read_table(patch_manifest, filters=filters, partitioning=None).to_pylist()}
    for row in graph_rows:
        patch = patch_rows[str(row['graph_id'])]
        for name in ('g4c_label', 'g4c_valid'):
            row[name] = patch.get(name)
        if str(row['split']).lower() != 'train':
            raise RuntimeError('official TEST entered mask downstream smoke')
    root = Path(cfg['paths']['data_root']) / 'smoke' / f'graph_limit_{graph_limit}' / 'manifest'
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'downstream_manifest.parquet'
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(graph_rows), temporary, compression='zstd')
    os.replace(temporary, destination)
    partition = {graph: str(data.roles[data.graph_index[graph]]) for graph in graph_ids}
    configure(cfg, graph_limit)
    store = MaskFoldTokenStore(cfg, fold, seed, destination, graph_ids, partition)
    audit: dict[str, Any] = {}
    for arm_name, arm in selected_arms.items():
        samples = store.samples(arm, graph_ids[:2])
        nucleus = np.concatenate([sample.nucleus_tokens for sample in samples])
        cell = np.concatenate([sample.cell_tokens for sample in samples])
        if nucleus.shape[1] != MASK_RESIDUAL_TOKEN_DIM or cell.shape != nucleus.shape:
            raise AssertionError('mask token width changed')
        if not np.array_equal(nucleus[:, :RAY_DIM], cell[:, :RAY_DIM]):
            raise AssertionError('capacity-matched branches do not have identical mask rays')
        if not np.array_equal(nucleus[:, -1], cell[:, -1]):
            raise AssertionError('shared residual NCR differs between mask branches')
        if arm_name == 'M3_MASK2D' and (np.any(nucleus[:, RAY_DIM:] != 0) or np.any(cell[:, RAY_DIM:] != 0)):
            raise AssertionError('mask-only baseline residual channels are not exact zero')
        audit[arm_name] = {'graphs': len(samples), 'anchors': len(nucleus), 'finite': bool(np.isfinite(nucleus).all() and np.isfinite(cell).all())}
    mapping_sample = store.shuffle_rows[:min(10000, len(store.shuffle_rows))]
    if not all((row.graph_id != row.donor_graph_id for row in mapping_sample)):
        raise AssertionError('mask shuffle donor is not cross-graph')
    if not all((row.count_decile == row.donor_count_decile for row in mapping_sample)):
        raise AssertionError('mask shuffle donor crosses count deciles')
    return {'status': 'PASS', 'dataset': cfg['dataset'], 'fold': fold, 'graphs': len(graph_ids), 'official_test_touched': False, 'rgb_held_constant': rgb_mode, 'arms': audit, 'shuffle_rows': len(store.shuffle_rows)}
