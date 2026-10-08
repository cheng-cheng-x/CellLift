from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.common.probe_data import FoldProbeData
from celllift.conditional_geometry.common.token_store import FoldTokenStore, configure
from celllift.conditional_geometry.experiment import ARMS
from celllift.conditional_geometry.protocol import RESIDUAL_TOKEN_DIM

def downstream_store_smoke(cfg: Mapping[str, Any], fold: int, graph_limit: int=64, seed: int=42) -> dict[str, Any]:
    import pyarrow.parquet as pq
    data = FoldProbeData(cfg, fold, graph_limit=graph_limit)
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
            raise RuntimeError('official TEST entered downstream smoke')
    root = Path(cfg['paths']['data_root']) / 'smoke' / f'graph_limit_{graph_limit}' / 'manifest'
    root.mkdir(parents=True, exist_ok=True)
    destination = root / 'downstream_manifest.parquet'
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(graph_rows), temporary, compression='zstd')
    os.replace(temporary, destination)
    partition = {graph: str(data.roles[data.graph_index[graph]]) for graph in graph_ids}
    configure(cfg, graph_limit)
    store = FoldTokenStore(cfg, fold, seed, destination, graph_ids, partition)
    audit: dict[str, Any] = {}
    for arm_name, arm in ARMS.items():
        samples = store.samples(arm, graph_ids[:2])
        n = np.concatenate([sample.nucleus_tokens for sample in samples])
        c = np.concatenate([sample.cell_tokens for sample in samples])
        if n.shape[1] != RESIDUAL_TOKEN_DIM or c.shape[1] != RESIDUAL_TOKEN_DIM:
            raise AssertionError('token width changed')
        if not np.array_equal(n[:, 15], c[:, 15]):
            raise AssertionError('shared residual NCR differs between object branches')
        if arm_name == 'conditional_geometry_2D' and (np.any(n[:, 11:] != 0) or np.any(c[:, 11:] != 0)):
            raise AssertionError('conditional_geometry_2D residual channels are not exact zero')
        audit[arm_name] = {'graphs': len(samples), 'anchors': len(n), 'finite': bool(np.isfinite(n).all() and np.isfinite(c).all())}
    mapping_sample = store.shuffle_rows[:min(10000, len(store.shuffle_rows))]
    cross_graph = all((row.graph_id != row.donor_graph_id for row in mapping_sample))
    same_decile = all((row.count_decile == row.donor_count_decile for row in mapping_sample))
    if not cross_graph or not same_decile:
        raise AssertionError('shuffle donor violates cross-graph/same-decile constraints')
    return {'status': 'PASS', 'dataset': cfg['dataset'], 'fold': fold, 'graphs': len(graph_ids), 'official_test_touched': False, 'arms': audit, 'shuffle_rows': len(store.shuffle_rows)}
