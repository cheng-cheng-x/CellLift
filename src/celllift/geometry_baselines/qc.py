from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .io_utils import atomic_json, atomic_parquet
from .protocol import PROTOCOL_ID, require_test_gate

def compute_ncr_qc(cfg: Mapping[str, Any], *, include_test: bool=False) -> dict[str, Any]:
    if include_test:
        require_test_gate(cfg['paths']['result_root'])
    import pyarrow.parquet as pq
    graph_table = pq.read_table(Path(cfg['paths']['model_input_root']) / '03_graph_cache' / 'graph_index.parquet', columns=['graph_id', 'split', 'label_id'], filters=None if include_test else [('split', 'in', ['TRAIN', 'train', 'Train'])], partitioning=None)
    allowed = {'train', 'test'} if include_test else {'train'}
    metadata = {str(graph): (str(split).lower(), int(label)) for graph, split, label in zip(graph_table['graph_id'].to_pylist(), graph_table['split'].to_pylist(), graph_table['label_id'].to_numpy(zero_copy_only=False)) if str(split).lower() in allowed}
    keys = sorted(set(metadata.values()))
    key_index = {key: index for index, key in enumerate(keys)}
    graph_code = {graph: key_index[value] for graph, value in metadata.items()}
    totals = np.zeros(len(keys), np.int64)
    invalids = np.zeros(len(keys), np.int64)
    seen_graphs: set[str] = set()
    root = Path(cfg['paths']['set_encoding_data_root']) / '04_ncr_features'
    for file in sorted(root.glob('*.parquet')):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=['graph_id', 'valid_ncr_3d'], batch_size=200000):
            graphs = batch.column(0).to_pylist()
            valid = np.asarray(batch.column(1).to_numpy(zero_copy_only=False), bool)
            codes = np.fromiter((graph_code.get(str(graph), -1) for graph in graphs), dtype=np.int32, count=len(graphs))
            selected = codes >= 0
            if selected.any():
                totals += np.bincount(codes[selected], minlength=len(keys))
                invalids += np.bincount(codes[selected], weights=(~valid[selected]).astype(np.int8), minlength=len(keys)).astype(np.int64)
                seen_graphs.update((str(graphs[index]) for index in np.flatnonzero(selected)))
    if seen_graphs != set(metadata):
        missing = sorted(set(metadata) - seen_graphs)
        raise RuntimeError(f'NCR QC graph coverage mismatch: {len(missing)} missing, e.g. {missing[:3]}')
    rows = []
    for (split, label), anchors, invalid in zip(keys, totals.tolist(), invalids.tolist()):
        rows.append({'split': split, 'label_id': label, 'anchors': anchors, 'invalid_ncr3d': invalid, 'invalid_fraction': invalid / anchors if anchors else None})
    total = sum((row['anchors'] for row in rows))
    invalid = sum((row['invalid_ncr3d'] for row in rows))
    phase = 'train_test' if include_test else 'train_only'
    output = Path(cfg['paths']['result_root']) / 'metrics' / 'qc' / f'ncr_invalid_{phase}'
    atomic_parquet(output.with_suffix('.parquet'), rows)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'phase': phase, 'test_gate_required': include_test, 'graphs': len(seen_graphs), 'anchors': total, 'invalid_ncr3d': invalid, 'invalid_fraction': invalid / total if total else None, 'by_split_class': rows}
    atomic_json(output.with_suffix('.json'), payload)
    return payload
