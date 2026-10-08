from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from .utils import read_parquet_rows, stable_shard

class PublicGraphInferenceDataset:

    def __init__(self, graph_index: str | Path, graph_root: str | Path, training_code_root: str | Path, *, shards: int=64, feature_stats: str | Path | None=None):
        code_root = str(Path(training_code_root))
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        from celllift.graph_input_schema.dataset import GraphInferenceSample, collate_inference_samples
        from celllift.graph_input_schema.schemas import GraphRecord
        self.GraphInferenceSample = GraphInferenceSample
        self.GraphRecord = GraphRecord
        self.collate = collate_inference_samples
        self.rows = read_parquet_rows(graph_index)
        self.root = Path(graph_root)
        self.shards = int(shards)
        self._envs = None
        stats_path = Path(feature_stats or self.root / 'inference_feature_stats.json')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        self.mean = np.asarray(stats['mean'], np.float32)
        self.std = np.asarray(stats['std'], np.float32)
        if self.mean.shape != (36,) or self.std.shape != (36,) or np.any(self.std <= 0):
            raise RuntimeError('invalid frozen training feature statistics')

    def _open(self):
        if self._envs is None:
            import lmdb
            self._envs = [lmdb.open(str(self.root / f'graph_cache_{index:02d}.lmdb'), subdir=True, readonly=True, lock=False, readahead=False, max_readers=2048) for index in range(self.shards)]
        return self._envs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        graph_id = str(row['graph_id'])
        env = self._open()[stable_shard(graph_id, self.shards)]
        with env.begin(buffers=False) as transaction:
            payload = transaction.get(graph_id.encode())
        if payload is None:
            raise KeyError(graph_id)
        graph = self.GraphRecord.from_bytes(bytes(payload))
        zero = np.zeros(len(graph.nucleus_id), np.int64)
        return self.GraphInferenceSample(graph, zero.copy(), zero.copy(), zero.copy(), zero.copy(), self.mean, self.std)

    def label(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {'graph_id': row['graph_id'], 'label_id': int(row['label_id']), 'label_name': row['label_name'], 'label_scope': row['label_scope'], 'patient_id': row['patient_id'], 'validation_fold': row.get('validation_fold')}

    def sample_cost(self, index: int) -> dict[str, int]:
        row = self.rows[index]
        return {'nodes': int(row['node_count']), 'edges': int(row['edge_count']), 'projection_voxels': 0}

    def close(self) -> None:
        if self._envs:
            for env in self._envs:
                env.close()
            self._envs = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_envs'] = None
        return state
