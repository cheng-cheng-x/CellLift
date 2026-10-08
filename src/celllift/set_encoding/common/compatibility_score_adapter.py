from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
import importlib
from celllift.runtime import json
import math
import sys
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch.utils.data import Dataset
FROZEN_RUN_ID = 'compatibility_score_iqr3stage_zdistill_seed42'
FROZEN_CHECKPOINT_SHA256 = 'c3d54ae713364c3b18cd0c6501adffd4102c490c2795e913ed9dbded5405e6fd'
EXPECTED_SAMPLE_FIELDS = ('graph', 'nucleus_observation_count', 'cell_observation_count', 'gold_observation_count', 'silver_observation_count', 'nucleus_pose_evidence', 'cell_pose_evidence', 'feature_mean', 'feature_std')
POSE_EVIDENCE_SCHEMA = {'anchor_node_index': ((0,), np.int64), 'plane_index': ((0,), np.int64), 'weight': ((0,), np.float32), 'confidence_code': ((0,), np.int64), 'target_centroid_xy_um': ((0, 2), np.float32), 'target_area_px': ((0,), np.float32)}

def sha256_file(path: str | Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def stable_shard(key: str, count: int) -> int:
    if count <= 0:
        raise ValueError('shard count must be positive')
    value = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(value, 'big') % int(count)

def empty_pose_evidence() -> dict[str, np.ndarray]:
    return {key: np.empty(shape, dtype=dtype) for key, (shape, dtype) in POSE_EVIDENCE_SCHEMA.items()}

def _import_compatibility_score(training_code_root: str | Path) -> dict[str, Any]:
    root = Path(training_code_root).resolve()
    if not (root / 'src' / 'dataset.py').is_file():
        raise FileNotFoundError(root / 'src' / 'dataset.py')
    dataset = importlib.import_module('celllift.shape_curriculum.src.dataset')
    pipeline = importlib.import_module('celllift.shape_curriculum.src.pipeline')
    predict = importlib.import_module('celllift.shape_curriculum.src.predict')
    losses = importlib.import_module('celllift.shape_curriculum.src.losses')
    schemas = importlib.import_module('celllift.shape_curriculum.src.schemas')
    actual_fields = tuple(dataset.GraphInferenceSample.__dataclass_fields__)
    if actual_fields != EXPECTED_SAMPLE_FIELDS:
        raise RuntimeError(f'compatibility_score GraphInferenceSample schema changed: expected={EXPECTED_SAMPLE_FIELDS}, actual={actual_fields}')
    return {'dataset': dataset, 'pipeline': pipeline, 'predict': predict, 'losses': losses, 'schemas': schemas}

def _read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('pyarrow is required for public graph indexes') from exc
    return pq.read_table(path, partitioning=None).to_pylist()

class PublicShapeCurriculumGraphDataset(Dataset):

    def __init__(self, graph_index: str | Path, graph_root: str | Path, training_code_root: str | Path, *, shards: int=64, feature_stats: str | Path | None=None, limit: int | None=None, dataset_name: str | None=None) -> None:
        modules = _import_compatibility_score(training_code_root)
        self._sample_type = modules['dataset'].GraphInferenceSample
        self._graph_type = modules['schemas'].GraphRecord
        self._upstream_collate = modules['dataset'].collate_inference_samples
        rows = _read_parquet_rows(graph_index)
        if limit is not None:
            if limit <= 0:
                raise ValueError('limit must be positive')
            rows = rows[:int(limit)]
        if not rows:
            raise RuntimeError('public graph index is empty')
        if any(('graph_id' not in row for row in rows)):
            raise KeyError('public graph index requires graph_id')
        self.rows = rows
        self.root = Path(graph_root)
        self.shards = int(shards)
        self.dataset_name = dataset_name
        self._envs: list[Any] | None = None
        stats_path = Path(feature_stats or self.root / 'inference_feature_stats.json')
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        self.mean = np.asarray(stats['mean'], np.float32)
        self.std = np.asarray(stats['std'], np.float32)
        if self.mean.shape != (36,) or self.std.shape != (36,) or (not np.all(np.isfinite(self.mean))) or (not np.all(np.isfinite(self.std))) or np.any(self.std <= 0):
            raise RuntimeError('invalid frozen 36-ray feature statistics')

    def _open(self) -> list[Any]:
        if self._envs is None:
            try:
                import lmdb
            except ImportError as exc:
                raise RuntimeError('lmdb is required for public graph caches') from exc
            paths = [self.root / f'graph_cache_{index:02d}.lmdb' for index in range(self.shards)]
            missing = [str(path) for path in paths if not path.is_dir()]
            if missing:
                raise FileNotFoundError(f'missing graph LMDB shard(s): {missing[:3]}')
            self._envs = [lmdb.open(str(path), subdir=True, readonly=True, lock=False, readahead=False, max_readers=2048) for path in paths]
        return self._envs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        graph_id = str(self.rows[index]['graph_id'])
        env = self._open()[stable_shard(graph_id, self.shards)]
        with env.begin(buffers=False) as transaction:
            payload = transaction.get(graph_id.encode('utf-8'))
        if payload is None:
            raise KeyError(f'graph cache has no key {graph_id!r}')
        graph = self._graph_type.from_bytes(bytes(payload))
        node_count = len(graph.nucleus_id)
        zero = np.zeros(node_count, dtype=np.int64)
        return self._sample_type(graph=graph, nucleus_observation_count=zero.copy(), cell_observation_count=zero.copy(), gold_observation_count=zero.copy(), silver_observation_count=zero.copy(), nucleus_pose_evidence=empty_pose_evidence(), cell_pose_evidence=empty_pose_evidence(), feature_mean=self.mean, feature_std=self.std)

    def collate(self, samples: Sequence[Any]) -> dict[str, Any]:
        batch = self._upstream_collate(samples)
        batch['anchor_border_flag'] = torch.from_numpy(np.ascontiguousarray(np.concatenate([sample.graph.border_flag for sample in samples])))
        if tuple(batch['anchor_border_flag'].shape) != (int(batch['node_features'].shape[0]),):
            raise RuntimeError('anchor border flags do not align with compatibility_score nodes')
        return batch

    def sample_cost(self, index: int) -> dict[str, int]:
        row = self.rows[index]
        return {'nodes': int(row['node_count']), 'edges': int(row['edge_count']), 'projection_voxels': 0}

    def metadata(self, index: int) -> dict[str, Any]:
        return dict(self.rows[index])

    def close(self) -> None:
        if self._envs is not None:
            for env in self._envs:
                env.close()
            self._envs = None
    close_cache_handles = close

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state['_envs'] = None
        return state

@dataclass(frozen=True)
class CurriculumCheckpointInfo:
    path: str
    sha256: str
    run_id: str
    epoch: int
    schema_version: Any
    config_sha256: str
    model_tensor_count: int
    config: Mapping[str, Any]

def inspect_compatibility_score_checkpoint(checkpoint: str | Path, *, expected_sha256: str=FROZEN_CHECKPOINT_SHA256) -> tuple[CurriculumCheckpointInfo, Mapping[str, Any]]:
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f'checkpoint SHA256 mismatch: expected {expected_sha256}, got {digest}')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get('model'), Mapping):
        raise RuntimeError('compatibility_score checkpoint must contain a model state mapping')
    provenance = payload.get('provenance')
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get('config'), Mapping):
        raise RuntimeError('compatibility_score checkpoint lacks provenance.config')
    cfg = provenance['config']
    run_id = str(cfg.get('experiment', {}).get('run_id', ''))
    if run_id != FROZEN_RUN_ID:
        raise RuntimeError(f'unexpected compatibility_score run_id: {run_id!r}')
    model_cfg = cfg.get('model', {})
    expected_model = {'hidden_dim': 128, 'message_layers': 4, 'architecture': 'edge_gated_gnn', 'pose_feature_gradient_mode': 'shared', 'hypothesis_mode': 'single'}
    changed = {key: (model_cfg.get(key), expected) for key, expected in expected_model.items() if model_cfg.get(key) != expected}
    if changed:
        raise RuntimeError(f'selected compatibility_score architecture changed: {changed}')
    if cfg.get('graph', {}).get('node_feature_dim') != 36:
        raise RuntimeError('selected compatibility_score checkpoint does not use 36-ray inputs')
    if cfg.get('loss', {}).get('object_pose_routing') is not True:
        raise RuntimeError('selected compatibility_score checkpoint did not freeze object-pose routing')
    required_state = {'encoder.0.weight', 'nucleus_axis_gap_head.2.weight', 'nucleus_pose_head.2.weight', 'cell_axis_gap_head.2.weight', 'cell_pose_head.2.weight'}
    missing = sorted(required_state - set(payload['model']))
    if missing:
        raise RuntimeError(f'compatibility_score model state misses required tensors: {missing}')
    info = CurriculumCheckpointInfo(path=str(source.resolve()), sha256=digest, run_id=run_id, epoch=int(payload.get('epoch', -1)), schema_version=payload.get('schema_version'), config_sha256=str(provenance.get('config_sha256', '')), model_tensor_count=len(payload['model']), config=cfg)
    return (info, payload)

def load_frozen_compatibility_score_model(training_code_root: str | Path, checkpoint: str | Path, *, expected_sha256: str=FROZEN_CHECKPOINT_SHA256) -> tuple[torch.nn.Module, CurriculumCheckpointInfo, dict[str, Any]]:
    modules = _import_compatibility_score(training_code_root)
    info, payload = inspect_compatibility_score_checkpoint(checkpoint, expected_sha256=expected_sha256)
    cfg = dict(info.config)
    mpp = float(cfg['experiment']['native_mpp'])
    dummy_area_px = math.pi / (mpp * mpp)
    model = modules['pipeline'].build_model(cfg, {'nucleus_median_area_px': dummy_area_px, 'cell_median_area_px': dummy_area_px})
    incompatible = model.load_state_dict(payload['model'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'strict compatibility_score load failed: {incompatible}')
    model.requires_grad_(False).eval()
    return (model, info, modules)

def validate_empty_inference_batch(batch: Mapping[str, Any]) -> dict[str, int]:
    node_count = int(batch['node_features'].shape[0])
    for key in ('nucleus_observation_count', 'cell_observation_count', 'gold_observation_count', 'silver_observation_count'):
        value = batch[key]
        if tuple(value.shape) != (node_count,) or torch.any(value != 0):
            raise RuntimeError(f'public inference batch has non-zero {key}')
    for object_type in ('nucleus', 'cell'):
        evidence = batch[f'{object_type}_pose_evidence']
        for key, (shape, _) in POSE_EVIDENCE_SCHEMA.items():
            if tuple(evidence[key].shape) != shape:
                raise RuntimeError(f'{object_type} pose evidence {key} has shape {tuple(evidence[key].shape)}, expected {shape}')
    return {'nodes': node_count, 'graphs': int(batch['num_graphs'])}

class NodeEdgeBudgetBatchSampler:

    def __init__(self, dataset: PublicShapeCurriculumGraphDataset, *, max_nodes: int, max_edges: int) -> None:
        if max_nodes <= 0 or max_edges <= 0:
            raise ValueError('batch budgets must be positive')
        self.dataset = dataset
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)

    def __iter__(self):
        batch: list[int] = []
        nodes = edges = 0
        for index in range(len(self.dataset)):
            cost = self.dataset.sample_cost(index)
            if cost['nodes'] > self.max_nodes or cost['edges'] > self.max_edges:
                raise RuntimeError(f"graph {self.dataset.rows[index]['graph_id']} exceeds batch budget")
            overflow = batch and (nodes + cost['nodes'] > self.max_nodes or edges + cost['edges'] > self.max_edges)
            if overflow:
                yield batch
                batch, nodes, edges = ([], 0, 0)
            batch.append(index)
            nodes += cost['nodes']
            edges += cost['edges']
        if batch:
            yield batch

    def __len__(self) -> int:
        count = 0
        nodes = edges = 0
        for index in range(len(self.dataset)):
            cost = self.dataset.sample_cost(index)
            if count == 0 or nodes + cost['nodes'] > self.max_nodes or edges + cost['edges'] > self.max_edges:
                count += 1
                nodes = edges = 0
            nodes += cost['nodes']
            edges += cost['edges']
        return count
