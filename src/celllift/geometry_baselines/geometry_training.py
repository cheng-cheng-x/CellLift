from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import importlib.util
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
import sys
from typing import Any, Mapping, Sequence
import numpy as np
from .io_utils import atomic_json, sha256
from .models import DualGeometryExpert
from .protocol import ARMS, PROTOCOL_ID, Arm
from .tokens import ModalGraph, ModalTokenStore, collate
ROOT = Path(__file__).resolve().parent
PUBLIC_ROOT = ROOT.parent
set_encoding_ROOT = PUBLIC_ROOT / 'set_encoding'
conditional_geometry_ROOT = PUBLIC_ROOT / 'conditional_geometry'
_CFG: Mapping[str, Any] | None = None
_COMPATIBILITY_MODULE = None
_STORE_CACHE: dict[tuple[Any, ...], 'GeometryFoldTokenStore'] = {}

def configure(cfg: Mapping[str, Any], graph_limit: int | None=None) -> None:
    if graph_limit is not None:
        raise RuntimeError('geometry_baselines does not permit partial/smoke geometry stores')
    global _CFG, _COMPATIBILITY_MODULE
    _CFG = cfg
    if str(conditional_geometry_ROOT) not in sys.path:
        sys.path.insert(0, str(conditional_geometry_ROOT))
    import celllift.conditional_geometry.common.mask_token_store as compatibility
    compatibility_cfg = dict(cfg)
    compatibility_cfg['paths'] = dict(cfg['paths'])
    compatibility_cfg['paths']['data_root'] = str(cfg['paths']['mask_conditioning_data_root'])
    compatibility_cfg['protocol_id'] = 'matched_mask_full3d_vs_residual3d_raw_residual_comparison'
    compatibility_cfg['residual_source_protocol_id'] = 'conditional_3d_residual_patient_crossfit_mask_conditioning_nucleus_mask36_only'
    compatibility_cfg['probe'] = {'conditioning_mode': 'mask_rays_only', 'inner_folds': 3}
    compatibility.configure(compatibility_cfg, graph_limit=None)
    _COMPATIBILITY_MODULE = compatibility

@dataclass(frozen=True)
class GraphTokenSample:
    graph_id: str
    nucleus_tokens: np.ndarray
    cell_tokens: np.ndarray
    metadata: Mapping[str, object]

def preaggregate_meanpool(samples: Sequence[GraphTokenSample]) -> list[GraphTokenSample]:
    output: list[GraphTokenSample] = []
    for sample in samples:
        nucleus_count, cell_count = (len(sample.nucleus_tokens), len(sample.cell_tokens))
        if nucleus_count <= 0 or cell_count <= 0:
            raise RuntimeError(f'empty geometry set cannot be preaggregated: {sample.graph_id}')
        metadata = dict(sample.metadata)
        metadata.update({'nucleus_count': nucleus_count, 'cell_count': cell_count, 'meanpool_preaggregated': True})
        output.append(GraphTokenSample(sample.graph_id, sample.nucleus_tokens.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32), sample.cell_tokens.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32), metadata))
    return output

class _MeanPoolStoreView:

    def __init__(self, store: GeometryFoldTokenStore) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def samples(self, arm: Arm, graph_ids: Sequence[str]) -> list[GraphTokenSample]:
        return preaggregate_meanpool(self._store.samples(arm, graph_ids))

class GeometryFoldTokenStore:

    def __init__(self, compatibility: Any, seed: int) -> None:
        self._compatibility = compatibility
        self.seed = int(seed)
        self.graph_ids = compatibility.graph_ids
        self.training_graph_ids = compatibility.training_graph_ids
        self.partition_by_graph = compatibility.partition_by_graph
        self.metadata = compatibility.metadata
        graphs = [ModalGraph(graph_id=graph, mask2d=compatibility._base[graph], nucleus_raw3d=compatibility._raw_n[graph], cell_raw3d=compatibility._raw_c[graph], nucleus_residual3d=compatibility._residual_n[graph], cell_residual3d=compatibility._residual_c[graph], metadata=compatibility.metadata[graph]) for graph in self.graph_ids]
        self.modal = ModalTokenStore(graphs, self.partition_by_graph, seed=self.seed)

    @classmethod
    def from_parquet(cls, nucleus_root: Path, cell_root: Path, ncr_root: Path, graph_manifest: Path, *, training_graph_ids: Sequence[str], seed: int, included_graph_ids: Sequence[str], partition_by_graph: Mapping[str, str]) -> 'GeometryFoldTokenStore':
        if _CFG is None or _COMPATIBILITY_MODULE is None:
            raise RuntimeError('geometry_baselines geometry store is not configured')
        validation_folds = {int(row['validation_fold']) for row in _read_rows(graph_manifest) if str(row['graph_id']) in partition_by_graph and partition_by_graph[str(row['graph_id'])] == 'validation'}
        if len(validation_folds) != 1:
            raise RuntimeError(f'cannot infer one validation fold: {validation_folds}')
        fold = next(iter(validation_folds))
        key = (fold, int(seed), tuple(sorted(included_graph_ids)))
        if key not in _STORE_CACHE:
            compatibility = _COMPATIBILITY_MODULE.MaskFoldTokenStore.from_parquet(nucleus_root, cell_root, ncr_root, graph_manifest, training_graph_ids=training_graph_ids, seed=seed, included_graph_ids=included_graph_ids, partition_by_graph=partition_by_graph)
            _STORE_CACHE[key] = cls(compatibility, seed)
        return _STORE_CACHE[key]

    def samples(self, arm: Arm, graph_ids: Sequence[str]) -> list[GraphTokenSample]:
        if arm.arm_id.startswith('R'):
            arm = ARMS['O' + arm.arm_id[1:]]
        if arm.three_d_mode.startswith('shuffled_'):
            output = []
            residual = arm.three_d_mode.endswith('residual')
            flat_n = self._compatibility._flat_res_n if residual else self._compatibility._flat_raw_n
            flat_c = self._compatibility._flat_res_c if residual else self._compatibility._flat_raw_c
            for graph_id in graph_ids:
                base = self._compatibility._base[graph_id]
                donor = self._compatibility._donor[self._compatibility._ranges[graph_id]]
                mask = base if arm.mask_mode == 'real' else np.zeros_like(base)
                donor_n, donor_c = (flat_n[donor].copy(), flat_c[donor].copy())
                invalid_target = ~self._compatibility._valid3d[graph_id]
                donor_n[invalid_target, -1] = 0.0
                donor_c[invalid_target, -1] = 0.0
                nucleus = np.concatenate((mask, donor_n), axis=1).astype(np.float32, copy=False)
                cell = np.concatenate((mask, donor_c), axis=1).astype(np.float32, copy=False)
                if arm.mask_mode == 'none' and (np.any(nucleus[:, :36]) or np.any(cell[:, :36])):
                    raise AssertionError('shuffled 3D-only arm leaked MASK2D')
                metadata = dict(self.metadata[graph_id])
                metadata['three_d_shuffle'] = 'raw_residual_comparison anchor donor map: same split/count-decile/cross-graph; paired nucleus-cell'
                output.append(GraphTokenSample(graph_id, nucleus, cell, metadata))
            return output
        result = self.modal.samples(arm, graph_ids)
        return [GraphTokenSample(row.graph_id, row.nucleus_tokens, row.cell_tokens, row.metadata) for row in result]

    def ncr_statistics(self, arm: Arm) -> None:
        return None
FoldTokenStore = GeometryFoldTokenStore

def _read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def collate_variable_tokens(samples: Sequence[GraphTokenSample]) -> dict[str, object]:
    return collate(samples)

def audit_geometry_store(cfg: Mapping[str, Any], fold: int, *, seed: int=42) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    configure(cfg)
    model_input = Path(cfg['paths']['model_input_root'])
    rows = _read_rows(model_input / '03_graph_cache' / 'graph_index.parquet')
    patch_rows = {str(row['graph_id']): row for row in _read_rows(model_input / '00_manifest' / 'patch_manifest.parquet')}
    for row in rows:
        patch = patch_rows[str(row['graph_id'])]
        for name in ('g4c_label', 'g4c_valid', 'rgb_path', 'physical_width_um', 'physical_height_um'):
            row[name] = patch.get(name)
    data_root = Path(cfg['paths']['data_root'])
    graph_manifest = data_root / '00_manifest' / 'downstream_manifest.parquet'
    graph_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = graph_manifest.with_name(f'.{graph_manifest.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, graph_manifest)
    eligible = [row for row in rows if str(row['split']).lower() == 'train']
    train_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) != fold]
    val_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) == fold]
    partition = {graph: 'train' for graph in train_ids} | {graph: 'validation' for graph in val_ids}
    store = GeometryFoldTokenStore.from_parquet(Path(cfg['paths']['set_encoding_data_root']) / '02_nucleus_tokens', Path(cfg['paths']['set_encoding_data_root']) / '03_cell_tokens', Path(cfg['paths']['set_encoding_data_root']) / '04_ncr_features', graph_manifest, training_graph_ids=train_ids, seed=seed, included_graph_ids=train_ids + val_ids, partition_by_graph=partition)
    modes: dict[str, dict[str, int]] = {}
    for geometry_id in ('G1', 'G1S', 'G2', 'G2S', 'G3', 'G3S', 'G4', 'G4S', 'G5', 'G5S'):
        arm = ARMS['O' + geometry_id[1:]]
        anchors = 0
        for graph_id in store.graph_ids:
            sample = store.samples(arm, [graph_id])[0]
            anchors += len(sample.nucleus_tokens)
        modes[geometry_id] = {'graphs': len(store.graph_ids), 'anchors': anchors, 'token_dim': 41}
    mapping = _persist_maps(store, data_root, phase='validation', fold=fold, seed=seed)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'fold': fold, 'seed': seed, 'training_graphs': len(train_ids), 'validation_graphs': len(val_ids), 'modes': modes, 'shuffle_mapping': mapping}
    atomic_json(data_root / '06_geometry_views' / f'fold_{fold:02d}' / f'seed_{seed}' / 'manifest.json', payload)
    return payload

class GeometryDualSetFusion(DualGeometryExpert):

    def __init__(self, set_encoder: str='meanpool', object_mode: str='both', use_rgb: bool=False, rgb_dim: int=512, dropout: float=0.1) -> None:
        if object_mode != 'both' or use_rgb:
            raise ValueError('geometry_baselines geometry experts are no-RGB dual-branch models')
        super().__init__(set_encoder=set_encoder, dropout=dropout)
        self.object_mode, self.use_rgb, self.rgb_dim = (object_mode, use_rgb, rgb_dim)

def _persist_maps(store: GeometryFoldTokenStore, data_root: Path, *, phase: str, fold: int, seed: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(data_root) / '05_shuffle_maps' / phase / f'fold_{fold:02d}' / f'seed_{seed}'
    mask_destination = root / 'mask_graph_donors.parquet'
    three_d_destination = root / 'three_d_anchor_donors.parquet'
    manifest_path = root / 'manifest.json'
    mask_rows = []
    for graph in store.graph_ids:
        mask_rows.append({'graph_id': graph, 'partition': store.partition_by_graph[graph], 'mask_donor_graph_id': store.modal.mask_donor[graph], 'mask_cross_self': store.modal.mask_donor[graph] != graph})
    root.mkdir(parents=True, exist_ok=True)
    temporary = mask_destination.with_name(f'.{mask_destination.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(mask_rows), temporary, compression='zstd')
    os.replace(temporary, mask_destination)
    compatibility = store._compatibility
    temporary = three_d_destination.with_name(f'.{three_d_destination.name}.tmp.{os.getpid()}')
    writer = None
    try:
        for start in range(0, len(compatibility._row_graph), 500000):
            stop = min(len(compatibility._row_graph), start + 500000)
            donor = compatibility._donor[start:stop]
            table = pa.table({'graph_id': pa.array(compatibility._row_graph[start:stop]), 'anchor_id': pa.array(compatibility._row_anchor[start:stop]), 'donor_graph_id': pa.array(compatibility._row_graph[donor]), 'donor_anchor_id': pa.array(compatibility._row_anchor[donor]), 'partition': pa.array(compatibility._row_role[start:stop]), 'count_decile': pa.array(compatibility._row_decile[start:stop]), 'donor_count_decile': pa.array(compatibility._row_decile[donor])})
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table, row_group_size=100000)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError('empty raw_residual_comparison-compatible 3D donor map')
    os.replace(temporary, three_d_destination)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'phase': phase, 'fold': fold, 'seed': seed, 'mask_mapping': {'rows': len(mask_rows), 'path': str(mask_destination), 'sha256': sha256(mask_destination)}, 'three_d_mapping': {'rows': len(compatibility._row_graph), 'path': str(three_d_destination), 'sha256': sha256(three_d_destination)}, 'constraints': 'MASK: whole-set same-split/count-decile/cross-graph; 3D: exact raw_residual_comparison anchor-level same-split/count-decile/cross-graph deterministic map shared by raw/residual and paired nucleus-cell'}
    atomic_json(manifest_path, payload)
    return payload

def _load_trainer():
    compatibility_path_root = str(conditional_geometry_ROOT)
    while compatibility_path_root in sys.path:
        sys.path.remove(compatibility_path_root)
    sys.path.insert(0, compatibility_path_root)
    existing_models = sys.modules.get('models')
    if existing_models is not None:
        module_path = Path(str(getattr(existing_models, '__file__', ''))).resolve()
        try:
            is_compatibility_models = module_path.is_relative_to((conditional_geometry_ROOT / 'models').resolve())
        except ValueError:
            is_compatibility_models = False
        if not is_compatibility_models:
            for name in [key for key in sys.modules if key == 'models' or key.startswith('models.')]:
                del sys.modules[name]
    import celllift.conditional_geometry.common as common
    import celllift.geometry_baselines.models as compatibility_models
    compatibility_path = Path(str(compatibility_models.__file__)).resolve()
    if not compatibility_path.is_relative_to((conditional_geometry_ROOT / 'models').resolve()):
        raise RuntimeError(f'audited trainer resolved the wrong models package: {compatibility_path}')
    module_name = 'geometry_baselines_experiment_binding'
    binding = type(sys)(module_name)
    binding.ARMS = {key: value for key, value in ARMS.items() if key.startswith('O')}
    binding.ExperimentArm = Arm
    binding.SCREENING_PROTOCOL_ID = PROTOCOL_ID
    sys.modules[module_name] = binding
    common.token_store = sys.modules[__name__]
    sys.modules['common.token_store'] = sys.modules[__name__]
    sys.modules['experiment'] = binding
    compatibility_models.DualSetFusion = GeometryDualSetFusion
    spec = importlib.util.spec_from_file_location('geometry_baselines_audited_trainer', set_encoding_ROOT / 'scripts' / 'run_downstream.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load audited geometry trainer')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if trainer.SCREENING_PROTOCOL_ID != PROTOCOL_ID:
        raise RuntimeError('geometry trainer protocol binding failed')
    trainer._load_rgb = lambda _: {}
    trainer._persist_shuffle_mapping = _persist_maps
    original_run_job = trainer._run_job

    def run_job_with_preaggregation(cfg, store, rgb, arm, encoder, *args, **kwargs):
        active_store = _MeanPoolStoreView(store) if encoder == 'meanpool' else store
        return original_run_job(cfg, active_store, rgb, arm, encoder, *args, **kwargs)

    def fusion_inputs_with_counts(batch, rgb, arm, device):
        result = trainer._fusion_inputs_original(batch, rgb, arm, device)
        for name in ('nucleus_count', 'cell_count'):
            if name in batch:
                result[name] = trainer._tensor(batch[name], device)
        return result
    trainer._run_job = run_job_with_preaggregation
    trainer._fusion_inputs_original = trainer._fusion_inputs
    trainer._fusion_inputs = fusion_inputs_with_counts
    return trainer

def train_patient_geometry(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', encoders: Sequence[str] | None=None, seeds: Sequence[int] | None=None, geometry_ids: Sequence[str] | None=None) -> dict[str, Any]:
    from .reuse import materialize_reuse
    reuse = materialize_reuse(cfg)
    configure(cfg)
    trainer = _load_trainer()
    runtime = dict(cfg)
    runtime['paths'] = dict(cfg['paths'])
    runtime['training'] = dict(cfg['geometry_training'])
    runtime['paths']['data_root'] = str(cfg['paths']['data_root'])
    runtime['paths']['result_root'] = str(Path(cfg['paths']['result_root']) / 'geometry_only')
    budget = int(runtime['training'].get('token_budget', 32000))
    runtime['training']['token_budget'] = {'meanpool': budget, 'deepsets': budget}
    ids = tuple(geometry_ids or ('G1', 'G1S', 'G2', 'G2S', 'G3', 'G3S', 'G4', 'G4S', 'G5', 'G5S'))
    arms = [ARMS['O' + geometry_id[1:]] for geometry_id in ids]
    result = trainer.run_grid(runtime, arms, encoders=tuple(encoders or ('meanpool', 'deepsets')), seeds=tuple(map(int, seeds or (17, 42, 73, 101, 137))), fold=int(fold), validation_only=True, device=device)
    result['reuse'] = reuse
    return result
