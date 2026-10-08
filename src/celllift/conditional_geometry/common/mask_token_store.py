from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData, load_mask_prepared_fold
from celllift.conditional_geometry.mask_protocol import MASK_PROTOCOL_ID, MASK_RESIDUAL_TOKEN_DIM, TARGET_COLUMNS
_CONFIG: Mapping[str, Any] | None = None
_GRAPH_LIMIT: int | None = None
_PROTOCOL_ID = MASK_PROTOCOL_ID
_REUSABLE_STORES: dict[tuple[Any, ...], 'MaskFoldTokenStore'] = {}

def configure(cfg: Mapping[str, Any], graph_limit: int | None=None) -> None:
    global _CONFIG, _GRAPH_LIMIT, _PROTOCOL_ID
    previous_key = None if _CONFIG is None else (str(_CONFIG['paths']['data_root']), _GRAPH_LIMIT)
    next_key = (str(cfg['paths']['data_root']), graph_limit)
    if previous_key != next_key:
        _REUSABLE_STORES.clear()
    _CONFIG, _GRAPH_LIMIT = (cfg, graph_limit)
    _PROTOCOL_ID = str(cfg.get('residual_source_protocol_id', cfg.get('protocol_id', MASK_PROTOCOL_ID)))

@dataclass(frozen=True)
class GraphTokenSample:
    graph_id: str
    nucleus_tokens: np.ndarray
    cell_tokens: np.ndarray
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ShuffleRow:
    graph_id: str
    anchor_id: int
    donor_graph_id: str
    donor_anchor_id: int
    partition: str
    count_decile: int
    donor_count_decile: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

class _ShuffleRows(Sequence[ShuffleRow]):

    def __init__(self, store: 'MaskFoldTokenStore') -> None:
        self.store = store

    def __len__(self) -> int:
        return len(self.store._row_graph)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(len(self)))]
        donor = int(self.store._donor[item])
        return ShuffleRow(str(self.store._row_graph[item]), int(self.store._row_anchor[item]), str(self.store._row_graph[donor]), int(self.store._row_anchor[donor]), str(self.store._row_role[item]), int(self.store._row_decile[item]), int(self.store._row_decile[donor]))

def _rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

class MaskFoldTokenStore:

    def __init__(self, cfg: Mapping[str, Any], fold: int, seed: int, graph_manifest: Path, included_graph_ids: Sequence[str], partition_by_graph: Mapping[str, str]) -> None:
        import pyarrow.parquet as pq
        self.fold, self.seed = (int(fold), int(seed))
        self.partition_by_graph = dict(partition_by_graph)
        self.graph_ids = tuple(sorted(map(str, included_graph_ids)))
        self.training_graph_ids = frozenset((graph for graph in self.graph_ids if self.partition_by_graph[graph] == 'train'))
        metadata_rows = {str(row['graph_id']): row for row in _rows(graph_manifest)}
        self.metadata = {graph: dict(metadata_rows[graph]) for graph in self.graph_ids}
        if str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays':
            from celllift.conditional_geometry.common.rgb_mask_probe_data import RGBMaskFoldProbeData, load_rgb_mask_prepared_fold
            data = RGBMaskFoldProbeData(cfg, fold, graph_limit=_GRAPH_LIMIT)
            prepared_loader = load_rgb_mask_prepared_fold
        else:
            data = MaskFoldProbeData(cfg, fold, graph_limit=_GRAPH_LIMIT)
            prepared_loader = load_mask_prepared_fold
        if set(data.graph_ids) != set(self.graph_ids):
            raise RuntimeError('mask residual store graph set differs from registered outer fold')
        suffix = Path('smoke') / f'graph_limit_{_GRAPH_LIMIT}' if _GRAPH_LIMIT is not None else Path()
        prepared = Path(cfg['paths']['data_root']) / suffix / 'prepared' / f'fold_{fold:02d}'
        residual_root = Path(cfg['paths']['data_root']) / suffix / 'residuals' / f'fold_{fold:02d}'
        prepared_loader(data, prepared)
        manifest = json.loads((residual_root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS' or manifest.get('protocol_id') != _PROTOCOL_ID:
            raise RuntimeError('mask residual manifest is not a compatible PASS')
        residual_paths = sorted(residual_root.glob('residual_*.parquet'))
        if len(residual_paths) != len(manifest['files']):
            raise RuntimeError('mask residual shard count differs from manifest')
        residual_mean = np.asarray(manifest['residual_mean_training_only'], np.float32)
        residual_std = np.asarray(manifest['residual_std_training_only'], np.float32)
        stats = data.statistics
        ray_mean, ray_std = (np.asarray(stats.ray_mean, np.float32), np.asarray(stats.ray_std, np.float32))
        base: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        residual_n: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        residual_c: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        raw_n: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        raw_c: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        valid3d: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        anchors: dict[str, list[np.ndarray]] = {graph: [] for graph in self.graph_ids}
        raw_batches = data._iter_raw()
        target_mean = np.asarray(stats.target_mean, np.float32)
        target_std = np.asarray(stats.target_std, np.float32)
        for residual_path in residual_paths:
            try:
                batch = next(raw_batches)
            except StopIteration as error:
                raise RuntimeError('mask residual manifest has more shards than its source') from error
            table = pq.read_table(residual_path, partitioning=None).to_pydict()
            rg, ra = (np.asarray(table['graph_id'], object), np.asarray(table['anchor_id']))
            if not np.array_equal(batch['graph_id'], rg) or not np.array_equal(batch['anchor_id'], ra):
                raise RuntimeError(f'mask residual/source key mismatch: {residual_path}')
            residual = np.column_stack([np.asarray(table[f'residual_{name}'], np.float32) for name in TARGET_COLUMNS])
            residual = (residual - residual_mean) / residual_std
            valid_ncr3d = np.asarray(table['valid_ncr3d'], bool)
            residual[~valid_ncr3d, 8] = 0.0
            mask_base = ((batch['nucleus_rays'] - ray_mean) / ray_std).astype(np.float32)
            nucleus_residual = np.concatenate((residual[:, :4], residual[:, 8:9]), axis=1).astype(np.float32)
            cell_residual = np.concatenate((residual[:, 4:8], residual[:, 8:9]), axis=1).astype(np.float32)
            raw_ncr = np.where(valid_ncr3d, batch['ncr3d'], stats.ncr3d_median)
            raw_targets = np.concatenate((batch['nucleus_3d'], batch['cell_3d'], raw_ncr[:, None]), axis=1)
            raw_targets = ((raw_targets - target_mean) / target_std).astype(np.float32)
            raw_targets[~valid_ncr3d, 8] = 0.0
            nucleus_raw = np.concatenate((raw_targets[:, :4], raw_targets[:, 8:9]), axis=1)
            cell_raw = np.concatenate((raw_targets[:, 4:8], raw_targets[:, 8:9]), axis=1)
            if not np.isfinite(nucleus_raw).all() or not np.isfinite(cell_raw).all():
                raise RuntimeError('non-finite matched raw full-3D token')
            graph_values = batch['graph_id']
            for graph in dict.fromkeys(map(str, graph_values)):
                selected = graph_values == graph
                base[graph].append(mask_base[selected])
                residual_n[graph].append(nucleus_residual[selected])
                residual_c[graph].append(cell_residual[selected])
                raw_n[graph].append(nucleus_raw[selected])
                raw_c[graph].append(cell_raw[selected])
                valid3d[graph].append(valid_ncr3d[selected])
                anchors[graph].append(batch['anchor_id'][selected])
        try:
            next(raw_batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError('mask source contains an unmanifested residual shard')
        data._raw_cache.clear()
        self._base = {graph: np.concatenate(base[graph]) for graph in self.graph_ids}
        self._residual_n = {graph: np.concatenate(residual_n[graph]) for graph in self.graph_ids}
        self._residual_c = {graph: np.concatenate(residual_c[graph]) for graph in self.graph_ids}
        self._raw_n = {graph: np.concatenate(raw_n[graph]) for graph in self.graph_ids}
        self._raw_c = {graph: np.concatenate(raw_c[graph]) for graph in self.graph_ids}
        self._valid3d = {graph: np.concatenate(valid3d[graph]) for graph in self.graph_ids}
        self._anchors = {graph: np.concatenate(anchors[graph]) for graph in self.graph_ids}
        if any((len(self._base[graph]) == 0 for graph in self.graph_ids)):
            raise RuntimeError('one or more mask residual graphs are empty')
        self._prepare_shuffle()
        self.shuffle_rows = _ShuffleRows(self)

    @classmethod
    def from_parquet(cls, nucleus_root: Path, cell_root: Path, ncr_root: Path, graph_manifest: Path, *, training_graph_ids: Sequence[str], seed: int, included_graph_ids: Sequence[str], partition_by_graph: Mapping[str, str]) -> 'MaskFoldTokenStore':
        if _CONFIG is None:
            raise RuntimeError('mask residual store was not configured')
        manifest = _rows(graph_manifest)
        validation = {graph for graph, role in partition_by_graph.items() if role == 'validation'}
        folds = {int(row['validation_fold']) for row in manifest if str(row['graph_id']) in validation}
        if len(folds) != 1:
            raise RuntimeError(f'cannot infer one outer fold from mask validation IDs: {folds}')
        fold = next(iter(folds))
        cache_key = (str(_CONFIG['paths']['data_root']), int(fold), _GRAPH_LIMIT, tuple(sorted(map(str, included_graph_ids))), tuple(sorted(((str(graph), str(role)) for graph, role in partition_by_graph.items()))))
        store = _REUSABLE_STORES.get(cache_key)
        if store is None:
            store = cls(_CONFIG, fold, seed, graph_manifest, included_graph_ids, partition_by_graph)
            _REUSABLE_STORES[cache_key] = store
        else:
            store.seed = int(seed)
            store._prepare_shuffle()
            store.shuffle_rows = _ShuffleRows(store)
        if store.training_graph_ids != frozenset(map(str, training_graph_ids)):
            raise RuntimeError('mask token-store training graph IDs differ from trainer partition')
        return store

    def _prepare_shuffle(self) -> None:
        if not hasattr(self, '_row_graph'):
            row_graph, row_anchor, row_role, row_decile = ([], [], [], [])
            graph_decile: dict[str, int] = {}
            for role in ('train', 'validation'):
                role_graphs = sorted((graph for graph in self.graph_ids if self.partition_by_graph[graph] == role), key=lambda graph: (len(self._anchors[graph]), graph))
                if len(role_graphs) < 20:
                    raise RuntimeError(f'at least 20 graphs are required for mask shuffle deciles: {role}')
                for rank, graph in enumerate(role_graphs):
                    graph_decile[graph] = min(9, rank * 10 // len(role_graphs))
            self._ranges: dict[str, slice] = {}
            cursor = 0
            for graph in self.graph_ids:
                count = len(self._anchors[graph])
                self._ranges[graph] = slice(cursor, cursor + count)
                cursor += count
                row_graph.extend([graph] * count)
                row_anchor.extend(self._anchors[graph].tolist())
                row_role.extend([self.partition_by_graph[graph]] * count)
                row_decile.extend([graph_decile[graph]] * count)
            self._row_graph = np.asarray(row_graph, object)
            self._row_anchor = np.asarray(row_anchor)
            self._row_role = np.asarray(row_role, object)
            self._row_decile = np.asarray(row_decile, np.int8)
            self._flat_res_n = np.concatenate([self._residual_n[graph] for graph in self.graph_ids])
            self._flat_res_c = np.concatenate([self._residual_c[graph] for graph in self.graph_ids])
            self._flat_raw_n = np.concatenate([self._raw_n[graph] for graph in self.graph_ids])
            self._flat_raw_c = np.concatenate([self._raw_c[graph] for graph in self.graph_ids])
        self._donor = np.empty(len(self._row_graph), np.int64)
        seed_bytes = hashlib.sha256(f'{_PROTOCOL_ID}|fold={self.fold}|seed={self.seed}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], 'little'))
        for role in ('train', 'validation'):
            for decile in range(10):
                target = np.flatnonzero((self._row_role == role) & (self._row_decile == decile))
                if not len(target):
                    continue
                if len(np.unique(self._row_graph[target])) < 2:
                    raise RuntimeError(f'mask shuffle role/decile has fewer than two graphs: {role}/{decile}')
                proposal = target[rng.integers(0, len(target), size=len(target))]
                bad = self._row_graph[proposal] == self._row_graph[target]
                attempts = 0
                while bad.any() and attempts < 64:
                    proposal[bad] = target[rng.integers(0, len(target), size=int(bad.sum()))]
                    bad = self._row_graph[proposal] == self._row_graph[target]
                    attempts += 1
                if bad.any():
                    for position in np.flatnonzero(bad):
                        candidates = target[self._row_graph[target] != self._row_graph[target[position]]]
                        proposal[position] = candidates[int(rng.integers(0, len(candidates)))]
                self._donor[target] = proposal

    def samples(self, arm: Any, graph_ids: Sequence[str]) -> list[GraphTokenSample]:
        output: list[GraphTokenSample] = []
        for graph in graph_ids:
            base = self._base[graph]
            if arm.arm_id == 'M3_MASK2D':
                rn = np.zeros((len(base), 5), np.float32)
                rc = np.zeros_like(rn)
            elif arm.arm_id == 'M3_MASK_RES3D':
                rn, rc = (self._residual_n[graph], self._residual_c[graph])
            elif arm.arm_id == 'M3_MASK_SHUF_RES3D':
                donor = self._donor[self._ranges[graph]]
                rn, rc = (self._flat_res_n[donor].copy(), self._flat_res_c[donor].copy())
                invalid_target = ~self._valid3d[graph]
                rn[invalid_target, -1] = 0.0
                rc[invalid_target, -1] = 0.0
            elif arm.arm_id == 'M5_MASK_FULL3D':
                rn, rc = (self._raw_n[graph], self._raw_c[graph])
            elif arm.arm_id == 'M5_MASK_SHUF_FULL3D':
                donor = self._donor[self._ranges[graph]]
                rn, rc = (self._flat_raw_n[donor].copy(), self._flat_raw_c[donor].copy())
                invalid_target = ~self._valid3d[graph]
                rn[invalid_target, -1] = 0.0
                rc[invalid_target, -1] = 0.0
            else:
                raise ValueError(f'unknown mask residual arm {arm.arm_id}')
            nucleus = np.concatenate((base, rn), axis=1)
            cell = np.concatenate((base, rc), axis=1)
            if nucleus.shape[1] != MASK_RESIDUAL_TOKEN_DIM or cell.shape != nucleus.shape:
                raise AssertionError('mask residual token schema changed')
            output.append(GraphTokenSample(graph, nucleus, cell, self.metadata[graph]))
        return output

    def ncr_statistics(self, arm: Any) -> None:
        return None
FoldTokenStore = MaskFoldTokenStore

def collate_variable_tokens(samples: Sequence[GraphTokenSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot collate an empty mask sample list')
    maximum = max((len(sample.nucleus_tokens) for sample in samples))
    nucleus = np.zeros((len(samples), maximum, MASK_RESIDUAL_TOKEN_DIM), np.float32)
    cell = np.zeros_like(nucleus)
    nucleus_mask = np.zeros((len(samples), maximum), bool)
    cell_mask = np.zeros_like(nucleus_mask)
    for index, sample in enumerate(samples):
        count = len(sample.nucleus_tokens)
        nucleus[index, :count] = sample.nucleus_tokens
        cell[index, :count] = sample.cell_tokens
        nucleus_mask[index, :count] = True
        cell_mask[index, :count] = True
    return {'graph_ids': [sample.graph_id for sample in samples], 'nucleus_tokens': nucleus, 'cell_tokens': cell, 'nucleus_mask': nucleus_mask, 'cell_mask': cell_mask}
