from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import numpy as np
from celllift.set_encoding.experiment import ExperimentArm
from celllift.set_encoding.features.controls import AnchorShuffleRow, deterministic_anchor_derangement
from celllift.set_encoding.features.ncr import NCRAudit, NCRFeatures, NCRStatistics, compute_log_ncr, fit_ncr_statistics, transform_ncr
from celllift.set_encoding.features.tokens import MORPHOLOGY_3D_SLICE, NCR_SLICE, TOKEN_COLUMNS, TOKEN_DIM, apply_paired_ncr_channels, assert_token_schema
AnchorKey = tuple[str, str]

@dataclass(frozen=True)
class GraphTokenSample:
    graph_id: str
    anchor_ids: np.ndarray
    nucleus_tokens: np.ndarray | None
    cell_tokens: np.ndarray | None
    metadata: Mapping[str, Any]

def _anchor_text(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)

def _key(row: Mapping[str, Any]) -> AnchorKey:
    if 'graph_id' not in row or 'anchor_id' not in row:
        raise ValueError('anchor table rows require graph_id and anchor_id')
    return (str(row['graph_id']), _anchor_text(row['anchor_id']))

def _unique_rows(rows: Iterable[Mapping[str, Any]], name: str) -> dict[AnchorKey, Mapping[str, Any]]:
    result: dict[AnchorKey, Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f'duplicate {name} anchor key {key!r}')
        result[key] = row
    return result

def _audit_subset(audit: NCRAudit, mask: np.ndarray) -> NCRAudit:
    return NCRAudit(nucleus_measure=audit.nucleus_measure[mask], cell_measure=audit.cell_measure[mask], cytoplasm_measure=audit.cytoplasm_measure[mask], raw_log_ncr=audit.raw_log_ncr[mask], valid=audit.valid[mask], reason=audit.reason[mask])

def _expand_parquet_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    values = [paths] if isinstance(paths, (str, Path)) else list(paths)
    expanded: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            expanded.extend(sorted(path.glob('*.parquet')))
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(path)
    if not expanded:
        raise FileNotFoundError('no Parquet files found')
    return expanded

def _read_parquet_rows(paths: str | Path | Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    for path in _expand_parquet_paths(paths):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536):
            yield from batch.to_pylist()

class FoldTokenStore:

    def __init__(self, nucleus_rows: Iterable[Mapping[str, Any]], cell_rows: Iterable[Mapping[str, Any]], ncr_rows: Iterable[Mapping[str, Any]], manifest_rows: Iterable[Mapping[str, Any]], *, training_graph_ids: Iterable[str], seed: int, included_graph_ids: Iterable[str] | None=None, partition_by_graph: Mapping[str, str] | None=None) -> None:
        nucleus = _unique_rows(nucleus_rows, 'nucleus')
        cell = _unique_rows(cell_rows, 'cell')
        if nucleus.keys() != cell.keys():
            missing = {'missing_from_nucleus': sorted(cell.keys() - nucleus.keys())[:5], 'missing_from_cell': sorted(nucleus.keys() - cell.keys())[:5]}
            raise ValueError(f'nucleus/cell anchor joins are not one-to-one: {missing}')
        token_key_set = set(nucleus)
        manifest: dict[str, Mapping[str, Any]] = {}
        for row in manifest_rows:
            graph_id = str(row.get('graph_id', row.get('patch_id', '')))
            if not graph_id:
                raise ValueError('manifest rows require graph_id or patch_id')
            if graph_id in manifest:
                raise ValueError(f'duplicate graph manifest key {graph_id!r}')
            manifest[graph_id] = row
        selected = None if included_graph_ids is None else {str(value) for value in included_graph_ids}
        keys = sorted((key for key in nucleus if selected is None or key[0] in selected))
        if not keys:
            raise ValueError('fold token store selection is empty')
        graphs = sorted({key[0] for key in keys})
        missing_manifest = sorted(set(graphs) - manifest.keys())
        if missing_manifest:
            raise ValueError(f'token graphs absent from graph manifest: {missing_manifest[:5]}')
        if selected is not None:
            missing_selected = sorted(selected - set(graphs))
            if missing_selected:
                raise ValueError(f'selected graphs have no joined tokens: {missing_selected[:5]}')
        self.seed = int(seed)
        self.keys: tuple[AnchorKey, ...] = tuple(keys)
        self.graph_ids: tuple[str, ...] = tuple(graphs)
        self.metadata = {graph: dict(manifest[graph]) for graph in graphs}
        self._key_index = {key: index for index, key in enumerate(self.keys)}
        grouped_indices: dict[str, list[int]] = {graph: [] for graph in graphs}
        for index, key in enumerate(self.keys):
            grouped_indices[key[0]].append(index)
        self._graph_indices = {graph: np.asarray(indices, dtype=np.int64) for graph, indices in grouped_indices.items()}
        self.nucleus_tokens = self._token_matrix(nucleus, 'nucleus')
        self.cell_tokens = self._token_matrix(cell, 'cell')
        del nucleus, cell
        ncr = _unique_rows(ncr_rows, 'NCR')
        if set(ncr) != token_key_set:
            missing = {'missing_from_ncr': sorted(token_key_set - ncr.keys())[:5], 'extra_in_ncr': sorted(ncr.keys() - token_key_set)[:5]}
            raise ValueError(f'nucleus/cell/NCR anchor joins are not one-to-one: {missing}')
        self._nucleus_area = self._column(ncr, 'nucleus_area_2d_um2', float)
        self._cell_area = self._column(ncr, 'cell_area_2d_um2', float)
        self._nucleus_volume = self._column(ncr, 'nucleus_volume_3d_um3', float)
        self._cell_volume = self._column(ncr, 'cell_volume_3d_um3', float)
        self._original_audits = {'2d': compute_log_ncr(self._nucleus_area, self._cell_area), '3d': compute_log_ncr(self._nucleus_volume, self._cell_volume)}
        self._validate_raw_ncr(ncr)
        training_graphs = {str(value) for value in training_graph_ids}
        if not training_graphs or not training_graphs <= set(graphs):
            raise ValueError('training_graph_ids must be a non-empty subset of included token graphs')
        self.training_graph_ids = frozenset(training_graphs)
        self._training_mask = np.asarray([key[0] in training_graphs for key in self.keys], dtype=bool)
        if not self._training_mask.any():
            raise ValueError('training fold contains no anchors')
        explicit_partitions = {str(key): str(value) for key, value in (partition_by_graph or {}).items()}
        unknown_partitions = explicit_partitions.keys() - set(graphs)
        if unknown_partitions:
            raise ValueError(f'partition map contains unknown graphs: {sorted(unknown_partitions)[:5]}')
        self.partition_by_graph: dict[str, str] = {}
        for graph in graphs:
            if graph in explicit_partitions:
                partition = explicit_partitions[graph]
            elif graph in training_graphs:
                partition = 'train'
            else:
                raw = str(manifest[graph].get('fold_role', manifest[graph].get('official_split', manifest[graph].get('split', 'heldout')))).lower()
                partition = 'heldout' if raw in {'train', 'training'} else raw
            if not partition:
                raise ValueError(f'empty fold partition for graph {graph!r}')
            self.partition_by_graph[graph] = partition
        mislabeled_training = sorted((graph for graph in training_graphs if self.partition_by_graph[graph] != 'train'))
        mislabeled_nontraining = sorted((graph for graph in graphs if graph not in training_graphs and self.partition_by_graph[graph] == 'train'))
        if mislabeled_training or mislabeled_nontraining:
            raise ValueError(f'fold partition disagrees with training_graph_ids: training_not_train={mislabeled_training[:5]}, nontraining_marked_train={mislabeled_nontraining[:5]}')
        self._statistics_cache: dict[tuple[str, str], NCRStatistics] = {}
        self._audit_cache: dict[tuple[str, str], NCRAudit] = {}
        self._donor_indices: np.ndarray | None = None
        self._shuffle_rows: tuple[AnchorShuffleRow, ...] | None = None

    @classmethod
    def from_parquet(cls, nucleus_paths: str | Path | Iterable[str | Path], cell_paths: str | Path | Iterable[str | Path], ncr_paths: str | Path | Iterable[str | Path], graph_manifest_path: str | Path, *, training_graph_ids: Iterable[str], seed: int, included_graph_ids: Iterable[str] | None=None, partition_by_graph: Mapping[str, str] | None=None) -> 'FoldTokenStore':
        return cls(_read_parquet_rows(nucleus_paths), _read_parquet_rows(cell_paths), _read_parquet_rows(ncr_paths), _read_parquet_rows(graph_manifest_path), training_graph_ids=training_graph_ids, seed=seed, included_graph_ids=included_graph_ids, partition_by_graph=partition_by_graph)

    def _token_matrix(self, rows: Mapping[AnchorKey, Mapping[str, Any]], expected_type: str) -> np.ndarray:
        values = []
        for key in self.keys:
            row = rows[key]
            if 'object_type' in row and str(row['object_type']) != expected_type:
                raise ValueError(f"{expected_type} cache has object_type={row['object_type']!r} at {key!r}")
            values.append([row[column] for column in TOKEN_COLUMNS])
        matrix = np.asarray(values, dtype=np.float32)
        assert_token_schema(matrix, geometry_mode='3d', ncr_enabled=False)
        return matrix

    def _column(self, rows: Mapping[AnchorKey, Mapping[str, Any]], name: str, dtype: type) -> np.ndarray:
        try:
            return np.asarray([rows[key][name] for key in self.keys], dtype=dtype)
        except KeyError as exc:
            raise ValueError(f'NCR cache lacks required column {name!r}') from exc

    def _validate_raw_ncr(self, rows: Mapping[AnchorKey, Mapping[str, Any]]) -> None:
        for mode, audit in self._original_audits.items():
            raw_name, valid_name, reason_name = (f'raw_log_ncr_{mode}', f'valid_ncr_{mode}', f'ncr_reason_{mode}')
            cached_raw = self._column(rows, raw_name, float)
            cached_valid = self._column(rows, valid_name, bool)
            if not np.array_equal(cached_valid, audit.valid):
                raise ValueError(f'cached {mode} NCR validity disagrees with raw measures')
            if not np.allclose(cached_raw[audit.valid], audit.raw_log_ncr[audit.valid], rtol=1e-06, atol=1e-07):
                raise ValueError(f'cached {mode} NCR values disagree with raw measures')
            cached_reason = self._column(rows, reason_name, str)
            if not np.array_equal(cached_reason, audit.reason):
                raise ValueError(f'cached {mode} NCR reasons disagree with raw measures')

    @property
    def shuffle_rows(self) -> tuple[AnchorShuffleRow, ...]:
        self._ensure_shuffle_mapping()
        assert self._shuffle_rows is not None
        return self._shuffle_rows

    def _ensure_shuffle_mapping(self) -> None:
        if self._donor_indices is not None:
            return
        mapping = deterministic_anchor_derangement([key[0] for key in self.keys], [key[1] for key in self.keys], [self.partition_by_graph[key[0]] for key in self.keys], seed=self.seed)
        mapping_by_target = {(row.target_graph_id, row.target_anchor_id): row for row in mapping}
        if mapping_by_target.keys() != self._key_index.keys():
            raise RuntimeError('shuffle mapping does not cover the joined anchor domain')
        self._donor_indices = np.asarray([self._key_index[mapping_by_target[key].donor_graph_id, mapping_by_target[key].donor_anchor_id] for key in self.keys], dtype=np.int64)
        self._shuffle_rows = tuple((mapping_by_target[key] for key in self.keys))

    def _audit_for_arm(self, arm: ExperimentArm) -> NCRAudit | None:
        if arm.ncr_mode == 'none':
            return None
        cache_key = (arm.ncr_mode, arm.control_mode)
        if cache_key in self._audit_cache:
            return self._audit_cache[cache_key]
        mode = arm.ncr_mode
        if arm.control_mode in {'none', 'nucleus_duplicate', 'cell_duplicate', 'shuffled_cell_original_ncr'}:
            audit = self._original_audits[mode]
        else:
            self._ensure_shuffle_mapping()
            assert self._donor_indices is not None
            if mode == '2d':
                target_nucleus, target_cell = (self._nucleus_area, self._cell_area)
            else:
                target_nucleus, target_cell = (self._nucleus_volume, self._cell_volume)
            if arm.control_mode == 'shuffled_cell_recomputed_ncr':
                audit = compute_log_ncr(target_nucleus, target_cell[self._donor_indices])
            elif arm.control_mode == 'shuffled_ncr':
                audit = compute_log_ncr(target_nucleus[self._donor_indices], target_cell[self._donor_indices])
            else:
                raise ValueError(f'unsupported control_mode {arm.control_mode!r}')
        self._audit_cache[cache_key] = audit
        return audit

    def ncr_statistics(self, arm: ExperimentArm) -> NCRStatistics | None:
        audit = self._audit_for_arm(arm)
        if audit is None:
            return None
        key = (arm.ncr_mode, arm.control_mode)
        if key not in self._statistics_cache:
            self._statistics_cache[key] = fit_ncr_statistics(_audit_subset(audit, self._training_mask))
        return self._statistics_cache[key]

    def _arm_arrays(self, arm: ExperimentArm) -> tuple[np.ndarray | None, np.ndarray | None]:
        if arm.object_mode == 'rgb':
            return (None, None)
        nucleus = self.nucleus_tokens.copy()
        cell = self.cell_tokens.copy()
        if arm.geometry_mode == '2d':
            nucleus[:, MORPHOLOGY_3D_SLICE] = 0.0
            cell[:, MORPHOLOGY_3D_SLICE] = 0.0
        if arm.control_mode == 'nucleus_duplicate':
            cell = nucleus.copy()
        elif arm.control_mode == 'cell_duplicate':
            nucleus = cell.copy()
        elif arm.control_mode in {'shuffled_cell_original_ncr', 'shuffled_cell_recomputed_ncr'}:
            self._ensure_shuffle_mapping()
            assert self._donor_indices is not None
            cell = cell[self._donor_indices].copy()
        ncr = self._audit_for_arm(arm)
        features: NCRFeatures | None = None
        if ncr is not None:
            statistics = self.ncr_statistics(arm)
            assert statistics is not None
            features = transform_ncr(ncr, statistics)
        nucleus, cell = apply_paired_ncr_channels(nucleus, cell, features, object_mode=arm.object_mode, ncr_mode=arm.ncr_mode)
        if arm.object_mode == 'nucleus':
            return (nucleus, None)
        if arm.object_mode == 'cell':
            return (None, cell)
        return (nucleus, cell)

    def iter_samples(self, arm: ExperimentArm, graph_ids: Iterable[str] | None=None) -> Iterator[GraphTokenSample]:
        selected = self.graph_ids if graph_ids is None else tuple((str(value) for value in graph_ids))
        unknown = set(selected) - set(self.graph_ids)
        if unknown:
            raise KeyError(f'unknown graph IDs: {sorted(unknown)[:5]}')
        nucleus, cell = self._arm_arrays(arm)
        for graph in selected:
            indices = self._graph_indices[graph]
            yield GraphTokenSample(graph_id=graph, anchor_ids=np.asarray([self.keys[index][1] for index in indices]), nucleus_tokens=None if nucleus is None else nucleus[indices].copy(), cell_tokens=None if cell is None else cell[indices].copy(), metadata=self.metadata[graph])

    def samples(self, arm: ExperimentArm, graph_ids: Iterable[str] | None=None) -> list[GraphTokenSample]:
        return list(self.iter_samples(arm, graph_ids))

def collate_variable_tokens(samples: Sequence[GraphTokenSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot collate an empty sample list')
    if len({sample.graph_id for sample in samples}) != len(samples):
        raise ValueError('collate batch contains duplicate graph IDs')

    def pad(branch: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        arrays = [getattr(sample, branch) for sample in samples]
        if all((value is None for value in arrays)):
            return (None, None)
        if any((value is None for value in arrays)):
            raise ValueError(f'{branch} is inconsistently active within a batch')
        concrete = [np.asarray(value, dtype=np.float32) for value in arrays if value is not None]
        if any((value.ndim != 2 or value.shape[1] != TOKEN_DIM for value in concrete)):
            raise ValueError(f'{branch} rows must have shape [N,15]')
        maximum = max((len(value) for value in concrete))
        tokens = np.zeros((len(concrete), maximum, TOKEN_DIM), dtype=np.float32)
        mask = np.zeros((len(concrete), maximum), dtype=bool)
        for index, value in enumerate(concrete):
            tokens[index, :len(value)] = value
            mask[index, :len(value)] = True
        return (tokens, mask)
    nucleus_tokens, nucleus_mask = pad('nucleus_tokens')
    cell_tokens, cell_mask = pad('cell_tokens')
    return {'graph_ids': [sample.graph_id for sample in samples], 'metadata': [sample.metadata for sample in samples], 'nucleus_tokens': nucleus_tokens, 'nucleus_mask': nucleus_mask, 'cell_tokens': cell_tokens, 'cell_mask': cell_mask}
