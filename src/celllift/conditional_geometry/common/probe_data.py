from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import math
import re
from dataclasses import asdict, dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterator, Mapping
import numpy as np
from celllift.conditional_geometry.protocol import ANCHOR_2D_DIM, CONTEXT_DIM, INNER_FOLDS, PROBE_SEED, PROTOCOL_ID, RGB_DIM, TARGET_COLUMNS, TARGET_DIM, stable_patient_fold
TOKEN_2D_COLUMNS = ('log_area', 'log_equivalent_radius', 'log_major_minor', 'circularity', 'cos_2theta', 'sin_2theta', 'x_normalized', 'y_normalized', 'border_flag')
TOKEN_3D_COLUMNS = ('log_volume', 'log_a_b', 'log_b_c', 'long_axis_z_squared')

class RunningMoments:

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sum_sq = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.width or (not np.isfinite(array).all()):
            raise ValueError(f'moment input must be finite [N,{self.width}]')
        self.count += len(array)
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError('insufficient rows for standardization')
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 1e-12)
        return (mean.astype(np.float32), np.sqrt(variance).astype(np.float32))

@dataclass(frozen=True)
class ProbeStatistics:
    anchor_mean: tuple[float, ...]
    anchor_std: tuple[float, ...]
    rgb_mean: tuple[float, ...]
    rgb_std: tuple[float, ...]
    context_mean: tuple[float, ...]
    context_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    ncr2d_median: float
    ncr3d_median: float
    training_anchors: int
    training_graphs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'ProbeStatistics':
        return cls(**{key: tuple(item) if isinstance(item, list) else item for key, item in value.items()})

@dataclass(frozen=True)
class TransformedAnchorBatch:
    graph_indices: np.ndarray
    graph_ids: np.ndarray
    anchor_ids: np.ndarray
    anchor_2d: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray
    roles: np.ndarray
    inner_folds: np.ndarray

@dataclass(frozen=True)
class ProbeSample:
    graph_indices: np.ndarray
    anchor_2d: np.ndarray
    targets: np.ndarray
    target_valid: np.ndarray
    inner_folds: np.ndarray

def _read_rows(path: Path, *, filters: list[tuple[str, str, Any]] | None=None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, filters=filters, partitioning=None).to_pylist()

def _suffix(path: Path) -> int:
    match = re.search('(\\d+)$', path.stem)
    if match is None:
        raise ValueError(f'cache shard has no numeric suffix: {path}')
    return int(match.group(1))

def _shards(root: Path, folder: str) -> dict[int, Path]:
    result = {_suffix(path): path for path in sorted((root / folder).glob('*.parquet'))}
    if not result:
        raise FileNotFoundError(root / folder)
    return result

class FoldProbeData:

    def __init__(self, cfg: Mapping[str, Any], fold: int, *, graph_limit: int | None=None) -> None:
        self.cfg = dict(cfg)
        self.fold = int(fold)
        self.set_encoding_root = Path(cfg['paths']['set_encoding_data_root'])
        self.include_nucleus_rays = str(cfg.get('probe', {}).get('conditioning_mode', 'rich_2d')) == 'mask_rays_only'
        if self.include_nucleus_rays:
            from celllift.conditional_geometry.mask_protocol import stable_patient_fold as configured_patient_fold
        else:
            configured_patient_fold = stable_patient_fold
        model_root = Path(cfg['paths']['model_input_root'])
        graph_rows = _read_rows(model_root / '03_graph_cache' / 'graph_index.parquet', filters=[('split', 'in', ['TRAIN', 'train', 'Train'])])
        if any((str(row['split']).lower() != 'train' for row in graph_rows)):
            raise RuntimeError('non-TRAIN row crossed graph-index predicate')
        if graph_limit is not None:
            ordered = sorted(graph_rows, key=lambda row: str(row['graph_id']))
            validation = [row for row in ordered if int(row['validation_fold']) == self.fold]
            training = [row for row in ordered if int(row['validation_fold']) != self.fold]
            validation_quota = max(1, min(len(validation), int(graph_limit) // 2))
            training_quota = max(1, min(len(training), int(graph_limit) - validation_quota))
            inner_count = int(cfg['probe'].get('inner_folds', INNER_FOLDS))
            training_buckets = [[] for _ in range(inner_count)]
            for row in training:
                inner = configured_patient_fold(str(row['patient_id']), self.fold, inner_count)
                training_buckets[inner].append(row)
            balanced_training: list[dict[str, Any]] = []
            cursor = 0
            while len(balanced_training) < training_quota:
                made_progress = False
                for bucket in training_buckets:
                    if cursor < len(bucket) and len(balanced_training) < training_quota:
                        balanced_training.append(bucket[cursor])
                        made_progress = True
                if not made_progress:
                    break
                cursor += 1
            graph_rows = sorted(validation[:validation_quota] + balanced_training, key=lambda row: str(row['graph_id']))
        if not graph_rows:
            raise RuntimeError('outer TRAIN graph selection is empty')
        self.graph_ids = tuple((str(row['graph_id']) for row in graph_rows))
        self.graph_index = {graph: index for index, graph in enumerate(self.graph_ids)}
        self.patient_ids = tuple((str(row['patient_id']) for row in graph_rows))
        self.roles = np.asarray(['validation' if int(row['validation_fold']) == self.fold else 'train' for row in graph_rows], dtype=object)
        self.inner_folds = np.asarray([configured_patient_fold(patient, self.fold, int(cfg['probe'].get('inner_folds', INNER_FOLDS))) for patient in self.patient_ids], dtype=np.int8)
        self.node_counts = np.asarray([int(row['node_count']) for row in graph_rows], dtype=np.int64)
        if not np.any(self.roles == 'train') or not np.any(self.roles == 'validation'):
            raise RuntimeError('outer fold lacks train or validation graphs')
        registered_inner = int(cfg['probe'].get('inner_folds', INNER_FOLDS))
        observed_inner = set(map(int, self.inner_folds[self.roles == 'train']))
        if observed_inner != set(range(registered_inner)):
            raise RuntimeError(f'outer training lacks registered inner folds: {sorted(observed_inner)}')
        rgb_path = self.set_encoding_root / '05_rgb_features' / f'fold_{self.fold:02d}' / 'seed_42' / 'rgb_features.parquet'
        if not rgb_path.is_file():
            raise FileNotFoundError(rgb_path)
        rgb_rows = _read_rows(rgb_path, filters=[('graph_id', 'in', list(self.graph_ids))])
        rgb_map = {str(row['graph_id']): np.asarray(row['rgb_feature'], dtype=np.float32) for row in rgb_rows}
        missing = sorted(set(self.graph_ids) - rgb_map.keys())
        if missing:
            raise RuntimeError(f'RGB cache misses outer-TRAIN graphs: {missing[:5]}')
        self.rgb_raw = np.stack([rgb_map[graph] for graph in self.graph_ids])
        if self.rgb_raw.shape != (len(self.graph_ids), RGB_DIM) or not np.isfinite(self.rgb_raw).all():
            raise RuntimeError('invalid RGB graph table')
        nucleus = _shards(self.set_encoding_root, '02_nucleus_tokens')
        cell = _shards(self.set_encoding_root, '03_cell_tokens')
        ncr = _shards(self.set_encoding_root, '04_ncr_features')
        if nucleus.keys() != cell.keys() or nucleus.keys() != ncr.keys():
            raise RuntimeError('nucleus/cell/NCR shard suffixes differ')
        geometry = _shards(self.set_encoding_root, '01_dual_geometry') if self.include_nucleus_rays else None
        if geometry is not None and geometry.keys() != nucleus.keys():
            raise RuntimeError('dual-geometry/token shard suffixes differ')
        self.shard_paths = tuple(((nucleus[key], cell[key], ncr[key], None if geometry is None else geometry[key]) for key in sorted(nucleus)))
        self.context_raw: np.ndarray | None = None
        self.statistics: ProbeStatistics | None = None
        self._raw_cache: list[dict[str, np.ndarray]] = []
        self._raw_cache_ready = False

    def _iter_raw(self) -> Iterator[dict[str, np.ndarray]]:
        import pyarrow.parquet as pq
        if self._raw_cache_ready:
            yield from self._raw_cache
            return
        token_columns = ['graph_id', 'anchor_id', *TOKEN_2D_COLUMNS, *TOKEN_3D_COLUMNS]
        ncr_columns = ['graph_id', 'anchor_id', 'raw_log_ncr_2d', 'valid_ncr_2d', 'raw_log_ncr_3d', 'valid_ncr_3d']
        graph_filter = [('graph_id', 'in', list(self.graph_ids))]
        for nucleus_path, cell_path, ncr_path, geometry_path in self.shard_paths:
            nt = pq.read_table(nucleus_path, columns=token_columns, filters=graph_filter, partitioning=None)
            ct = pq.read_table(cell_path, columns=token_columns, filters=graph_filter, partitioning=None)
            rt = pq.read_table(ncr_path, columns=ncr_columns, filters=graph_filter, partitioning=None)
            if not len(nt) == len(ct) == len(rt):
                raise RuntimeError(f'paired shard length mismatch at {nucleus_path.name}')
            nrows, crows, rrows = (nt.to_pydict(), ct.to_pydict(), rt.to_pydict())
            ng = np.asarray(nrows['graph_id'], dtype=object)
            cg = np.asarray(crows['graph_id'], dtype=object)
            rg = np.asarray(rrows['graph_id'], dtype=object)
            na = np.asarray(nrows['anchor_id'])
            ca = np.asarray(crows['anchor_id'])
            ra = np.asarray(rrows['anchor_id'])
            if not (np.array_equal(ng, cg) and np.array_equal(ng, rg) and np.array_equal(na, ca) and np.array_equal(na, ra)):
                raise RuntimeError(f'paired anchor order mismatch at {nucleus_path.name}')
            if not len(ng):
                continue
            keep = np.ones(len(ng), dtype=bool)
            selected = {'graph_id': ng[keep], 'anchor_id': na[keep], 'nucleus_2d': np.column_stack([np.asarray(nrows[name], np.float32)[keep] for name in TOKEN_2D_COLUMNS]), 'cell_2d': np.column_stack([np.asarray(crows[name], np.float32)[keep] for name in TOKEN_2D_COLUMNS]), 'nucleus_3d': np.column_stack([np.asarray(nrows[name], np.float32)[keep] for name in TOKEN_3D_COLUMNS]), 'cell_3d': np.column_stack([np.asarray(crows[name], np.float32)[keep] for name in TOKEN_3D_COLUMNS]), 'ncr2d': np.asarray(rrows['raw_log_ncr_2d'], np.float32)[keep], 'valid2d': np.asarray(rrows['valid_ncr_2d'], bool)[keep], 'ncr3d': np.asarray(rrows['raw_log_ncr_3d'], np.float32)[keep], 'valid3d': np.asarray(rrows['valid_ncr_3d'], bool)[keep]}
            if geometry_path is not None:
                gt = pq.read_table(geometry_path, columns=['graph_id', 'anchor_id', 'object_type', 'nucleus_rays_um'], filters=[('object_type', '=', 'nucleus'), *graph_filter], partitioning=None)
                gg_unsorted = np.asarray(gt['graph_id'].to_pylist(), dtype=object)
                ga_unsorted = np.asarray(gt['anchor_id'].to_numpy(zero_copy_only=False))
                order = np.lexsort((ga_unsorted, gg_unsorted))
                gg, ga = (gg_unsorted[order], ga_unsorted[order])
                if not np.array_equal(ng, gg) or not np.array_equal(na, ga):
                    raise RuntimeError(f'mask-ray/source key mismatch at {geometry_path.name}')
                ray_array = gt['nucleus_rays_um'].combine_chunks()
                offsets = np.asarray(ray_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
                if len(offsets) != len(gg_unsorted) + 1 or np.any(np.diff(offsets) != 36):
                    raise RuntimeError(f'non-36 mask-ray list at {geometry_path.name}')
                flat_rays = np.asarray(ray_array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
                rays = flat_rays.reshape(len(gg_unsorted), 36)[order]
                if rays.shape != (len(ng), 36) or not np.isfinite(rays).all() or np.any(rays < 0):
                    raise RuntimeError(f'invalid 36-ray nucleus-mask input at {geometry_path.name}')
                selected['nucleus_rays'] = rays
            self._raw_cache.append(selected)
            yield selected
        self._raw_cache_ready = True

    def prepare(self) -> ProbeStatistics:
        graph_count = len(self.graph_ids)
        sums_n = np.zeros((graph_count, 9), dtype=np.float64)
        sums_c = np.zeros((graph_count, 9), dtype=np.float64)
        sq_n = np.zeros((graph_count, 9), dtype=np.float64)
        sq_c = np.zeros((graph_count, 9), dtype=np.float64)
        counts = np.zeros(graph_count, dtype=np.int64)
        valid2d: list[np.ndarray] = []
        valid3d: list[np.ndarray] = []
        for batch in self._iter_raw():
            indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            np.add.at(sums_n, indices, batch['nucleus_2d'])
            np.add.at(sums_c, indices, batch['cell_2d'])
            np.add.at(sq_n, indices, np.square(batch['nucleus_2d'], dtype=np.float64))
            np.add.at(sq_c, indices, np.square(batch['cell_2d'], dtype=np.float64))
            np.add.at(counts, indices, 1)
            train = self.roles[indices] == 'train'
            valid2d.append(batch['ncr2d'][train & batch['valid2d']].astype(np.float32, copy=True))
            valid3d.append(batch['ncr3d'][train & batch['valid3d']].astype(np.float32, copy=True))
        if np.any(counts == 0):
            raise RuntimeError('one or more selected graphs have no paired anchors')
        ncr2d_median = float(np.median(np.concatenate(valid2d)))
        ncr3d_median = float(np.median(np.concatenate(valid3d)))
        del valid2d, valid3d
        divisor = counts[:, None]
        mean_n, mean_c = (sums_n / divisor, sums_c / divisor)
        std_n = np.sqrt(np.maximum(sq_n / divisor - np.square(mean_n), 0.0))
        std_c = np.sqrt(np.maximum(sq_c / divisor - np.square(mean_c), 0.0))
        log_count = np.log1p(counts).astype(np.float64)[:, None]
        self.context_raw = np.concatenate((mean_n, std_n, mean_c, std_c, log_count, log_count), axis=1).astype(np.float32)
        if self.context_raw.shape != (graph_count, CONTEXT_DIM):
            raise AssertionError('context width changed')
        train_graph = self.roles == 'train'
        rgb_moments, context_moments = (RunningMoments(RGB_DIM), RunningMoments(CONTEXT_DIM))
        rgb_moments.update(self.rgb_raw[train_graph])
        context_moments.update(self.context_raw[train_graph])
        anchor_moments = RunningMoments(ANCHOR_2D_DIM)
        geometry_target_moments, ncr_target_moments = (RunningMoments(8), RunningMoments(1))
        training_anchors = 0
        for batch in self._iter_raw():
            indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            train = self.roles[indices] == 'train'
            if not train.any():
                continue
            ncr2d = np.where(batch['valid2d'], batch['ncr2d'], ncr2d_median)
            ncr3d = np.where(batch['valid3d'], batch['ncr3d'], ncr3d_median)
            anchor = np.concatenate((batch['nucleus_2d'], batch['cell_2d'], ncr2d[:, None], (~batch['valid2d'])[:, None]), axis=1)
            targets = np.concatenate((batch['nucleus_3d'], batch['cell_3d'], ncr3d[:, None]), axis=1)
            anchor_moments.update(anchor[train])
            geometry_target_moments.update(targets[train, :8])
            ncr_target_moments.update(targets[train & batch['valid3d'], 8:9])
            training_anchors += int(train.sum())
        anchor_mean, anchor_std = anchor_moments.finish()
        geometry_mean, geometry_std = geometry_target_moments.finish()
        ncr_mean, ncr_std = ncr_target_moments.finish()
        target_mean, target_std = (np.concatenate((geometry_mean, ncr_mean)), np.concatenate((geometry_std, ncr_std)))
        rgb_mean, rgb_std = rgb_moments.finish()
        context_mean, context_std = context_moments.finish()
        stats = ProbeStatistics(tuple(map(float, anchor_mean)), tuple(map(float, anchor_std)), tuple(map(float, rgb_mean)), tuple(map(float, rgb_std)), tuple(map(float, context_mean)), tuple(map(float, context_std)), tuple(map(float, target_mean)), tuple(map(float, target_std)), ncr2d_median, ncr3d_median, training_anchors, int(train_graph.sum()))
        self.statistics = stats
        return stats

    def set_statistics(self, statistics: ProbeStatistics, context_raw: np.ndarray) -> None:
        self.statistics = statistics
        self.context_raw = np.asarray(context_raw, dtype=np.float32)

    def transformed_graph_tables(self) -> tuple[np.ndarray, np.ndarray]:
        if self.statistics is None or self.context_raw is None:
            raise RuntimeError('prepare must run before graph table access')
        stats = self.statistics
        rgb = (self.rgb_raw - np.asarray(stats.rgb_mean)) / np.asarray(stats.rgb_std)
        context = (self.context_raw - np.asarray(stats.context_mean)) / np.asarray(stats.context_std)
        return (rgb.astype(np.float32), context.astype(np.float32))

    def iter_transformed(self) -> Iterator[TransformedAnchorBatch]:
        if self.statistics is None:
            raise RuntimeError('prepare must run before transformed iteration')
        stats = self.statistics
        anchor_mean, anchor_std = (np.asarray(stats.anchor_mean), np.asarray(stats.anchor_std))
        target_mean, target_std = (np.asarray(stats.target_mean), np.asarray(stats.target_std))
        for batch in self._iter_raw():
            graph_indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            ncr2d = np.where(batch['valid2d'], batch['ncr2d'], stats.ncr2d_median)
            ncr3d = np.where(batch['valid3d'], batch['ncr3d'], stats.ncr3d_median)
            anchor = np.concatenate((batch['nucleus_2d'], batch['cell_2d'], ncr2d[:, None], (~batch['valid2d'])[:, None]), axis=1)
            targets = np.concatenate((batch['nucleus_3d'], batch['cell_3d'], ncr3d[:, None]), axis=1)
            yield TransformedAnchorBatch(graph_indices=graph_indices, graph_ids=batch['graph_id'], anchor_ids=batch['anchor_id'], anchor_2d=((anchor - anchor_mean) / anchor_std).astype(np.float32), targets=((targets - target_mean) / target_std).astype(np.float32), target_valid=np.column_stack((np.ones((len(targets), 8), dtype=bool), batch['valid3d'])), roles=self.roles[graph_indices], inner_folds=self.inner_folds[graph_indices])

def collect_probe_sample(data: FoldProbeData, max_anchors: int, seed: int=PROBE_SEED) -> ProbeSample:
    total = int(data.statistics.training_anchors if data.statistics else data.node_counts[data.roles == 'train'].sum())
    probability = min(1.0, float(max_anchors) * 1.15 / max(1, total))
    rng = np.random.default_rng(int(seed) + data.fold * 1009)
    graphs: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    for batch in data.iter_transformed():
        eligible = batch.roles == 'train'
        chosen = eligible & (rng.random(len(eligible)) < probability)
        if chosen.any():
            graphs.append(batch.graph_indices[chosen])
            anchors.append(batch.anchor_2d[chosen])
            targets.append(batch.targets[chosen])
            valid.append(batch.target_valid[chosen])
            inner.append(batch.inner_folds[chosen])
    graph_values = np.concatenate(graphs)
    anchor_values = np.concatenate(anchors)
    target_values = np.concatenate(targets)
    valid_values = np.concatenate(valid)
    inner_values = np.concatenate(inner)
    if len(graph_values) > max_anchors:
        keep = rng.choice(len(graph_values), size=int(max_anchors), replace=False)
        keep.sort()
        graph_values, anchor_values = (graph_values[keep], anchor_values[keep])
        target_values, valid_values, inner_values = (target_values[keep], valid_values[keep], inner_values[keep])
    if len(graph_values) < min(1000, max_anchors):
        raise RuntimeError('probe sample is unexpectedly small')
    return ProbeSample(graph_values, anchor_values, target_values, valid_values, inner_values)

def save_prepared_fold(data: FoldProbeData, root: Path) -> None:
    if data.statistics is None or data.context_raw is None:
        raise RuntimeError('fold data is not prepared')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'statistics.json').write_text(json.dumps(data.statistics.as_dict(), indent=2, sort_keys=True) + '\n')
    np.save(root / 'context_raw.npy', data.context_raw, allow_pickle=False)
    (root / 'manifest.json').write_text(json.dumps({'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'fold': data.fold, 'graph_ids': list(data.graph_ids), 'roles': list(map(str, data.roles)), 'official_test_touched': False}, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def load_prepared_fold(data: FoldProbeData, root: Path) -> None:
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'PASS' or manifest.get('protocol_id') != PROTOCOL_ID or int(manifest.get('fold', -1)) != data.fold or (tuple(map(str, manifest.get('graph_ids', ()))) != data.graph_ids) or (tuple(map(str, manifest.get('roles', ()))) != tuple(map(str, data.roles))) or (manifest.get('official_test_touched') is not False):
        raise RuntimeError(f'prepared fold provenance mismatch: {root}')
    stats = ProbeStatistics.from_dict(json.loads((root / 'statistics.json').read_text()))
    context = np.load(root / 'context_raw.npy', allow_pickle=False)
    if context.shape != (len(data.graph_ids), CONTEXT_DIM) or not np.isfinite(context).all():
        raise RuntimeError(f'invalid prepared context: {root}')
    data.set_statistics(stats, context)
