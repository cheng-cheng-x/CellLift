from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from dataclasses import asdict, dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterator, Mapping
import numpy as np
from celllift.conditional_geometry.common.probe_data import FoldProbeData, ProbeSample, RunningMoments, TransformedAnchorBatch
from celllift.conditional_geometry.mask_protocol import INNER_FOLDS, MASK_CONTEXT_DIM, MASK_PROTOCOL_ID, PROBE_SEED, RAY_DIM, stable_patient_fold

@dataclass(frozen=True)
class MaskProbeStatistics:
    ray_mean: tuple[float, ...]
    ray_std: tuple[float, ...]
    context_mean: tuple[float, ...]
    context_std: tuple[float, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    ncr3d_median: float
    training_anchors: int
    training_graphs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'MaskProbeStatistics':
        return cls(**{key: tuple(item) if isinstance(item, list) else item for key, item in value.items()})

class MaskFoldProbeData(FoldProbeData):
    statistics: MaskProbeStatistics | None

    def __init__(self, cfg: Mapping[str, Any], fold: int, *, graph_limit: int | None=None) -> None:
        if str(cfg.get('probe', {}).get('conditioning_mode')) != 'mask_rays_only':
            raise RuntimeError('mask mask_conditioning requires probe.conditioning_mode=mask_rays_only')
        super().__init__(cfg, fold, graph_limit=graph_limit)
        registered = int(cfg['probe'].get('inner_folds', INNER_FOLDS))
        self.inner_folds = np.asarray([stable_patient_fold(patient, self.fold, registered) for patient in self.patient_ids], dtype=np.int8)
        observed = set(map(int, self.inner_folds[self.roles == 'train']))
        if observed != set(range(registered)):
            raise RuntimeError(f'outer training lacks mask-mask_conditioning inner folds: {sorted(observed)}')
        self.context_raw = None
        self.statistics = None

    def prepare(self) -> MaskProbeStatistics:
        graph_count = len(self.graph_ids)
        sums = np.zeros((graph_count, RAY_DIM), dtype=np.float64)
        sums_sq = np.zeros_like(sums)
        counts = np.zeros(graph_count, dtype=np.int64)
        valid3d: list[np.ndarray] = []
        for batch in self._iter_raw():
            rays = batch.get('nucleus_rays')
            if rays is None:
                raise RuntimeError('mask rays were not materialised')
            indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            np.add.at(sums, indices, rays)
            np.add.at(sums_sq, indices, np.square(rays, dtype=np.float64))
            np.add.at(counts, indices, 1)
            training = self.roles[indices] == 'train'
            valid3d.append(batch['ncr3d'][training & batch['valid3d']].astype(np.float32, copy=True))
        if np.any(counts == 0):
            raise RuntimeError('one or more selected graphs have no mask anchors')
        valid_values = np.concatenate(valid3d)
        if not len(valid_values):
            raise RuntimeError('outer training has no valid NCR3D target')
        ncr3d_median = float(np.median(valid_values))
        divisor = counts[:, None]
        mean = sums / divisor
        std = np.sqrt(np.maximum(sums_sq / divisor - np.square(mean), 0.0))
        log_count = np.log1p(counts).astype(np.float64)[:, None]
        self.context_raw = np.concatenate((mean, std, log_count), axis=1).astype(np.float32)
        if self.context_raw.shape != (graph_count, MASK_CONTEXT_DIM):
            raise AssertionError('mask context width changed')
        training_graph = self.roles == 'train'
        ray_moments = RunningMoments(RAY_DIM)
        context_moments = RunningMoments(MASK_CONTEXT_DIM)
        geometry_moments = RunningMoments(8)
        ncr_moments = RunningMoments(1)
        context_moments.update(self.context_raw[training_graph])
        training_anchors = 0
        for batch in self._iter_raw():
            indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            training = self.roles[indices] == 'train'
            if not training.any():
                continue
            rays = batch['nucleus_rays']
            ncr3d = np.where(batch['valid3d'], batch['ncr3d'], ncr3d_median)
            targets = np.concatenate((batch['nucleus_3d'], batch['cell_3d'], ncr3d[:, None]), axis=1)
            ray_moments.update(rays[training])
            geometry_moments.update(targets[training, :8])
            ncr_moments.update(targets[training & batch['valid3d'], 8:9])
            training_anchors += int(training.sum())
        ray_mean, ray_std = ray_moments.finish()
        context_mean, context_std = context_moments.finish()
        geometry_mean, geometry_std = geometry_moments.finish()
        ncr_mean, ncr_std = ncr_moments.finish()
        self.statistics = MaskProbeStatistics(tuple(map(float, ray_mean)), tuple(map(float, ray_std)), tuple(map(float, context_mean)), tuple(map(float, context_std)), tuple(map(float, np.concatenate((geometry_mean, ncr_mean)))), tuple(map(float, np.concatenate((geometry_std, ncr_std)))), ncr3d_median, training_anchors, int(training_graph.sum()))
        return self.statistics

    def set_statistics(self, statistics: MaskProbeStatistics, context_raw: np.ndarray) -> None:
        self.statistics = statistics
        self.context_raw = np.asarray(context_raw, dtype=np.float32)

    def transformed_context(self) -> np.ndarray:
        if self.statistics is None or self.context_raw is None:
            raise RuntimeError('prepare must run before mask context access')
        return ((self.context_raw - np.asarray(self.statistics.context_mean)) / np.asarray(self.statistics.context_std)).astype(np.float32)

    def iter_transformed(self) -> Iterator[TransformedAnchorBatch]:
        if self.statistics is None:
            raise RuntimeError('prepare must run before transformed iteration')
        stats = self.statistics
        ray_mean, ray_std = (np.asarray(stats.ray_mean), np.asarray(stats.ray_std))
        target_mean, target_std = (np.asarray(stats.target_mean), np.asarray(stats.target_std))
        for batch in self._iter_raw():
            graph_indices = np.fromiter((self.graph_index[str(graph)] for graph in batch['graph_id']), np.int64, len(batch['graph_id']))
            ncr3d = np.where(batch['valid3d'], batch['ncr3d'], stats.ncr3d_median)
            targets = np.concatenate((batch['nucleus_3d'], batch['cell_3d'], ncr3d[:, None]), axis=1)
            yield TransformedAnchorBatch(graph_indices=graph_indices, graph_ids=batch['graph_id'], anchor_ids=batch['anchor_id'], anchor_2d=((batch['nucleus_rays'] - ray_mean) / ray_std).astype(np.float32), targets=((targets - target_mean) / target_std).astype(np.float32), target_valid=np.column_stack((np.ones((len(targets), 8), dtype=bool), batch['valid3d'])), roles=self.roles[graph_indices], inner_folds=self.inner_folds[graph_indices])

def collect_mask_probe_sample(data: MaskFoldProbeData, max_anchors: int, seed: int=PROBE_SEED) -> ProbeSample:
    total = int(data.statistics.training_anchors if data.statistics else data.node_counts[data.roles == 'train'].sum())
    probability = min(1.0, float(max_anchors) * 1.15 / max(1, total))
    rng = np.random.default_rng(int(seed) + data.fold * 1009)
    graphs: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    for batch in data.iter_transformed():
        eligible = batch.roles == 'train'
        chosen = eligible & (rng.random(len(eligible)) < probability)
        if chosen.any():
            graphs.append(batch.graph_indices[chosen])
            rays.append(batch.anchor_2d[chosen])
            targets.append(batch.targets[chosen])
            valid.append(batch.target_valid[chosen])
            inner.append(batch.inner_folds[chosen])
    values = [np.concatenate(items) for items in (graphs, rays, targets, valid, inner)]
    graph_values, ray_values, target_values, valid_values, inner_values = values
    if len(graph_values) > max_anchors:
        keep = rng.choice(len(graph_values), size=int(max_anchors), replace=False)
        keep.sort()
        graph_values, ray_values = (graph_values[keep], ray_values[keep])
        target_values, valid_values, inner_values = (target_values[keep], valid_values[keep], inner_values[keep])
    if len(graph_values) < min(1000, max_anchors):
        raise RuntimeError('mask probe sample is unexpectedly small')
    return ProbeSample(graph_values, ray_values, target_values, valid_values, inner_values)

def save_mask_prepared_fold(data: MaskFoldProbeData, root: Path) -> None:
    if data.statistics is None or data.context_raw is None:
        raise RuntimeError('mask fold data is not prepared')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'statistics.json').write_text(json.dumps(data.statistics.as_dict(), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    np.save(root / 'context_raw.npy', data.context_raw, allow_pickle=False)
    (root / 'manifest.json').write_text(json.dumps({'status': 'PASS', 'protocol_id': MASK_PROTOCOL_ID, 'fold': data.fold, 'conditioning': 'raw nucleus-mask 36 rays only; graph mean/std/count derived from rays', 'graph_ids': list(data.graph_ids), 'roles': list(map(str, data.roles)), 'official_test_touched': False}, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def load_mask_prepared_fold(data: MaskFoldProbeData, root: Path) -> None:
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'PASS' or manifest.get('protocol_id') != MASK_PROTOCOL_ID or int(manifest.get('fold', -1)) != data.fold or (tuple(map(str, manifest.get('graph_ids', ()))) != data.graph_ids) or (tuple(map(str, manifest.get('roles', ()))) != tuple(map(str, data.roles))) or (manifest.get('official_test_touched') is not False):
        raise RuntimeError(f'mask prepared fold provenance mismatch: {root}')
    stats = MaskProbeStatistics.from_dict(json.loads((root / 'statistics.json').read_text(encoding='utf-8')))
    context = np.load(root / 'context_raw.npy', allow_pickle=False)
    if context.shape != (len(data.graph_ids), MASK_CONTEXT_DIM) or not np.isfinite(context).all():
        raise RuntimeError(f'invalid mask prepared context: {root}')
    data.set_statistics(stats, context)
