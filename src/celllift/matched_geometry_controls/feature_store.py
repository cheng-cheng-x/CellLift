from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.runtime import json
import numpy as np
from .io_utils import atomic_json, atomic_npz, atomic_parquet, safe_component
from .shuffle import GraphShuffleRecord, build_donor_map, replace_tail
from .tokens import FeatureNormalizer, make_tokens, preaggregate

@dataclass(frozen=True)
class ModalRow:
    graph_id: str
    split: str
    patient_id: str
    wsi_id: str
    roi_id: str
    fold: int | None
    final_fold: int | None
    label_id: int | None
    path: str
    count: int

def build_modal_file(input_path: str | Path, scene_path: str | Path, destination: str | Path) -> dict[str, Any]:
    import torch
    input_payload = torch.load(input_path, map_location='cpu', weights_only=False)
    scene = torch.load(scene_path, map_location='cpu', weights_only=False)
    graph = input_payload['graph']
    metadata = dict(input_payload['metadata'])
    ids = graph.nucleus_id.numpy()
    selected_ids = scene['nucleus_id'].numpy()
    if not np.array_equal(ids, selected_ids):
        raise RuntimeError('input/selected-scene nucleus order mismatch')
    rays = graph.nucleus_rays_um.numpy().astype(np.float32)
    direct = scene['direct_geometry9'].numpy().astype(np.float32)
    valid = scene['valid_ncr'].numpy().astype(bool)
    if len(ids) != len(direct) or not np.all(np.isfinite(direct[:, :8])):
        raise RuntimeError('invalid direct feature coverage')
    atomic_npz(destination, nucleus_id=ids, rays36=rays, direct9=direct, valid_ncr=valid)
    return {**metadata, 'graph_id': str(metadata['graph_id']), 'path': str(destination), 'count': len(ids), 'invalid_ncr': int((~valid).sum())}

def read_index(path: str | Path) -> list[ModalRow]:
    import pyarrow.parquet as pq
    names = set(ModalRow.__dataclass_fields__)
    output = []
    for row in pq.read_table(path, partitioning=None).to_pylist():
        selected = {key: value for key, value in row.items() if key in names}
        selected.setdefault('final_fold', None)
        output.append(ModalRow(**selected))
    return output

def _load(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as value:
        return {key: value[key].copy() for key in value.files}

class FoldModalStore:

    def __init__(self, rows: Sequence[ModalRow], fold: int | None, *, dataset: str, residual_root: str | Path | None=None, donor_map: Mapping[str, str] | None=None, training_splits: Sequence[str]=('train',), use_final_fold: bool=False):
        self.rows = list(rows)
        self.fold = None if fold is None else int(fold)
        self.dataset = dataset
        self.by_id = {row.graph_id: row for row in rows}
        self.arrays = {row.graph_id: _load(row.path) for row in rows}
        allowed = set(training_splits)
        fold_field = 'final_fold' if use_final_fold else 'fold'
        self.training_ids = [row.graph_id for row in rows if row.split in allowed and (self.fold is None or getattr(row, fold_field) != self.fold)]
        if not self.training_ids:
            raise RuntimeError('empty training set for fold-local normalization')
        ray_train = np.concatenate([self.arrays[g]['rays36'] for g in self.training_ids])
        direct_train = np.concatenate([self.arrays[g]['direct9'] for g in self.training_ids])
        valid_train = np.concatenate([self.arrays[g]['valid_ncr'] for g in self.training_ids])
        self.ray_mean = ray_train.mean(0).astype(np.float32)
        self.ray_std = np.maximum(ray_train.std(0), 1e-06).astype(np.float32)
        self.geometry_normalizer = FeatureNormalizer.fit(direct_train, valid_train, np.ones(len(direct_train), bool))
        self.residual_root = None if residual_root is None else Path(residual_root)
        self.residual_arrays = None
        if self.residual_root is not None:
            self.residual_arrays = {row.graph_id: _load(self.residual_root / f'{safe_component(row.graph_id)}.npz')['residual9'] for row in rows if (self.residual_root / f'{safe_component(row.graph_id)}.npz').is_file()}
        self.donor_map = dict(donor_map or {})

    def geometry(self, graph_id: str, mode: str) -> np.ndarray:
        base = self.arrays[graph_id]
        if mode == 'direct':
            return self.geometry_normalizer.transform(base['direct9'], base['valid_ncr'])
        if mode == 'residual':
            if self.residual_arrays is None or graph_id not in self.residual_arrays:
                raise RuntimeError(f'residual geometry missing for {graph_id}')
            return self.residual_arrays[graph_id]
        raise ValueError(mode)

    def tokens(self, graph_id: str, mode: str, shuffled: bool=False):
        rays = ((self.arrays[graph_id]['rays36'] - self.ray_mean) / self.ray_std).astype(np.float32)
        geometry = self.geometry(graph_id, mode)
        nucleus, cell = make_tokens(rays, geometry)
        if shuffled:
            donor_id = self.donor_map[graph_id]
            donor_geometry = self.geometry(donor_id, mode)
            donor_n, donor_c = make_tokens(np.zeros((len(donor_geometry), 36), np.float32), donor_geometry)
            nucleus = replace_tail(nucleus, donor_n[:, 36:])
            cell = replace_tail(cell, donor_c[:, 36:])
        return (nucleus, cell)

def build_shuffle_for_fold(rows: Sequence[ModalRow], fold: int, dataset: str) -> dict[str, str]:
    group_key = 'wsi_id' if dataset == 'bracs' else 'patient_id'
    registered = []
    for row in rows:
        if row.split == 'test':
            continue
        if row.split in {'val', 'validation'}:
            role = 'external'
        else:
            role = 'heldout' if row.fold == fold else 'train'
        registered.append(GraphShuffleRecord(row.graph_id, getattr(row, group_key), role, row.count))
    return build_donor_map(registered)

def write_meanpool_cache(store: FoldModalStore, rows: Sequence[ModalRow], arm: str, destination: str | Path) -> dict:
    mode = 'direct' if arm in {'D', 'DS'} else 'residual'
    shuffled = arm in {'DS', 'RS'}
    root = Path(destination)
    graph_ids = []
    nucleus_means = []
    cell_means = []
    counts = []
    for row in rows:
        nucleus, cell = store.tokens(row.graph_id, mode, shuffled)
        n_mean, count = preaggregate(nucleus)
        c_mean, c_count = preaggregate(cell)
        if count != c_count or count != row.count:
            raise RuntimeError(f'MeanPool count mismatch for {row.graph_id}')
        graph_ids.append(row.graph_id)
        nucleus_means.append(n_mean[0])
        cell_means.append(c_mean[0])
        counts.append(count)
    packed = root / 'features.npz'
    atomic_npz(packed, graph_id=np.asarray(graph_ids, dtype=np.str_), nucleus=np.asarray(nucleus_means, np.float32), cell=np.asarray(cell_means, np.float32), count=np.asarray(counts, np.int32))
    manifest = {'status': 'PASS', 'fold': store.fold, 'arm': arm, 'graphs': len(graph_ids), 'anchors': int(np.sum(counts)), 'packed_path': str(packed)}
    atomic_json(root / 'manifest.json', manifest)
    return manifest

class MeanPoolStore:
    preaggregated = True

    def __init__(self, manifest_path: str | Path):
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS':
            raise RuntimeError('MeanPool cache is incomplete')
        self.values = {}
        if manifest.get('packed_path'):
            value = _load(manifest['packed_path'])
            for graph_id, nucleus, cell, count in zip(value['graph_id'], value['nucleus'], value['cell'], value['count']):
                self.values[str(graph_id)] = (nucleus[None], cell[None], int(count))
        else:
            for row in manifest['items']:
                value = _load(row['path'])
                self.values[str(row['graph_id'])] = (value['nucleus'], value['cell'], int(value['nucleus_count']))

    def sample_tokens(self, graph_id: str):
        return self.values[graph_id]
