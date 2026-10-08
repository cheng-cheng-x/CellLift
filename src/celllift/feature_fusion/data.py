from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.cross_fitted_correction.data import read_parquet
from .protocol import ARMS

@dataclass(frozen=True)
class Sample:
    graph_id: str
    patient_id: str
    label_id: int
    fold: int
    rgb_features: np.ndarray
    paper_logits: np.ndarray
    nucleus_summary: np.ndarray
    cell_summary: np.ndarray

def _feature_rows(cfg: Mapping[str, Any], fold: int, seed: int) -> dict[str, np.ndarray]:
    path = Path(cfg['paths']['result_root']) / 'rgb_features' / f'seed_{seed}' / f'fold_{fold}.npz'
    saved = np.load(path, allow_pickle=False)
    return {key: saved[key] for key in saved.files}

def _geometry_cache(cfg: Mapping[str, Any], geometry_id: str, fold: int, seed: int, role: str) -> dict[str, Any]:
    path = Path(cfg['paths']['cross_fitted_correction_result_root']) / 'cache' / 'multistat' / geometry_id / f'seed_{seed}' / f'fold_{fold}' / f'{role}.npz'
    saved = np.load(path, allow_pickle=False)
    return {key: saved[key] for key in saved.files}

def load_samples(cfg: Mapping[str, Any], *, fold: int, seed: int, arm_id: str) -> tuple[list[Sample], list[Sample]]:
    arm = ARMS[arm_id]
    feature = _feature_rows(cfg, fold, seed)
    all_ids = feature['graph_ids'].astype(str)
    feature_index = {graph_id: index for index, graph_id in enumerate(all_ids)}
    reference_train = _geometry_cache(cfg, 'G1', fold, seed, 'train')
    reference_validation = _geometry_cache(cfg, 'G1', fold, seed, 'validation')
    ids = np.concatenate((reference_train['graph_ids'].astype(str), reference_validation['graph_ids'].astype(str)))
    missing = [graph_id for graph_id in ids if graph_id not in feature_index]
    if missing:
        raise RuntimeError(f'geometry IDs missing RGB features: {missing[:3]}')
    order = np.asarray([feature_index[graph_id] for graph_id in ids])
    partitions = np.concatenate((np.zeros(len(reference_train['graph_ids']), np.uint8), np.ones(len(reference_validation['graph_ids']), np.uint8)))
    if arm.geometry_id is None:
        nucleus = np.zeros((len(ids), 206), np.float32)
        cell = np.zeros_like(nucleus)
        nucleus[:, -1] = cell[:, -1] = 0.0
    else:
        train = _geometry_cache(cfg, arm.geometry_id, fold, seed, 'train')
        validation = _geometry_cache(cfg, arm.geometry_id, fold, seed, 'validation')
        geom_ids = np.concatenate((train['graph_ids'].astype(str), validation['graph_ids'].astype(str)))
        if not np.array_equal(ids, geom_ids):
            raise RuntimeError(f'RGB/geometry order mismatch for fold {fold} {arm_id}')
        nucleus = np.concatenate((train['nucleus'], validation['nucleus'])).astype(np.float32)
        cell = np.concatenate((train['cell'], validation['cell'])).astype(np.float32)
    rows = [Sample(graph_id=str(ids[i]), patient_id=str(feature['patient_ids'][order[i]]), label_id=int(feature['labels'][order[i]]), fold=fold, rgb_features=feature['features'][order[i]].astype(np.float32), paper_logits=feature['logits'][order[i]].astype(np.float32), nucleus_summary=nucleus[i], cell_summary=cell[i]) for i in range(len(ids))]
    return ([row for row, part in zip(rows, partitions) if part == 0], [row for row, part in zip(rows, partitions) if part == 1])
