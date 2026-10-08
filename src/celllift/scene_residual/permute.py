from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .data import SceneCache
from .io_utils import atomic_json
ZPERM_FILE = 'zperm.npz'
ZPERM_SEED = 20260917

def _derangement(count: int, generator: np.random.Generator) -> np.ndarray:
    order = np.arange(count)
    if count < 2:
        return order
    for _ in range(64):
        candidate = generator.permutation(count)
        if not np.any(candidate == order):
            return candidate
    return np.roll(order, 1)

def build_permutation(cfg: Mapping[str, Any], dataset: str, seed: int=ZPERM_SEED) -> dict[str, Any]:
    cache = SceneCache(cfg, dataset)
    generator = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    degenerate = 0
    for graph_id in sorted(cache.entries):
        graph = cache.get(graph_id)
        count = len(graph.center_z)
        order = _derangement(count, generator)
        if np.array_equal(order, np.arange(count)):
            degenerate += 1
        arrays[graph_id] = order.astype(np.int32)
    destination = Path(cfg['paths']['data_root']) / dataset / ZPERM_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    payload = {'status': 'PASS', 'dataset': dataset, 'seed': int(seed), 'graphs': len(arrays), 'degenerate_graphs': int(degenerate), 'path': str(destination)}
    atomic_json(destination.with_suffix('.json'), payload)
    return payload

def load_permutation(cfg: Mapping[str, Any], dataset: str) -> dict[str, np.ndarray]:
    path = Path(cfg['paths']['data_root']) / dataset / ZPERM_FILE
    if not path.is_file():
        return {}
    with np.load(path) as value:
        return {key: value[key] for key in value.files}
