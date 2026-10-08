from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass, field
import numpy as np
from .config import EDGE_2D, EDGE_3D, NODE_2D, NODE_3D
SCHEMA = 2

@dataclass
class _Moments:
    dim: int
    total: np.ndarray = field(init=False)
    second: np.ndarray = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.total = np.zeros(self.dim, np.float64)
        self.second = np.zeros(self.dim, np.float64)

    def add(self, rows: np.ndarray) -> None:
        if len(rows) == 0:
            return
        value = np.asarray(rows, np.float64)
        self.total += value.sum(0)
        self.second += np.square(value).sum(0)
        self.count += int(len(value))

    @property
    def ready(self) -> bool:
        return self.count > 0

    def mean(self) -> np.ndarray:
        return self.total / self.count

    def mean_square(self) -> np.ndarray:
        return self.second / self.count

def _lite(scene: dict) -> dict[str, np.ndarray]:
    node2d = np.asarray(scene['node2d'], np.float32)
    node3d = np.asarray(scene['node3d'], np.float32)
    include = np.asarray(scene.get('include', np.ones(len(node2d), bool)), bool).reshape(-1)
    valid3d = np.asarray(scene.get('valid3d', np.zeros(len(node2d), bool)), bool).reshape(-1)
    finite2 = np.isfinite(node2d).all(axis=1) if len(node2d) else np.zeros(0, bool)
    finite3 = np.isfinite(node3d).all(axis=1) if len(node3d) else np.zeros(0, bool)
    include = include & finite2
    valid3d = include & valid3d & finite3
    edge_index = np.asarray(scene.get('edge_index', np.zeros((2, 0), np.int64)), np.int64)
    edge2d = np.asarray(scene.get('edge2d', np.zeros((0, EDGE_2D), np.float32)), np.float32)
    edge3d = np.asarray(scene.get('edge3d', np.zeros((0, EDGE_3D), np.float32)), np.float32)
    if edge_index.size:
        source, target = (edge_index[0], edge_index[1])
        both = valid3d[source] & valid3d[target] & np.isfinite(edge3d).all(axis=1)
    else:
        both = np.zeros(0, bool)
    return {'node2d': node2d, 'node3d': node3d, 'include': include, 'valid3d': valid3d, 'edge_index': edge_index, 'edge2d': edge2d, 'edge3d': edge3d, 'both3d': both}

class _Graph:

    def __init__(self, packed: dict[str, np.ndarray]):
        self.node2d = packed['node2d']
        self.node3d = packed['node3d']
        self.include = packed['include']
        self.valid3d = packed['valid3d']
        self.edge_index = packed['edge_index']
        self.edge2d = packed['edge2d']
        self.edge3d = packed['edge3d']
        self.both3d = packed['both3d']

    @property
    def node(self) -> np.ndarray:
        if len(self.node2d) == 0:
            return np.zeros((0, NODE_2D + NODE_3D), np.float32)
        return np.concatenate((self.node2d, self.node3d), axis=1)

    @property
    def edge(self) -> np.ndarray:
        if len(self.edge2d) == 0 and len(self.edge3d) == 0:
            return np.zeros((0, EDGE_2D + EDGE_3D), np.float32)
        return np.concatenate((self.edge2d, self.edge3d), axis=1)

def _std(second_mean: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.maximum(np.sqrt(np.maximum(second_mean - np.square(mean), 0.0)), 0.001).astype(np.float32)

class SceneStats:

    def __init__(self, kind: str):
        if kind not in {'P', 'E'}:
            raise ValueError(kind)
        self.kind = kind
        self._node2: dict[str, _Moments] = {}
        self._node3: dict[str, _Moments] = {}
        self._edge2: dict[str, _Moments] = {}
        self._edge3: dict[str, _Moments] = {}

    def add(self, patient_id: str, scene: dict) -> None:
        packed = _lite(scene)
        node2 = self._node2.setdefault(patient_id, _Moments(NODE_2D))
        node3 = self._node3.setdefault(patient_id, _Moments(NODE_3D))
        if packed['include'].any():
            node2.add(packed['node2d'][packed['include']])
        if packed['valid3d'].any():
            node3.add(packed['node3d'][packed['valid3d']])
        if self.kind == 'P':
            edge2 = self._edge2.setdefault(patient_id, _Moments(EDGE_2D))
            edge3 = self._edge3.setdefault(patient_id, _Moments(EDGE_3D))
            finite_edge2 = packed['edge2d'][np.isfinite(packed['edge2d']).all(axis=1)] if len(packed['edge2d']) else packed['edge2d']
            edge2.add(finite_edge2)
            if packed['both3d'].any():
                edge3.add(packed['edge3d'][packed['both3d']])

    def _pool(self, groups: dict[str, _Moments], dim: int) -> tuple[np.ndarray, np.ndarray]:
        ready = [item for item in groups.values() if item.ready]
        if not ready:
            return (np.zeros(dim, np.float32), np.ones(dim, np.float32))
        mean = np.mean([item.mean() for item in ready], axis=0)
        second = np.mean([item.mean_square() for item in ready], axis=0)
        return (mean.astype(np.float32), _std(second, mean))

    def finalize(self) -> dict[str, np.ndarray]:
        mean2, std2 = self._pool(self._node2, NODE_2D)
        mean3, std3 = self._pool(self._node3, NODE_3D)
        if not any((item.ready for item in self._node2.values())):
            raise RuntimeError('no included nodes for standardization')
        payload = {'schema': np.asarray(SCHEMA, np.int64), 'mean': np.concatenate((mean2, mean3)).astype(np.float32), 'std': np.concatenate((std2, std3)).astype(np.float32)}
        if self.kind == 'P':
            edge_mean2, edge_std2 = self._pool(self._edge2, EDGE_2D)
            edge_mean3, edge_std3 = self._pool(self._edge3, EDGE_3D)
            payload.update({'edge_mean2d': edge_mean2, 'edge_std2d': edge_std2, 'edge_mean3d': edge_mean3, 'edge_std3d': edge_std3})
        return payload

def apply_scene(scene: dict, stats: dict[str, np.ndarray], arm: str, kind: str) -> dict[str, np.ndarray]:
    packed = _lite(scene)
    graph = _Graph(packed)
    if kind == 'P':
        from celllift.morphology_interaction.data import standardize_graph
        node, edge, include = standardize_graph(graph, stats['mean'], stats['std'], stats['mean'], stats['edge_mean2d'], stats['edge_std2d'], stats['edge_mean3d'], stats['edge_std3d'], arm)
    elif kind == 'E':
        from celllift.geometry_experts.data import standardize_graph
        node, edge, include = standardize_graph(graph, stats['mean'], stats['std'], stats['mean'], arm)
    else:
        raise ValueError(kind)
    return {'node2d': np.asarray(node[:, :NODE_2D], np.float32), 'node3d': np.asarray(node[:, NODE_2D:], np.float32), 'edge2d': np.asarray(edge[:, :EDGE_2D], np.float32), 'edge3d': np.asarray(edge[:, EDGE_2D:], np.float32), 'edge_index': packed['edge_index'], 'include': np.asarray(include, bool), 'valid3d': packed['valid3d'], 'dino': np.asarray(scene.get('dino', np.zeros((len(include), 384), np.float32)), np.float32), 'standardized': True}
