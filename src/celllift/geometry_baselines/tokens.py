from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence
import numpy as np
from .protocol import ARMS, MASK_DIM, THREE_D_DIM, TOKEN_DIM, Arm, validate_arm

@dataclass(frozen=True)
class ModalGraph:
    graph_id: str
    mask2d: np.ndarray
    nucleus_raw3d: np.ndarray
    cell_raw3d: np.ndarray
    nucleus_residual3d: np.ndarray
    cell_residual3d: np.ndarray
    metadata: Mapping[str, object]

    def validate(self) -> None:
        arrays = (self.mask2d, self.nucleus_raw3d, self.cell_raw3d, self.nucleus_residual3d, self.cell_residual3d)
        count = len(self.mask2d)
        if self.mask2d.ndim != 2 or self.mask2d.shape[1] != MASK_DIM:
            raise ValueError(f'{self.graph_id}: MASK2D must be [N,{MASK_DIM}]')
        for value in arrays[1:]:
            if value.shape != (count, THREE_D_DIM):
                raise ValueError(f'{self.graph_id}: 3D block must be [{count},{THREE_D_DIM}]')
        if count == 0 or any((not np.isfinite(value).all() for value in arrays)):
            raise ValueError(f'{self.graph_id}: empty or non-finite modal tensor')
        if not np.array_equal(self.nucleus_raw3d[:, -1], self.cell_raw3d[:, -1]):
            raise ValueError(f'{self.graph_id}: raw NCR3D differs across branches')
        if not np.array_equal(self.nucleus_residual3d[:, -1], self.cell_residual3d[:, -1]):
            raise ValueError(f'{self.graph_id}: residual NCR3D differs across branches')

@dataclass(frozen=True)
class GeometrySample:
    graph_id: str
    nucleus_tokens: np.ndarray
    cell_tokens: np.ndarray
    metadata: Mapping[str, object]

def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256('|'.join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], 'little')

def graph_deciles(graphs: Sequence[ModalGraph], partitions: Mapping[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    roles = sorted(set(partitions.values()))
    for role in roles:
        selected = sorted((graph for graph in graphs if partitions[graph.graph_id] == role), key=lambda graph: (len(graph.mask2d), graph.graph_id))
        if len(selected) < 2:
            raise ValueError(f'partition {role!r} needs at least two graphs for shuffle')
        for rank, graph in enumerate(selected):
            result[graph.graph_id] = min(9, rank * 10 // len(selected))
    return result

def deterministic_graph_derangement(graphs: Sequence[ModalGraph], partitions: Mapping[str, str], *, seed: int, kind: str) -> dict[str, str]:
    deciles = graph_deciles(graphs, partitions)
    result: dict[str, str] = {}
    cells: dict[tuple[str, int], list[str]] = {}
    for graph in graphs:
        cells.setdefault((partitions[graph.graph_id], deciles[graph.graph_id]), []).append(graph.graph_id)
    for cell, ids in sorted(cells.items()):
        ids = sorted(ids)
        if len(ids) < 2:
            ids = sorted((g.graph_id for g in graphs if partitions[g.graph_id] == cell[0]))
        rng = np.random.default_rng(_stable_seed('geometry_baselines', seed, kind, *cell))
        shift = int(rng.integers(1, len(ids)))
        donor = ids[shift:] + ids[:shift]
        result.update(dict(zip(ids, donor)))
    for graph in graphs:
        if graph.graph_id not in result or result[graph.graph_id] == graph.graph_id:
            raise RuntimeError('shuffle donor must exist and differ from target')
        if partitions[result[graph.graph_id]] != partitions[graph.graph_id]:
            raise RuntimeError('shuffle crossed a split')
    return result

class ModalTokenStore:

    def __init__(self, graphs: Sequence[ModalGraph], partitions: Mapping[str, str], *, seed: int) -> None:
        self.graphs = {graph.graph_id: graph for graph in graphs}
        if len(self.graphs) != len(graphs):
            raise ValueError('duplicate graph ID')
        for graph in graphs:
            graph.validate()
        self.partitions = dict(partitions)
        self.mask_donor = deterministic_graph_derangement(graphs, partitions, seed=seed, kind='mask')
        self.raw_donor = deterministic_graph_derangement(graphs, partitions, seed=seed, kind='raw3d')
        self.residual_donor = deterministic_graph_derangement(graphs, partitions, seed=seed, kind='residual3d')

    @staticmethod
    def _resize_rows(values: np.ndarray, count: int) -> np.ndarray:
        if len(values) == count:
            return values
        index = np.floor(np.arange(count, dtype=np.float64) * len(values) / count).astype(np.int64)
        return values[index]

    def sample(self, graph_id: str, arm: Arm | str) -> GeometrySample:
        arm = ARMS[arm] if isinstance(arm, str) else arm
        validate_arm(arm)
        if arm.geometry_id is None:
            raise ValueError('R0 has no geometry sample')
        target = self.graphs[graph_id]
        count = len(target.mask2d)
        if arm.mask_mode == 'none':
            mask = np.zeros((count, MASK_DIM), np.float32)
        elif arm.mask_mode == 'real':
            mask = target.mask2d
        else:
            donor = self.graphs[self.mask_donor[graph_id]]
            mask = donor.mask2d
            count = len(mask)
        zeros = np.zeros((count, THREE_D_DIM), np.float32)
        if arm.three_d_mode == 'none':
            nucleus3d = cell3d = zeros
        else:
            shuffled = arm.three_d_mode.startswith('shuffled_')
            residual = arm.three_d_mode.endswith('residual')
            donor_map = self.residual_donor if residual else self.raw_donor
            source = self.graphs[donor_map[graph_id]] if shuffled else target
            nucleus3d = source.nucleus_residual3d if residual else source.nucleus_raw3d
            cell3d = source.cell_residual3d if residual else source.cell_raw3d
            nucleus3d = self._resize_rows(nucleus3d, count)
            cell3d = self._resize_rows(cell3d, count)
        nucleus = np.concatenate((mask, nucleus3d), axis=1).astype(np.float32, copy=False)
        cell = np.concatenate((mask, cell3d), axis=1).astype(np.float32, copy=False)
        if nucleus.shape != (count, TOKEN_DIM) or cell.shape != nucleus.shape:
            raise AssertionError('geometry_baselines token schema drift')
        if arm.mask_mode == 'none' and (np.any(nucleus[:, :MASK_DIM]) or np.any(cell[:, :MASK_DIM])):
            raise AssertionError('3D-only arm leaked MASK2D')
        if arm.three_d_mode == 'none' and (np.any(nucleus[:, MASK_DIM:]) or np.any(cell[:, MASK_DIM:])):
            raise AssertionError('MASK2D-only arm leaked 3D')
        metadata = dict(target.metadata)
        metadata.update({'arm_id': arm.arm_id, 'mask_donor_graph_id': self.mask_donor.get(graph_id) if arm.mask_mode == 'shuffled' else None, 'three_d_donor_graph_id': (self.residual_donor.get(graph_id) if arm.three_d_mode.endswith('residual') else self.raw_donor.get(graph_id)) if arm.three_d_mode.startswith('shuffled_') else None})
        return GeometrySample(graph_id, nucleus, cell, metadata)

    def samples(self, arm: Arm | str, graph_ids: Sequence[str]) -> list[GeometrySample]:
        return [self.sample(graph_id, arm) for graph_id in graph_ids]

def collate(samples: Sequence[GeometrySample]) -> dict[str, object]:
    if not samples:
        raise ValueError('cannot collate an empty batch')
    maximum = max((len(sample.nucleus_tokens) for sample in samples))
    nucleus = np.zeros((len(samples), maximum, TOKEN_DIM), np.float32)
    cell = np.zeros_like(nucleus)
    nucleus_mask = np.zeros((len(samples), maximum), bool)
    cell_mask = np.zeros_like(nucleus_mask)
    for index, sample in enumerate(samples):
        count = len(sample.nucleus_tokens)
        nucleus[index, :count] = sample.nucleus_tokens
        cell[index, :count] = sample.cell_tokens
        nucleus_mask[index, :count] = True
        cell_mask[index, :count] = True
    output = {'graph_ids': [sample.graph_id for sample in samples], 'nucleus_tokens': nucleus, 'cell_tokens': cell, 'nucleus_mask': nucleus_mask, 'cell_mask': cell_mask}
    preaggregated = [bool(sample.metadata.get('meanpool_preaggregated', False)) for sample in samples]
    if any(preaggregated):
        if not all(preaggregated):
            raise ValueError('cannot mix raw and MeanPool-preaggregated sets in one batch')
        output['nucleus_count'] = np.asarray([int(sample.metadata['nucleus_count']) for sample in samples], dtype=np.float32)
        output['cell_count'] = np.asarray([int(sample.metadata['cell_count']) for sample in samples], dtype=np.float32)
    return output
