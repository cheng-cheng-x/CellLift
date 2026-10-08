from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from dataclasses import dataclass
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class ShuffleRow:
    graph_id: str
    donor_graph_id: str
    split: str
    count_decile: int
    seed: int

    def as_dict(self) -> dict[str, str | int]:
        return {'graph_id': self.graph_id, 'donor_graph_id': self.donor_graph_id, 'split': self.split, 'count_decile': self.count_decile, 'seed': self.seed}

@dataclass(frozen=True)
class AnchorShuffleRow:
    target_graph_id: str
    target_anchor_id: str
    donor_graph_id: str
    donor_anchor_id: str
    split: str
    count_decile: int
    donor_tier: str
    seed: int

    def as_dict(self) -> dict[str, str | int]:
        return {'target_graph_id': self.target_graph_id, 'target_anchor_id': self.target_anchor_id, 'donor_graph_id': self.donor_graph_id, 'donor_anchor_id': self.donor_anchor_id, 'split': self.split, 'count_decile': self.count_decile, 'donor_tier': self.donor_tier, 'seed': self.seed}

def _hash_order(value: str, seed: int, group: str) -> bytes:
    return hashlib.sha256(f'{seed}\x00{group}\x00{value}'.encode('utf-8')).digest()

def object_count_deciles(graph_ids: Iterable[str], splits: Iterable[str], counts: Iterable[int]) -> np.ndarray:
    graph_ids = np.asarray(list(graph_ids), dtype=str)
    splits = np.asarray(list(splits), dtype=str)
    counts = np.asarray(list(counts), dtype=np.int64)
    if not len(graph_ids) == len(splits) == len(counts):
        raise ValueError('graph_ids, splits and counts must have equal length')
    if len(set(graph_ids.tolist())) != len(graph_ids):
        raise ValueError('graph_ids must be unique')
    if np.any(counts < 0):
        raise ValueError('object counts cannot be negative')
    result = np.empty(len(graph_ids), dtype=np.int8)
    for split in sorted(set(splits.tolist())):
        indices = np.flatnonzero(splits == split)
        edges = np.quantile(counts[indices].astype(np.float64), np.arange(1, 10) / 10.0)
        result[indices] = np.searchsorted(edges, counts[indices], side='left').astype(np.int8)
    return result

def deterministic_graph_derangement(graph_ids: Iterable[str], splits: Iterable[str], counts: Iterable[int], *, seed: int, require_equal_count: bool=False) -> list[ShuffleRow]:
    graph_ids = np.asarray(list(graph_ids), dtype=str)
    splits = np.asarray(list(splits), dtype=str)
    counts = np.asarray(list(counts), dtype=np.int64)
    deciles = object_count_deciles(graph_ids, splits, counts)
    groups: dict[tuple[str, int, int | None], list[int]] = {}
    for index in range(len(graph_ids)):
        key = (str(splits[index]), int(deciles[index]), int(counts[index]) if require_equal_count else None)
        groups.setdefault(key, []).append(index)
    donor = np.empty(len(graph_ids), dtype=np.int64)
    for key, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError(f'cannot derange singleton shuffle group {key}')
        group_name = '|'.join(map(str, key))
        ordered = sorted(indices, key=lambda i: _hash_order(str(graph_ids[i]), int(seed), group_name))
        digest = hashlib.sha256(f'{seed}\x00{group_name}\x00offset'.encode('utf-8')).digest()
        offset = 1 + int.from_bytes(digest[:8], 'little') % (len(ordered) - 1)
        for position, index in enumerate(ordered):
            donor[index] = ordered[(position + offset) % len(ordered)]
    rows = [ShuffleRow(graph_id=str(graph_ids[index]), donor_graph_id=str(graph_ids[donor[index]]), split=str(splits[index]), count_decile=int(deciles[index]), seed=int(seed)) for index in range(len(graph_ids))]
    if any((row.graph_id == row.donor_graph_id for row in rows)):
        raise AssertionError('derangement unexpectedly contains a self donor')
    return rows

def validate_shuffle_rows(rows: Iterable[ShuffleRow]) -> None:
    rows = list(rows)
    if len({row.graph_id for row in rows}) != len(rows):
        raise ValueError('shuffle mapping contains duplicate target graph IDs')
    lookup = {row.graph_id: row for row in rows}
    for row in rows:
        if row.graph_id == row.donor_graph_id:
            raise ValueError('shuffle mapping contains a self donor')
        donor = lookup.get(row.donor_graph_id)
        if donor is None:
            raise ValueError('shuffle donor is absent from the mapping domain')
        if donor.split != row.split or donor.count_decile != row.count_decile or donor.seed != row.seed:
            raise ValueError('shuffle donor violates split/decile/seed constraints')

def deterministic_anchor_derangement(graph_ids: Iterable[str], anchor_ids: Iterable[str | int], splits: Iterable[str], *, seed: int) -> list[AnchorShuffleRow]:
    graph_ids = np.asarray(list(graph_ids), dtype=str)
    anchor_ids = np.asarray([str(value) for value in anchor_ids], dtype=str)
    splits = np.asarray(list(splits), dtype=str)
    if not len(graph_ids) == len(anchor_ids) == len(splits):
        raise ValueError('graph_ids, anchor_ids and splits must have equal length')
    keys = list(zip(graph_ids.tolist(), anchor_ids.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError('(graph_id, anchor_id) keys must be unique')
    graph_to_rows: dict[str, list[int]] = {}
    graph_to_split: dict[str, str] = {}
    for index, graph_id in enumerate(graph_ids):
        graph_to_rows.setdefault(str(graph_id), []).append(index)
        previous = graph_to_split.setdefault(str(graph_id), str(splits[index]))
        if previous != str(splits[index]):
            raise ValueError(f'graph {graph_id!r} spans multiple splits')
    unique_graphs = np.asarray(sorted(graph_to_rows), dtype=str)
    graph_splits = np.asarray([graph_to_split[graph] for graph in unique_graphs], dtype=str)
    graph_counts = np.asarray([len(graph_to_rows[graph]) for graph in unique_graphs], dtype=np.int64)
    graph_deciles = object_count_deciles(unique_graphs, graph_splits, graph_counts)
    graph_index = {graph: index for index, graph in enumerate(unique_graphs)}
    result: list[AnchorShuffleRow] = []
    for target_graph in unique_graphs:
        target_index = graph_index[str(target_graph)]
        split = str(graph_splits[target_index])
        count = int(graph_counts[target_index])
        decile = int(graph_deciles[target_index])
        exact = [str(graph) for graph, candidate_split, candidate_count, candidate_decile in zip(unique_graphs, graph_splits, graph_counts, graph_deciles) if graph != target_graph and candidate_split == split and (int(candidate_decile) == decile) and (int(candidate_count) == count)]
        if exact:
            candidates, tier = (exact, 'exact_count')
        else:
            candidates, tier = ([str(graph) for graph, candidate_split, candidate_decile in zip(unique_graphs, graph_splits, graph_deciles) if graph != target_graph and candidate_split == split and (int(candidate_decile) == decile)], 'count_decile')
        if not candidates:
            raise ValueError(f'no cross-graph donor for graph={target_graph!r}, split={split!r}, count_decile={decile}')
        graph_digest = hashlib.sha256(f'{seed}\x00{split}\x00{target_graph}\x00donor-graph'.encode('utf-8')).digest()
        donor_graph = sorted(candidates)[int.from_bytes(graph_digest[:8], 'little') % len(candidates)]
        target_rows = sorted(graph_to_rows[str(target_graph)], key=lambda i: anchor_ids[i])
        donor_rows = sorted(graph_to_rows[donor_graph], key=lambda i: anchor_ids[i])
        offset_digest = hashlib.sha256(f'{seed}\x00{target_graph}\x00{donor_graph}\x00anchor-offset'.encode('utf-8')).digest()
        offset = int.from_bytes(offset_digest[:8], 'little') % len(donor_rows)
        for position, target_row in enumerate(target_rows):
            donor_row = donor_rows[(position + offset) % len(donor_rows)]
            result.append(AnchorShuffleRow(target_graph_id=str(target_graph), target_anchor_id=str(anchor_ids[target_row]), donor_graph_id=donor_graph, donor_anchor_id=str(anchor_ids[donor_row]), split=split, count_decile=decile, donor_tier=tier, seed=int(seed)))
    return sorted(result, key=lambda row: (row.target_graph_id, row.target_anchor_id))

def validate_anchor_shuffle_rows(rows: Iterable[AnchorShuffleRow]) -> None:
    rows = list(rows)
    target_keys = [(row.target_graph_id, row.target_anchor_id) for row in rows]
    if len(set(target_keys)) != len(rows):
        raise ValueError('anchor shuffle mapping contains duplicate target keys')
    domain = set(target_keys)
    for row in rows:
        if row.target_graph_id == row.donor_graph_id:
            raise ValueError('anchor shuffle mapping contains a same-graph donor')
        if (row.donor_graph_id, row.donor_anchor_id) not in domain:
            raise ValueError('anchor shuffle donor key is absent from the mapping domain')
        if row.donor_tier not in {'exact_count', 'count_decile'}:
            raise ValueError('invalid donor tier')

def gather_donor_values(rows: Iterable[AnchorShuffleRow], values_by_anchor: dict[tuple[str, str], np.ndarray | float]) -> tuple[list[tuple[str, str]], np.ndarray]:
    rows = sorted(rows, key=lambda row: (row.target_graph_id, row.target_anchor_id))
    validate_anchor_shuffle_rows(rows)
    target_keys: list[tuple[str, str]] = []
    values: list[np.ndarray] = []
    for row in rows:
        donor_key = (row.donor_graph_id, row.donor_anchor_id)
        if donor_key not in values_by_anchor:
            raise KeyError(f'missing donor value for {donor_key!r}')
        target_keys.append((row.target_graph_id, row.target_anchor_id))
        values.append(np.asarray(values_by_anchor[donor_key]))
    try:
        gathered = np.stack(values)
    except ValueError as exc:
        raise ValueError('donor values must have a common row shape') from exc
    return (target_keys, gathered)
