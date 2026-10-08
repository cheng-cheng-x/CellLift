from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from collections import Counter
from typing import Iterable, Iterator, Mapping
import hashlib
import numpy as np
from .protocol import SHUFFLE_SEED

@dataclass(frozen=True)
class GraphShuffleRecord:
    graph_id: str
    group_id: str
    role: str
    count: int

def count_deciles(records: Iterable[GraphShuffleRecord]) -> dict[str, int]:
    rows = list(records)
    output: dict[str, int] = {}
    for role in sorted({row.role for row in rows}):
        selected = sorted(((row.count, row.graph_id) for row in rows if row.role == role))
        n = len(selected)
        for rank, (_, graph_id) in enumerate(selected):
            output[graph_id] = min(9, 10 * rank // max(1, n))
    return output

def _rotation_derangement(ids: list[str], groups: Mapping[str, str], seed: int) -> dict[str, str]:
    if len(ids) < 2:
        raise RuntimeError('shuffle stratum has fewer than two graphs')
    digest = hashlib.sha256(('|'.join(sorted(ids)) + f'|{seed}').encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    order = np.asarray(sorted(ids), object)
    rng.shuffle(order)
    for shift in range(1, len(order)):
        donor = np.roll(order, shift)
        if all((groups[str(a)] != groups[str(b)] for a, b in zip(order, donor))):
            return {str(a): str(b) for a, b in zip(order, donor)}
    targets = list(map(str, order))
    donors_by_group: dict[str, list[str]] = {}
    for value in targets:
        donors_by_group.setdefault(groups[value], []).append(value)

    def candidates(target: str) -> Iterator[str]:
        own = groups[target]
        for group in sorted(donors_by_group):
            if group != own:
                yield from donors_by_group[group]
    target_donor: dict[str, str] = {}
    donor_target: dict[str, str] = {}
    for root in targets:
        if target_donor.get(root):
            continue
        stack = [(root, candidates(root))]
        visited: set[str] = set()
        path: list[tuple[str, str]] = []
        found = False
        while stack and (not found):
            target, options = stack[-1]
            advanced = False
            for donor in options:
                if donor in visited:
                    continue
                visited.add(donor)
                owner = donor_target.get(donor)
                path.append((target, donor))
                if owner is None:
                    found = True
                    advanced = True
                    break
                stack.append((owner, candidates(owner)))
                advanced = True
                break
            if not advanced:
                stack.pop()
                if stack and path:
                    path.pop()
        if not found:
            raise RuntimeError('cannot build group-disjoint graph derangement')
        for target, donor in reversed(path):
            target_donor[target] = donor
            donor_target[donor] = target
    result = target_donor
    return result

def _coalesce_infeasible_deciles(rows: list[GraphShuffleRecord], decile: Mapping[str, int]) -> list[tuple[tuple[int, ...], list[GraphShuffleRecord]]]:
    strata: list[tuple[tuple[int, ...], list[GraphShuffleRecord]]] = []
    for value in range(10):
        selected = [row for row in rows if decile[row.graph_id] == value]
        if selected:
            strata.append(((value,), selected))
    if len(strata) == 1:
        return strata
    while True:
        index = next((i for i, (_, members) in enumerate(strata) if 2 * max(Counter((row.group_id for row in members)).values()) > len(members)), None)
        if index is None:
            return strata
        bins, _ = strata[index]
        distances = [(min((abs(a - b) for a in bins for b in other_bins)), other_bins, i) for i, (other_bins, members) in enumerate(strata) if i != index]
        if not distances:
            return strata
        _, other_bins, neighbor = min(distances)
        merged_bins = tuple(sorted(bins + other_bins))
        merged = strata[index][1] + strata[neighbor][1]
        for remove in sorted((index, neighbor), reverse=True):
            strata.pop(remove)
        strata.append((merged_bins, merged))
        strata.sort(key=lambda item: item[0][0])

def build_donor_map(records: Iterable[GraphShuffleRecord], seed: int=SHUFFLE_SEED) -> dict[str, str]:
    rows = list(records)
    decile = count_deciles(rows)
    groups = {row.graph_id: row.group_id for row in rows}
    output: dict[str, str] = {}
    for role in sorted({row.role for row in rows}):
        selected = [row for row in rows if row.role == role]
        strata = _coalesce_infeasible_deciles(selected, decile)
        if len(strata) == 1 and 2 * max(Counter((row.group_id for row in strata[0][1])).values()) > len(strata[0][1]):
            output.update(_fallback_train_donor_map(selected, rows, seed))
            continue
        for _, members in strata:
            ids = [row.graph_id for row in members]
            if ids:
                bins = tuple((decile[graph_id] for graph_id in ids))
                bin_digest = hashlib.sha256('|'.join(map(str, bins)).encode()).digest()
                layer_seed = seed + int.from_bytes(bin_digest[:8], 'little') % 1000003
                output.update(_rotation_derangement(ids, groups, layer_seed))
    if set(output) != {row.graph_id for row in rows}:
        raise AssertionError('donor map coverage mismatch')
    return output

def _fallback_train_donor_map(targets: list[GraphShuffleRecord], rows: list[GraphShuffleRecord], seed: int) -> dict[str, str]:
    target_rows = sorted(targets, key=lambda row: (row.count, row.graph_id))
    target_groups = {row.group_id for row in target_rows}
    donor_rows = sorted((row for row in rows if row.role == 'train' and row.group_id not in target_groups), key=lambda row: (row.count, row.graph_id))
    if len(donor_rows) < len(target_rows):
        raise RuntimeError('shuffle role has insufficient disjoint development training donors')
    selected = [donor_rows[int(index)] for index in quantile_indices(len(donor_rows), len(target_rows))]
    digest = hashlib.sha256(('|'.join(sorted((row.graph_id for row in target_rows))) + f'|{seed}').encode()).digest()
    rotation = int.from_bytes(digest[:8], 'little') % len(target_rows)
    return {target.graph_id: donor.graph_id for target, donor in zip(target_rows, selected[rotation:] + selected[:rotation])}

def quantile_indices(donor_count: int, target_count: int) -> np.ndarray:
    if donor_count <= 0 or target_count <= 0:
        raise ValueError('anchor counts must be positive')
    q = (np.arange(target_count, dtype=np.float64) + 0.5) / target_count
    return np.minimum(donor_count - 1, np.floor(q * donor_count).astype(np.int64))

def replace_tail(target_tokens: np.ndarray, donor_tail: np.ndarray) -> np.ndarray:
    target = np.asarray(target_tokens, np.float32)
    donor = np.asarray(donor_tail, np.float32)
    if target.ndim != 2 or target.shape[1] != 41 or donor.ndim != 2 or (donor.shape[1] != 5):
        raise ValueError('expected target [N,41] and donor tail [M,5]')
    output = target.copy()
    output[:, 36:] = donor[quantile_indices(len(donor), len(target))]
    return output
