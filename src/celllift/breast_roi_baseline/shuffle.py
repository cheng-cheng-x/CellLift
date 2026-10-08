from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from dataclasses import dataclass, replace
from typing import Iterable, Sequence
import numpy as np

@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    roi_id: str
    wsi_id: str
    split: str
    anchor_count: int
    count_decile: int | None = None

    def __post_init__(self) -> None:
        if not self.tile_id or not self.roi_id or (not self.wsi_id) or (not self.split):
            raise ValueError('tile_id, roi_id, wsi_id and split must be non-empty')
        if self.anchor_count <= 0:
            raise ValueError('anchor_count must be positive')
        if self.count_decile is not None and (not 0 <= self.count_decile <= 9):
            raise ValueError('count_decile must be in [0, 9]')

@dataclass(frozen=True)
class DonorRow:
    target_tile_id: str
    target_anchor_index: int
    donor_tile_id: str
    donor_anchor_index: int
    split: str
    count_decile: int
    donor_roi_id: str
    donor_wsi_id: str

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()

def assign_count_deciles(records: Sequence[TileRecord], *, bins: int=10) -> tuple[TileRecord, ...]:
    if not 1 <= bins <= 10:
        raise ValueError('bins must be in [1, 10]')
    if len({record.tile_id for record in records}) != len(records):
        raise ValueError('tile_id values must be unique')
    output: dict[str, TileRecord] = {}
    for split in sorted({record.split for record in records}):
        members = sorted((record for record in records if record.split == split), key=lambda record: (record.anchor_count, record.tile_id))
        for rank, record in enumerate(members):
            decile = min(bins - 1, rank * bins // len(members))
            output[record.tile_id] = replace(record, count_decile=decile)
    return tuple((output[record.tile_id] for record in records))

def _rng(protocol_id: str, fold: int, seed: int, role: str) -> np.random.Generator:
    payload = f'{protocol_id}|fold={fold}|seed={seed}|role={role}'.encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))

def build_donor_mapping(records: Sequence[TileRecord], *, seed: int, fold: int, role: str, protocol_id: str='breast_roi_baseline', bins: int=10) -> tuple[DonorRow, ...]:
    if not records:
        raise ValueError('cannot build a donor mapping from no tiles')
    materialized = tuple(records)
    if any((record.count_decile is None for record in materialized)):
        if any((record.count_decile is not None for record in materialized)):
            raise ValueError('count_decile must be supplied for every tile or no tile')
        materialized = assign_count_deciles(materialized, bins=bins)
    if len({record.tile_id for record in materialized}) != len(materialized):
        raise ValueError('tile_id values must be unique')
    grouped: dict[tuple[str, int], list[TileRecord]] = {}
    for record in materialized:
        grouped.setdefault((record.split, int(record.count_decile)), []).append(record)
    generator = _rng(protocol_id, fold, seed, role)
    rows: list[DonorRow] = []
    for target in sorted(materialized, key=lambda record: record.tile_id):
        candidates = [donor for donor in grouped[target.split, int(target.count_decile)] if donor.tile_id != target.tile_id and donor.roi_id != target.roi_id and (donor.wsi_id != target.wsi_id)]
        if not candidates:
            raise ValueError(f'no eligible donor in split/count-decile for tile={target.tile_id!r}, split={target.split!r}, decile={target.count_decile}')
        candidates.sort(key=lambda record: record.tile_id)
        order = generator.permutation(len(candidates))
        anchor_phase = int(generator.integers(0, 2 ** 31 - 1))
        for target_anchor in range(target.anchor_count):
            donor = candidates[int(order[(target_anchor + anchor_phase) % len(order)])]
            donor_anchor = int(generator.integers(0, donor.anchor_count))
            rows.append(DonorRow(target_tile_id=target.tile_id, target_anchor_index=target_anchor, donor_tile_id=donor.tile_id, donor_anchor_index=donor_anchor, split=target.split, count_decile=int(target.count_decile), donor_roi_id=donor.roi_id, donor_wsi_id=donor.wsi_id))
    validate_donor_mapping(materialized, rows)
    return tuple(rows)

def validate_donor_mapping(records: Sequence[TileRecord], rows: Iterable[DonorRow]) -> None:
    by_tile = {record.tile_id: record for record in records}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        target = by_tile.get(row.target_tile_id)
        donor = by_tile.get(row.donor_tile_id)
        if target is None or donor is None:
            raise ValueError('donor mapping references an unknown tile')
        key = (row.target_tile_id, row.target_anchor_index)
        if key in seen:
            raise ValueError(f'duplicate target anchor mapping: {key}')
        seen.add(key)
        if not 0 <= row.target_anchor_index < target.anchor_count:
            raise ValueError('target anchor index is out of range')
        if not 0 <= row.donor_anchor_index < donor.anchor_count:
            raise ValueError('donor anchor index is out of range')
        if target.split != donor.split or row.split != target.split:
            raise ValueError('donor mapping crosses split')
        if target.count_decile != donor.count_decile or row.count_decile != target.count_decile:
            raise ValueError('donor mapping crosses count decile')
        if target.tile_id == donor.tile_id:
            raise ValueError('donor mapping uses the same tile')
        if target.roi_id == donor.roi_id:
            raise ValueError('donor mapping uses the same ROI')
        if target.wsi_id == donor.wsi_id:
            raise ValueError('donor mapping uses the same WSI')
        if row.donor_roi_id != donor.roi_id or row.donor_wsi_id != donor.wsi_id:
            raise ValueError('persisted donor provenance disagrees with tile manifest')
    expected = {(record.tile_id, anchor_index) for record in records for anchor_index in range(record.anchor_count)}
    if seen != expected:
        raise ValueError('donor mapping does not cover every target anchor exactly once')

def mapping_checksum(rows: Iterable[DonorRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f'{row.target_tile_id}\t{row.target_anchor_index}\t{row.donor_tile_id}\t{row.donor_anchor_index}\t{row.split}\t{row.count_decile}\t{row.donor_roi_id}\t{row.donor_wsi_id}\n'.encode('utf-8'))
    return digest.hexdigest()
