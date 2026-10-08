from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import io
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any
import numpy as np

class ObjectType(IntEnum):
    NUCLEUS = 0
    CELL = 1

class ObservationState(IntEnum):
    UNKNOWN = 0
    POSITIVE = 1
    CONFIRMED_EMPTY = 2
PLANE_TO_INDEX = {'lower': 0, 'middle': 1, 'upper': 2}
TRAINING_REQUIRED_COLUMNS = {'split', 'component_id', 'track_id', 'biocell_track_id', 'anchor_layer_idx', 'anchor_section_id', 'anchor_nucleus_id', 'anchor_cell_id', 'lower_section_id', 'lower_nucleus_state', 'lower_nucleus_id', 'lower_cell_state', 'lower_cell_id', 'lower_link_class', 'upper_section_id', 'upper_nucleus_state', 'upper_nucleus_id', 'upper_cell_state', 'upper_cell_id', 'upper_link_class'}
CONFIDENCE_REQUIRED_COLUMNS = {'anchor_layer_idx', 'anchor_nucleus_id', 'lower_confidence_code', 'lower_cycle_code', 'lower_link_class_code', 'lower_neighbor_entity_id', 'upper_confidence_code', 'upper_cycle_code', 'upper_link_class_code', 'upper_neighbor_entity_id'}
LAYER_REQUIRED_COLUMNS = {'layer_idx', 'roi_layer_id', 'component_id', 'track_id', 'section_id', 'split', 'nucleus_mask_path', 'nucleus_mask_sha256', 'nucleus_instances_path', 'nucleus_instances_sha256', 'cell_mask_path', 'cell_mask_sha256', 'cell_instances_path', 'cell_instances_sha256', 'pairs_path', 'pairs_sha256', 'nucleus_count', 'cell_count'}

def _npz_bytes(**arrays: Any) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()

def _read_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

@dataclass
class GraphRecord:
    graph_id: str
    layer_idx: int
    split: str
    component_id: str
    track_id: str
    section_id: int
    nucleus_id: np.ndarray
    xy_px: np.ndarray
    xy_um: np.ndarray
    rho_um: np.ndarray
    border_flag: np.ndarray
    anchor_cell_id: np.ndarray
    biological_entity_id: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_feat: np.ndarray
    undirected_edge_src: np.ndarray
    undirected_edge_dst: np.ndarray

    def to_bytes(self) -> bytes:
        return _npz_bytes(metadata=np.asarray([self.graph_id, self.split, self.component_id, self.track_id], dtype='U128'), scalar=np.asarray([self.layer_idx, self.section_id], dtype=np.int64), nucleus_id=self.nucleus_id, xy_px=self.xy_px, xy_um=self.xy_um, rho_um=self.rho_um, border_flag=self.border_flag, anchor_cell_id=self.anchor_cell_id, biological_entity_id=self.biological_entity_id, edge_src=self.edge_src, edge_dst=self.edge_dst, edge_feat=self.edge_feat, undirected_edge_src=self.undirected_edge_src, undirected_edge_dst=self.undirected_edge_dst)

    @classmethod
    def from_bytes(cls, payload: bytes) -> 'GraphRecord':
        d = _read_npz(payload)
        meta = d.pop('metadata').tolist()
        scalar = d.pop('scalar')
        return cls(meta[0], int(scalar[0]), meta[1], meta[2], meta[3], int(scalar[1]), **d)

@dataclass
class ObservationRecord:
    anchor_node_idx: np.ndarray
    object_type: np.ndarray
    plane_idx: np.ndarray
    weight: np.ndarray
    confidence_code: np.ndarray
    cycle_code: np.ndarray
    triplet_group_id: np.ndarray
    target_uid: np.ndarray | None = None
    evidence_code: np.ndarray | None = None

    def to_bytes(self) -> bytes:
        values = {key: value for key, value in self.__dict__.items() if value is not None}
        return _npz_bytes(**values)

    @classmethod
    def from_bytes(cls, payload: bytes) -> 'ObservationRecord':
        return cls(**_read_npz(payload))

@dataclass
class TargetRecord:
    target_uid: str
    layer_idx: int
    object_type: int
    instance_id: int
    bbox_xyxy: np.ndarray
    packed_mask: np.ndarray
    crop_shape: np.ndarray
    area_px: int
    centroid_xy: np.ndarray
    border_flag: bool
    support_xyxy: np.ndarray
    _BINARY_MAGIC = b'TRG1'
    _BINARY_HEADER = struct.Struct('<4sHiBBqi4i2i2f4iI')

    def decode_mask(self) -> np.ndarray:
        count = int(np.prod(self.crop_shape))
        return np.unpackbits(self.packed_mask, count=count).reshape(tuple(self.crop_shape)).astype(bool)

    @classmethod
    def centroid_from_bytes(cls, payload: bytes) -> np.ndarray:
        if payload.startswith(cls._BINARY_MAGIC):
            values = cls._BINARY_HEADER.unpack_from(payload)
            tail = values[7:]
            return np.asarray(tail[6:8], dtype=np.float32)
        return cls.from_bytes(payload).centroid_xy.astype(np.float32, copy=False)

    @classmethod
    def pose_header_from_bytes(cls, payload: bytes) -> tuple[np.ndarray, int]:
        if payload.startswith(cls._BINARY_MAGIC):
            values = cls._BINARY_HEADER.unpack_from(payload)
            tail = values[7:]
            return (np.asarray(tail[6:8], dtype=np.float32), int(values[6]))
        record = cls.from_bytes(payload)
        return (record.centroid_xy.astype(np.float32, copy=False), int(record.area_px))

    def to_bytes(self) -> bytes:
        uid = self.target_uid.encode('utf-8')
        if len(uid) > 65535:
            raise ValueError('target_uid is too long for the binary target record')
        bbox = np.asarray(self.bbox_xyxy, dtype=np.int32).reshape(4)
        shape = np.asarray(self.crop_shape, dtype=np.int32).reshape(2)
        centroid = np.asarray(self.centroid_xy, dtype=np.float32).reshape(2)
        support = np.asarray(self.support_xyxy, dtype=np.int32).reshape(4)
        packed = np.ascontiguousarray(self.packed_mask, dtype=np.uint8).tobytes()
        header = self._BINARY_HEADER.pack(self._BINARY_MAGIC, len(uid), int(self.layer_idx), int(self.object_type), int(self.border_flag), int(self.instance_id), int(self.area_px), *bbox.tolist(), *shape.tolist(), *centroid.tolist(), *support.tolist(), len(packed))
        return header + uid + packed

    @classmethod
    def from_bytes(cls, payload: bytes) -> 'TargetRecord':
        if payload.startswith(cls._BINARY_MAGIC):
            values = cls._BINARY_HEADER.unpack_from(payload)
            _, uid_len, layer_idx, object_type, border_flag, instance_id, area_px, *tail = values
            bbox = np.asarray(tail[0:4], dtype=np.int32)
            crop_shape = np.asarray(tail[4:6], dtype=np.int32)
            centroid = np.asarray(tail[6:8], dtype=np.float32)
            support = np.asarray(tail[8:12], dtype=np.int32)
            packed_len = int(tail[12])
            offset = cls._BINARY_HEADER.size
            uid = payload[offset:offset + uid_len].decode('utf-8')
            offset += uid_len
            packed = np.frombuffer(payload, dtype=np.uint8, count=packed_len, offset=offset).copy()
            if offset + packed_len != len(payload):
                raise ValueError('invalid binary target record length')
            return cls(uid, int(layer_idx), int(object_type), int(instance_id), bbox, packed, crop_shape, int(area_px), centroid, bool(border_flag), support)
        d = _read_npz(payload)
        uid = str(d.pop('metadata')[0])
        s = d.pop('scalar')
        return cls(uid, int(s[0]), int(s[1]), int(s[2]), d['bbox_xyxy'], d['packed_mask'], d['crop_shape'], int(s[3]), d['centroid_xy'], bool(s[4]), d['support_xyxy'])
