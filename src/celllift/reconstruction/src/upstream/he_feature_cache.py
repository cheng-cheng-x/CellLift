from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from functools import partial
import csv
import hashlib
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
import struct
import sys
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .cache_io import ShardedLMDBReader, ShardedLMDBWriter, write_lmdb_manifest
from .schemas import GraphRecord
_HEADER = struct.Struct('<4sIH')
_MAGIC = b'HE19'
FEATURE_DIM = 384

def sha256_file(path: str | Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def encode_he_feature(nucleus_id: np.ndarray, feature: np.ndarray) -> bytes:
    ids = np.ascontiguousarray(nucleus_id, dtype=np.int64)
    values = np.ascontiguousarray(feature, dtype=np.float16)
    if ids.ndim != 1 or values.shape != (len(ids), FEATURE_DIM):
        raise ValueError('frozen_image feature payload requires nucleus_id [N] and feature [N,384]')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate nucleus_id in frozen_image feature payload')
    if not np.isfinite(values).all():
        raise ValueError('non-finite frozen_image H&E feature')
    return _HEADER.pack(_MAGIC, len(ids), FEATURE_DIM) + ids.tobytes() + values.tobytes()

def decode_he_feature(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) < _HEADER.size:
        raise ValueError('truncated frozen_image H&E feature payload')
    magic, count, dim = _HEADER.unpack_from(payload)
    if magic != _MAGIC or dim != FEATURE_DIM:
        raise ValueError('invalid frozen_image H&E feature payload header')
    id_bytes = int(count) * np.dtype(np.int64).itemsize
    feature_bytes = int(count) * int(dim) * np.dtype(np.float16).itemsize
    if len(payload) != _HEADER.size + id_bytes + feature_bytes:
        raise ValueError('invalid frozen_image H&E feature payload length')
    ids = np.frombuffer(payload, np.int64, int(count), _HEADER.size).copy()
    features = np.frombuffer(payload, np.float16, int(count) * int(dim), _HEADER.size + id_bytes).reshape(int(count), int(dim)).copy()
    return (ids, features)

def sample_patch_tokens(token_field: Tensor, nucleus_xy_px: Tensor, node_graph_index: Tensor, *, encoded_size_px: int, source_padding_px: int) -> Tensor:
    if token_field.ndim != 4 or token_field.shape[1] != FEATURE_DIM:
        raise ValueError('token_field must have shape [B,384,Ht,Wt]')
    if nucleus_xy_px.ndim != 2 or nucleus_xy_px.shape[1] != 2:
        raise ValueError('nucleus_xy_px must have shape [N,2]')
    if node_graph_index.shape != (nucleus_xy_px.shape[0],):
        raise ValueError('node_graph_index must have shape [N]')
    if nucleus_xy_px.numel() and (not torch.isfinite(nucleus_xy_px).all() or bool(torch.any(nucleus_xy_px < 0)) or bool(torch.any(nucleus_xy_px > 1023))):
        raise ValueError('nucleus centroid lies outside the 1024-pixel ROI')
    if node_graph_index.numel() and (int(node_graph_index.min()) < 0 or int(node_graph_index.max()) >= token_field.shape[0]):
        raise ValueError('node_graph_index lies outside token-field batch')
    padded_xy = nucleus_xy_px.float() + float(source_padding_px)
    grid = 2.0 * padded_xy / float(encoded_size_px) - 1.0
    selected = token_field.index_select(0, node_graph_index.long())
    sampled = F.grid_sample(selected, grid[:, None, None, :], mode='bilinear', padding_mode='border', align_corners=False)
    return sampled[:, :, 0, 0]

class FrozenTokenEncoder(nn.Module):
    encoder_id: str
    patch_size: int
    encoded_size: int
    source_padding: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def preprocess(self, image_uint8: Tensor) -> Tensor:
        value = image_uint8.float().div_(255.0)
        if self.source_padding:
            value = F.pad(value, (self.source_padding,) * 4, mode='constant', value=1.0)
        mean = value.new_tensor(self.mean)[None, :, None, None]
        std = value.new_tensor(self.std)[None, :, None, None]
        return (value - mean) / std

class DINOv2S14Encoder(FrozenTokenEncoder):

    def __init__(self, source_root: Path, weight_path: Path) -> None:
        super().__init__()
        self.encoder_id = 'dinov2_vits14_lvd142m'
        self.patch_size = 14
        self.encoded_size = 1036
        self.source_padding = 6
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        model = torch.hub.load(str(source_root), 'dinov2_vits14', source='local', pretrained=False)
        state = torch.load(weight_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        tokens = self.model.forward_features(self.preprocess(image))['x_norm_patchtokens']
        if tokens.shape[1:] != (74 * 74, FEATURE_DIM):
            raise RuntimeError(f'unexpected DINOv2 token shape: {tuple(tokens.shape)}')
        return tokens.transpose(1, 2).reshape(image.shape[0], FEATURE_DIM, 74, 74)
