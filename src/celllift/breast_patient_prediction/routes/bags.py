from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import OrderedDict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from ..model_input import layout
from .config import DINO_DIM, TASK_SPEC, MAX_TILES
from .scale import assert_nucleus_identity

def _read_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)

def _finite_dino(value) -> np.ndarray:
    array = np.asarray(value, np.float32).reshape(-1)
    if array.shape != (DINO_DIM,) or not np.isfinite(array).all():
        raise ValueError(f'DINO must be finite [{DINO_DIM}], got {array.shape}')
    return array

def _label_int(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text in {'', 'nan', 'None'}:
        return None
    return int(value)

class SlideArrays:

    def __init__(self, path: Path, array_keys: tuple[str, ...], extra: dict[str, np.ndarray] | None=None):
        with np.load(path, allow_pickle=False) as data:
            ids = [str(item) for item in data['graph_ids'].tolist()]
            self.index = {graph_id: i for i, graph_id in enumerate(ids)}
            self.empty = np.asarray(data['empty'], bool)
            self.arrays = {key: data[key] for key in array_keys}
            self.ptr = {key: data[f'{key}_ptr'] for key in array_keys}
            if extra:
                for key, value in extra.items():
                    self.arrays[key] = np.asarray(data[value[0]]) if isinstance(value, tuple) else np.asarray(data[key])
                    if f'{key}_ptr' in data:
                        self.ptr[key] = data[f'{key}_ptr']
            if 'edge_index' in data.files:
                self.arrays['edge_index'] = data['edge_index']
                self.ptr['edge_index'] = data['edge_index_ptr']

    def _slice(self, key: str, index: int) -> np.ndarray:
        ptr = self.ptr[key]
        begin, end = (int(ptr[index]), int(ptr[index + 1]))
        if key == 'edge_index':
            return self.arrays[key][:, begin:end]
        return self.arrays[key][begin:end]

    def tile(self, graph_id: str) -> dict[str, Any] | None:
        index = self.index.get(str(graph_id))
        if index is None:
            return None
        row = {'empty': bool(self.empty[index])}
        for key in self.ptr:
            row[key] = self._slice(key, index)
        return row

class SlideCache:

    def __init__(self, base: Path, kind: str, max_slides: int=2048):
        self.base = Path(base)
        self.kind = kind
        self.max_slides = int(max_slides)
        self._store: OrderedDict[str, SlideArrays] = OrderedDict()

    def _load(self, slide_id: str) -> SlideArrays:
        if self.kind == 'geometry':
            path = layout.geometry_path(self.base, slide_id)
            keys = ('nucleus_id', 'rays', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy')
        elif self.kind == 'scene':
            path = layout.scene_graph_path(self.base, slide_id)
            keys = ('nucleus_id', 'dino', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy', 'nucleus_transform', 'cell_transform', 'edge2d', 'edge3d')
        else:
            raise ValueError(self.kind)
        return SlideArrays(path, keys)

    def get(self, slide_id: str, graph_id: str) -> dict[str, Any]:
        slide_id = str(slide_id)
        if slide_id not in self._store:
            if len(self._store) >= self.max_slides:
                self._store.popitem(last=False)
            self._store[slide_id] = self._load(slide_id)
        else:
            self._store.move_to_end(slide_id)
        row = self._store[slide_id].tile(graph_id)
        if row is None:
            raise KeyError(f'{self.kind} missing {graph_id} on {slide_id}')
        return row

class ResidualCache:

    def __init__(self, probe_root: Path, max_slides: int=2048):
        self.root = Path(probe_root) / 'residuals'
        self.max_slides = int(max_slides)
        self._store: OrderedDict[str, SlideArrays] = OrderedDict()

    def get(self, slide_id: str, graph_id: str, nucleus_id: np.ndarray) -> np.ndarray:
        slide_id = str(slide_id)
        path = self.root / f'{layout.shard_name(slide_id)}__{layout._slide_stem(slide_id)}.npz'
        if slide_id not in self._store:
            if not path.is_file():
                raise FileNotFoundError(f'residual file missing for {slide_id}: {path}')
            if len(self._store) >= self.max_slides:
                self._store.popitem(last=False)
            self._store[slide_id] = SlideArrays(path, ('nucleus_id', 'residual9', 'valid_target'))
        else:
            self._store.move_to_end(slide_id)
        row = self._store[slide_id].tile(graph_id)
        if row is None:
            raise KeyError(f'residual missing {graph_id}')
        assert_nucleus_identity(row['nucleus_id'], nucleus_id, graph_id)
        return np.asarray(row['residual9'], np.float32)

def load_indices(data_root: Path | None=None):
    base = layout.root(data_root)
    tiles = _read_parquet(base / 'indices' / 'tiles.parquet')
    patients = _read_parquet(base / 'indices' / 'patients.parquet')
    return (base, tiles, patients)

def load_global_dino(base: Path) -> dict[str, np.ndarray]:
    path = base / 'features' / 'tile_dino_global.parquet'
    if not path.is_file():
        return {}
    frame = _read_parquet(path)
    return {str(row['tile_id']): _finite_dino(row['dino']) for row in frame.to_dict('records')}

def task_rows(patients, task: str) -> list[dict[str, Any]]:
    spec = TASK_SPEC[task]
    rows = []
    for raw in patients.to_dict('records'):
        if not bool(raw.get('evaluable', False)):
            continue
        label = _label_int(raw.get(spec['label_id']))
        name = str(raw.get(spec['label']) or '').strip()
        if label is None or not name:
            continue
        split = str(raw.get('split') or '').lower()
        if split not in {'fit', 'val', 'test'}:
            continue
        rows.append({'patient_id': str(raw['patient_id']), 'split': split, 'label': int(label), 'n_tiles': int(raw.get('n_success') or raw.get('n_tiles') or 0)})
    return rows

def build_bags(task: str, *, data_root: Path | None=None, need_image: bool=False, max_tiles: int=MAX_TILES) -> tuple[Path, list[dict[str, Any]], dict[str, np.ndarray]]:
    base, tiles, patients = load_indices(data_root)
    global_dino = load_global_dino(base) if need_image else {}
    if need_image and len(global_dino) != int((tiles['status'] == 'success').sum()):
        raise RuntimeError(f'tile_dino_global coverage {len(global_dino)} != success tiles')
    keep = {row['patient_id'] for row in task_rows(patients, task)}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in tiles.to_dict('records'):
        pid = str(raw['patient_id'])
        if pid not in keep or str(raw.get('status')) != 'success':
            continue
        grouped.setdefault(pid, []).append(dict(raw))
    spec = TASK_SPEC[task]
    bags = []
    for meta in task_rows(patients, task):
        members = grouped.get(meta['patient_id'], [])
        members.sort(key=lambda item: str(item['tile_id']))
        if len(members) > max_tiles:
            members = members[:int(max_tiles)]
        if not members:
            continue
        dino = None
        if need_image:
            dino = np.stack([global_dino[str(item['tile_id'])] for item in members], 0)
        bags.append({**meta, 'tiles': [{'tile_id': str(item['tile_id']), 'graph_id': str(item['graph_id']), 'slide_id': str(item['slide_id'])} for item in members], 'dino_global': dino, 'classes': spec['classes']})
    if not bags:
        raise RuntimeError(f'no evaluable bags for {task}')
    return (base, bags, global_dino)

def split_bags(bags: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {'fit': [], 'val': [], 'test': []}
    for bag in bags:
        out[bag['split']].append(bag)
    return out

def empty_geometry() -> dict[str, np.ndarray]:
    return {'nucleus_id': np.empty(0, np.int64), 'rays': np.empty((0, 36), np.float32), 'node2d': np.empty((0, 38), np.float32), 'node3d': np.empty((0, 12), np.float32), 'include': np.empty(0, bool), 'valid3d': np.empty(0, bool), 'center_xy': np.empty((0, 2), np.float32), 'empty': True}

def empty_scene() -> dict[str, np.ndarray]:
    row = empty_geometry()
    row.update({'dino': np.empty((0, DINO_DIM), np.float32), 'edge_index': np.empty((2, 0), np.int32), 'edge2d': np.empty((0, 4), np.float32), 'edge3d': np.empty((0, 8), np.float32)})
    return row

def load_tile_geometry(cache: SlideCache, tile: dict[str, str]) -> dict[str, Any]:
    try:
        row = cache.get(tile['slide_id'], tile['graph_id'])
    except FileNotFoundError:
        return empty_geometry()
    if row.get('empty'):
        payload = empty_geometry()
        payload['empty'] = True
        return payload
    return row

def load_tile_scene(cache: SlideCache, tile: dict[str, str]) -> dict[str, Any]:
    try:
        row = cache.get(tile['slide_id'], tile['graph_id'])
    except FileNotFoundError:
        return empty_scene()
    if row.get('empty'):
        return empty_scene()
    return row

def gpu_tile_chunk(device: str='cuda') -> int:
    try:
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            return 2
        name = torch.cuda.get_device_name(0)
        if 'A800' in name or 'A100' in name:
            return 16
        if 'auxiliary_ad5736' in name:
            return 8
        return 4
    except Exception:
        return 4
