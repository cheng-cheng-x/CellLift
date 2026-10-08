from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import sys
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.runtime import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from .constants import ARVANITI_DATA, LIZARD_DATA
_MODEL_INPUT = Path(__file__).resolve().parents[1] / 'model_inputs'
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
_PUBLIC = Path(__file__).resolve().parents[1]
if str(_PUBLIC) not in sys.path:
    sys.path.insert(0, str(_PUBLIC))
from celllift.model_inputs.arvaniti_lizard.constants import ProjectionScene_SOURCE
from celllift.model_inputs.arvaniti_lizard.graphs import GraphReader
from celllift.model_inputs.arvaniti_lizard.io_utils import read_parquet
from celllift.model_inputs.arvaniti_lizard.rgb import pad_mask
from celllift.model_inputs.common.graph import GraphRecord
from celllift.matched_geometry_controls.upstream import load_projection_scene
load_projection_scene(ProjectionScene_SOURCE)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
GEOM_DIM = {'g2': 38, 'g3': 9, 'gr': 9, 'g23': 47, 'g2r': 47, 'n2': 38, 'n3': 9, 'nr': 9, 'n23': 47, 'n2r': 47}

def _role_ok(row: dict[str, Any], split: str) -> bool:
    role = str(row.get('role') or '').upper()
    want = {'fit': 'FIT', 'train': 'FIT', 'val': 'VAL', 'test': 'TEST'}[split.lower()]
    return role == want

def _x2_from_rays(rays: np.ndarray) -> np.ndarray:
    rays = np.nan_to_num(np.asarray(rays, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if rays.ndim == 1:
        rays = rays.reshape(1, -1) if rays.size else np.zeros((0, 36), np.float32)
    if rays.ndim != 2 or rays.shape[0] == 0:
        return np.zeros((0, 38), np.float32)
    if rays.shape[1] > 36:
        rays = rays[:, :36]
    elif rays.shape[1] < 36:
        rays = np.pad(rays, ((0, 0), (0, 36 - rays.shape[1])))
    rays = np.clip(rays, 0.0, 200.0)
    area = np.clip(np.pi * np.square(rays.mean(1)), 0.001, None)
    ratio = np.clip(rays.min(1) / np.clip(rays.max(1), 0.001, None), 0.001, 1.0)
    return np.concatenate((np.log(area)[:, None], np.log(ratio)[:, None], rays), 1).astype(np.float32)

def _x2_from_record(record) -> np.ndarray:
    return _x2_from_rays(getattr(record, 'rho_um', np.zeros((0, 36), np.float32)))

def _residual_file(root: Path | None, graph_id: str) -> Path | None:
    if root is None:
        return None
    direct = root / f'{graph_id}.npz'
    if direct.is_file():
        return direct
    try:
        from celllift.matched_geometry_controls.io_utils import safe_component
        hashed = root / f'{safe_component(graph_id)}.npz'
        if hashed.is_file():
            return hashed
    except Exception:
        pass
    return None

def _arrays_from_projection_scene(root: Path, graph_id: str) -> dict[str, np.ndarray] | None:
    path = root / '04_projection_scene_inputs_conditional_geometry' / f'{graph_id}.pt'
    if not path.is_file():
        path = root / '04_projection_scene_inputs' / f'{graph_id}.pt'
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        graph = payload['graph']
    except Exception:
        return None

    def _np(name, default):
        value = getattr(graph, name, default)
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return np.asarray(value)
    ids = _np('nucleus_id', np.zeros((0,), np.int64)).astype(np.int64).reshape(-1)
    rays = _np('nucleus_rays_um', getattr(graph, 'rho_um', np.zeros((len(ids), 36), np.float32)))
    xy = _np('xy_px', np.zeros((len(ids), 2), np.float32)).astype(np.float32)
    if xy.ndim == 1:
        xy = xy.reshape(-1, 2) if xy.size else np.zeros((0, 2), np.float32)
    src = _np('edge_src', np.zeros((0,), np.int64)).astype(np.int64).reshape(-1)
    dst = _np('edge_dst', np.zeros((0,), np.int64)).astype(np.int64).reshape(-1)
    edge = _np('edge_feat', np.zeros((0, 3), np.float32)).astype(np.float32)
    if edge.ndim == 1:
        edge = edge.reshape(-1, 3) if edge.size else np.zeros((0, 3), np.float32)
    return {'ids': ids, 'x2': _x2_from_rays(rays), 'xy': xy if len(xy) == len(ids) else np.zeros((len(ids), 2), np.float32), 'src': src, 'dst': dst, 'edge': edge if edge.ndim == 2 else np.zeros((len(src), 3), np.float32)}

def _load_geometry(reader, root: Path, scene_root: Path, residual_root: Path | None, graph_id: str) -> dict[str, Any]:
    record = None
    try:
        record = GraphRecord.from_bytes(reader.get(graph_id))
        x2 = _x2_from_record(record)
        ids = np.asarray(record.nucleus_id, np.int64).reshape(-1)
        xy = np.asarray(record.xy_px, np.float32)
        src = np.asarray(record.edge_src, np.int64).reshape(-1)
        dst = np.asarray(record.edge_dst, np.int64).reshape(-1)
        edge = np.asarray(record.edge_feat, np.float32)
        if len(ids) == 0 or x2.shape[0] != len(ids):
            raise RuntimeError('empty_or_misaligned_record')
    except Exception:
        packed = _arrays_from_projection_scene(root, graph_id)
        if packed is None:
            x2 = np.zeros((0, 38), np.float32)
            ids = np.zeros((0,), np.int64)
            xy = np.zeros((0, 2), np.float32)
            src = dst = np.zeros((0,), np.int64)
            edge = np.zeros((0, 3), np.float32)
        else:
            x2, ids, xy, src, dst, edge = (packed['x2'], packed['ids'], packed['xy'], packed['src'], packed['dst'], packed['edge'])
        record = None
    if xy.ndim != 2:
        xy = np.zeros((len(ids), 2), np.float32)
    if edge.ndim != 2:
        edge = np.zeros((len(src), 3), np.float32)
    scene = _load_scene(scene_path(scene_root, graph_id))
    x3 = np.nan_to_num(_align(ids, scene), nan=0.0, posinf=0.0, neginf=0.0)
    xr = np.zeros_like(x3)
    residual_file = _residual_file(residual_root, graph_id)
    if residual_file is not None:
        packed = np.load(residual_file)
        xr = np.nan_to_num(_align(ids, {'nucleus_id': packed.get('nucleus_id', ids), 'x3': packed['residual9']}), nan=0.0, posinf=0.0, neginf=0.0)
    return {'ids': ids, 'x2': np.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0), 'x3': x3, 'xr': xr, 'xy': xy, 'src': src, 'dst': dst, 'edge': edge, 'record': record, 'scene': scene}

def scene_path(root: Path, graph_id: str) -> Path:
    direct = root / f'{graph_id}.pt'
    if direct.is_file():
        return direct
    try:
        from celllift.matched_geometry_controls.io_utils import safe_component
        hashed = root / f'{safe_component(graph_id)}.pt'
        if hashed.is_file():
            return hashed
    except Exception:
        pass
    return direct

def _load_scene(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    ids = np.asarray(payload.get('nucleus_id', []), np.int64).reshape(-1)
    geom = np.asarray(payload.get('direct_geometry9', np.zeros((len(ids), 9))), np.float32)
    if geom.ndim == 1:
        geom = geom.reshape(-1, 9)
    out = {'nucleus_id': ids, 'x3': geom}
    for key in ('nucleus_center', 'nucleus_transform', 'cell_center', 'cell_transform', 'valid'):
        if key in payload:
            value = payload[key]
            out[key] = value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
    return out

def _align(ids: np.ndarray, packed: dict[str, np.ndarray] | None, key: str='x3', width: int=9) -> np.ndarray:
    out = np.zeros((len(ids), width), np.float32)
    if packed is None or len(packed.get('nucleus_id', [])) == 0 or key not in packed:
        return out
    src = np.asarray(packed[key], np.float32)
    if src.ndim == 1:
        src = src.reshape(-1, width)
    index = {int(n): i for i, n in enumerate(packed['nucleus_id'])}
    for row, nid in enumerate(ids):
        loc = index.get(int(nid))
        if loc is not None and loc < len(src):
            out[row] = src[loc].reshape(-1)[:width]
    return out

def _author_224(rgb: np.ndarray, augment: bool=False) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, np.uint8)).resize((250, 250), Image.Resampling.BILINEAR)
    return _author_250(np.asarray(image, np.uint8), augment=augment)

def _imagenet_224(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    tensor = np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD

def _nucleus_crop(rgb: np.ndarray, xy_px: np.ndarray, size: int=128) -> np.ndarray:
    height, width = rgb.shape[:2]
    cx, cy = (float(xy_px[0]), float(xy_px[1]))
    r0 = int(round(cy - size / 2))
    c0 = int(round(cx - size / 2))
    canvas = np.full((size, size, 3), 255, np.uint8)
    src_r0 = max(0, r0)
    src_c0 = max(0, c0)
    src_r1 = min(height, r0 + size)
    src_c1 = min(width, c0 + size)
    dst_r0 = src_r0 - r0
    dst_c0 = src_c0 - c0
    if src_r1 > src_r0 and src_c1 > src_c0:
        canvas[dst_r0:dst_r0 + (src_r1 - src_r0), dst_c0:dst_c0 + (src_c1 - src_c0)] = rgb[src_r0:src_r1, src_c0:src_c1]
    return canvas

def _graph_roots(root: Path) -> tuple[Path, Path, Path | None]:
    conditional_geometry = root / '03_graph_cache_conditional_geometry'
    set_encoding = root / '03_graph_cache'
    graph = conditional_geometry if (conditional_geometry / 'graph_index.parquet').is_file() else set_encoding
    scene = root / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
    if not scene.is_dir():
        scene = root / '04_projection_scene_inputs' / 'selected_scene'
    residual = root / '04_projection_scene_inputs_conditional_geometry' / 'residual'
    return (graph, scene, residual if residual.is_dir() else None)

def _dino_from_input(root: Path, graph_id: str, ids: np.ndarray) -> np.ndarray:
    path = root / '04_projection_scene_inputs_conditional_geometry' / f'{graph_id}.pt'
    if not path.is_file():
        path = root / '04_projection_scene_inputs' / f'{graph_id}.pt'
    if not path.is_file():
        return np.zeros((len(ids), 384), np.float32)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    graph = payload.get('graph')
    feats = getattr(graph, 'dino_features', None)
    if feats is None:
        return np.zeros((len(ids), 384), np.float32)
    array = np.asarray(feats.float().cpu() if hasattr(feats, 'float') else feats, np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 384)
    if len(array) == len(ids):
        return array
    return array[:len(ids)] if len(array) > len(ids) else np.pad(array, ((0, len(ids) - len(array)), (0, 0)))

def _tokens(geom: str, x2: np.ndarray, x3: np.ndarray, xr: np.ndarray) -> np.ndarray:
    key = geom.lower()
    empty2, empty3 = (np.zeros((0, 38), np.float32), np.zeros((0, 9), np.float32))
    if len(x2) == 0:
        x2 = empty2
    if len(x3) == 0:
        x3 = empty3
    if len(xr) == 0:
        xr = empty3
    return {'g2': x2, 'n2': x2, 'g3': x3, 'n3': x3, 'gr': xr, 'nr': xr, 'g23': np.concatenate((x2, x3), 1) if len(x2) else np.zeros((0, 47), np.float32), 'n23': np.concatenate((x2, x3), 1) if len(x2) else np.zeros((0, 47), np.float32), 'g2r': np.concatenate((x2, xr), 1) if len(x2) else np.zeros((0, 47), np.float32), 'n2r': np.concatenate((x2, xr), 1) if len(x2) else np.zeros((0, 47), np.float32)}[key]

class ArvanitiWindowDataset(Dataset):

    def __init__(self, split: str, *, geom: str='g2', supervised: bool=True, augment: bool=False, load_dino: bool=False, load_rgb: bool=True, cache_items: bool | None=None, load_geometry: bool=True):
        root = ARVANITI_DATA
        infer = root / '00_manifest' / 'inference_window_manifest_v2.parquet'
        if supervised:
            path = root / '00_manifest' / 'window_manifest.parquet'
            rows = read_parquet(path)
            self.rows = [row for row in rows if _role_ok(row, split) and int(row.get('label_id', -1)) >= 0]
        else:
            rows = read_parquet(infer) if infer.is_file() else read_parquet(root / '00_manifest' / 'window_manifest.parquet')
            self.rows = [row for row in rows if _role_ok(row, split)]
        self.geom = geom.lower()
        self.augment = augment and split.lower() in {'fit', 'train'}
        self.load_dino = load_dino
        self.load_rgb = bool(load_rgb)
        self.load_geometry = bool(load_geometry)
        if not self.load_geometry and load_dino:
            raise ValueError('DINO node inputs require geometry')
        self._rgb_250_cache: dict[int, np.ndarray] | None = {} if self.augment and (not self.load_geometry) and self.load_rgb else None
        if cache_items is None:
            self.cache_items = not self.augment and (not supervised or split.lower() in {'val', 'test'})
        else:
            self.cache_items = bool(cache_items)
        self._item_cache: dict[int, dict[str, Any]] = {}
        graph_root, self.scene_root, self.residual_root = _graph_roots(root)
        self.reader = GraphReader(graph_root)
        self.input_root = root
        self.spatial_root = root / '04_v16c_inputs_v2' / 'dino_spatial'

    def __len__(self) -> int:
        return len(self.rows)

    def _geom(self, graph_id: str) -> dict[str, Any]:
        return _load_geometry(self.reader, self.input_root, self.scene_root, self.residual_root, graph_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache_items and index in self._item_cache:
            return self._item_cache[index]
        row = self.rows[index]
        graph_id = row['graph_id']
        if self.load_geometry:
            packed = self._geom(graph_id)
        else:
            packed = {'ids': np.zeros(0, np.int64), 'x2': np.zeros((0, 38), np.float32), 'x3': np.zeros((0, 9), np.float32), 'xr': np.zeros((0, 9), np.float32), 'src': np.zeros(0, np.int64), 'dst': np.zeros(0, np.int64), 'edge': np.zeros((0, 3), np.float32), 'xy': np.zeros((0, 2), np.float32), 'scene': None}
        token = _tokens(self.geom, packed['x2'], packed['x3'], packed['xr'])
        rgb_path = row.get('rgb_path')
        author = None
        if self._rgb_250_cache is not None and index in self._rgb_250_cache:
            author = _author_250(self._rgb_250_cache[index], augment=True)
        elif self.load_rgb and rgb_path and Path(str(rgb_path)).is_file():
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert('RGB'))
            if rgb.shape[0] >= 700 and rgb.shape[1] >= 700 and (rgb.shape[0] != 375 or rgb.shape[1] != 375):
                r0 = int(row.get('row0_native') or 2 * int(row.get('row0_target') or 0))
                c0 = int(row.get('col0_native') or 2 * int(row.get('col0_target') or 0))
                if rgb.shape[0] == 1550:
                    r0 = int(row.get('row0_target') or 0)
                    c0 = int(row.get('col0_target') or 0)
                    crop = rgb[r0:r0 + 375, c0:c0 + 375]
                else:
                    crop = rgb[r0:r0 + 750, c0:c0 + 750]
                if crop.shape[0] >= 224 and crop.shape[1] >= 224:
                    rgb = crop
            if self._rgb_250_cache is not None:
                resized = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize((250, 250), Image.Resampling.BILINEAR), np.uint8)
                self._rgb_250_cache[index] = resized
                author = _author_250(resized, augment=True)
            else:
                author = _author_224(rgb, augment=self.augment)
        dino = _dino_from_input(self.input_root, graph_id, packed['ids']) if self.load_dino else np.zeros((len(packed['ids']), 384), np.float32)
        item = {'graph_id': graph_id, 'core_id': row.get('core_id', graph_id), 'tokens': token.astype(np.float32), 'label': int(row.get('label_id', -1)), 'label_p1': int(row.get('label_id_p1', -1)), 'label_p2': int(row.get('label_id_p2', -1)), 'rgb': author, 'dino': dino.astype(np.float32), 'image_summary': dino.mean(0) if len(dino) else np.zeros(384, np.float32), 'src': packed['src'], 'dst': packed['dst'], 'edge': packed['edge'], 'xy': packed['xy'], 'scene': packed['scene'], 'mask_path': row.get('nucleus_mask_path'), 'row': row}
        if self.cache_items:
            self._item_cache[index] = item
        return item

def pad_collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max((item['tokens'].shape[-1] for item in batch), default=38)
    count = max((len(item['tokens']) for item in batch), default=1)
    tokens = np.zeros((len(batch), count, width), np.float32)
    mask = np.zeros((len(batch), count), np.float32)
    dino = np.zeros((len(batch), count, 384), np.float32)
    labels, ids, cores, images, summaries, mask_paths = ([], [], [], [], [], [])
    srcs, dsts, edges, xys = ([], [], [], [])
    for i, item in enumerate(batch):
        n = len(item['tokens'])
        if n:
            tokens[i, :n] = item['tokens']
            mask[i, :n] = 1
            d = item.get('dino')
            if d is not None and len(d):
                dino[i, :min(n, len(d))] = d[:n]
        labels.append(item['label'])
        ids.append(item['graph_id'])
        cores.append(item['core_id'])
        if item.get('rgb') is not None:
            images.append(item['rgb'])
        summaries.append(item.get('image_summary', np.zeros(384, np.float32)))
        mask_paths.append((item.get('row') or {}).get('nucleus_mask_path') or item.get('mask_path'))
        srcs.append(np.asarray(item.get('src', []), np.int64))
        dsts.append(np.asarray(item.get('dst', []), np.int64))
        edges.append(np.asarray(item.get('edge', np.zeros((0, 3), np.float32)), np.float32))
        xys.append(np.asarray(item.get('xy', np.zeros((0, 2), np.float32)), np.float32))
    out = {'tokens': torch.from_numpy(tokens), 'mask': torch.from_numpy(mask), 'dino': torch.from_numpy(dino), 'label': torch.tensor(labels, dtype=torch.long), 'graph_id': ids, 'core_id': cores, 'image_summary': torch.from_numpy(np.stack(summaries).astype(np.float32)), 'mask_path': mask_paths, 'src': srcs, 'dst': dsts, 'edge': edges, 'xy': xys}
    if images:
        out['rgb'] = torch.from_numpy(np.stack(images).astype(np.float32))
    return out

class LizardTileDataset(Dataset):

    def __init__(self, split: str, *, geom: str='n2', crops: bool=False, load_dino: bool=False):
        root = LIZARD_DATA
        tiles = root / '00_manifest' / 'tile_manifest_conditional_geometry.parquet'
        if not tiles.is_file():
            tiles = root / '00_manifest' / 'tile_manifest.parquet'
        labels = root / '04_labels_splits_conditional_geometry' / 'nucleus_labels.parquet'
        if not labels.is_file():
            labels = root / '04_labels_splits' / 'nucleus_labels.parquet'
        self.rows = [row for row in read_parquet(tiles) if _role_ok(row, split)]
        by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_parquet(labels):
            if _role_ok(row, split):
                by_graph[row['graph_id']].append(row)
        self.labels = by_graph
        self.geom = geom.lower()
        self.crops = crops
        self.load_dino = load_dino
        graph_root, self.scene_root, self.residual_root = _graph_roots(root)
        self.reader = GraphReader(graph_root)
        self.input_root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        graph_id = row['graph_id']
        packed = _load_geometry(self.reader, self.input_root, self.scene_root, self.residual_root, graph_id)
        x2, ids, xy, src, dst, edge = (packed['x2'], packed['ids'], packed['xy'], packed['src'], packed['dst'], packed['edge'])
        scene, x3, xr = (packed['scene'], packed['x3'], packed['xr'])
        if self.geom in {'j2', 'a2', 'h2'}:
            token = x2
        elif self.geom in {'js', 'jr', 'a3', 'h23'}:
            token = np.concatenate((x2, x3), 1) if len(x2) else np.zeros((0, 47), np.float32)
        else:
            token = _tokens(self.geom if self.geom.startswith('n') else 'n2', x2, x3, xr)
        owned_map = {int(item['nucleus_id']): item for item in self.labels.get(graph_id, []) if item.get('owned')}
        owned = np.array([int(nid) in owned_map for nid in ids], dtype=np.float32)
        labels = np.array([max(int(owned_map[int(nid)]['class_id']) - 1, 0) if int(nid) in owned_map else -1 for nid in ids], dtype=np.int64)
        dino = _dino_from_input(self.input_root, graph_id, ids) if self.load_dino else np.zeros((len(ids), 384), np.float32)
        crops = None
        if self.crops and row.get('rgb_path') and Path(row['rgb_path']).is_file():
            with Image.open(row['rgb_path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            crops = np.stack([_imagenet_224(_nucleus_crop(rgb, xy[i])) if i < len(xy) else _imagenet_224(np.full((128, 128, 3), 255, np.uint8)) for i in range(len(ids))]) if len(ids) else np.zeros((0, 3, 224, 224), np.float32)
        edge3 = np.zeros((len(src), 4), np.float32)
        if scene is not None and len(src) and ('nucleus_center' in scene):
            centers = _align(ids, {'nucleus_id': scene['nucleus_id'], 'x3': np.asarray(scene['nucleus_center'], np.float32)[:, :3]}, 'x3', 3)
            delta = centers[dst] - centers[src] if len(centers) else np.zeros((len(src), 3), np.float32)
            edge3 = np.concatenate((delta, np.linalg.norm(delta, axis=1, keepdims=True)), 1).astype(np.float32)
        return {'graph_id': graph_id, 'roi_id': row.get('roi_id', graph_id), 'group_id': row.get('group_id') or row.get('patient_id'), 'tokens': token.astype(np.float32), 'x2': x2.astype(np.float32), 'x3': x3.astype(np.float32), 'owned': owned, 'label': labels, 'dino': dino.astype(np.float32), 'src': src, 'dst': dst, 'edge': edge.astype(np.float32) if edge.ndim == 2 else np.zeros((0, 3), np.float32), 'edge3': edge3, 'crops': crops, 'ids': ids, 'xy': xy}

class TileGroupedSampler(Sampler):

    def __init__(self, items: list[tuple[int, int, Any]], seed: int=42):
        groups: dict[int, list[int]] = defaultdict(list)
        for index, (tile_index, _, _) in enumerate(items):
            groups[int(tile_index)].append(index)
        self.groups = list(groups.values())
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return sum((len(group) for group in self.groups))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        order = list(self.groups)
        rng.shuffle(order)
        for group in order:
            local = list(group)
            rng.shuffle(local)
            yield from local
        self.epoch += 1

class LizardNucleusDataset(Dataset):

    def __init__(self, split: str, *, geom: str='n2', owned_only: bool=True, crops: bool=False, load_dino: bool=False):
        self.tiles = LizardTileDataset(split, geom=geom, crops=False, load_dino=load_dino or bool(crops) or geom.startswith('h') or geom.startswith('a'))
        self.items = []
        for tile_index, row in enumerate(self.tiles.rows):
            for item in self.tiles.labels.get(row['graph_id'], []):
                if owned_only and (not item.get('owned')):
                    continue
                self.items.append((tile_index, int(item['nucleus_id']), item))
        self.geom = geom
        self.want_crops = crops
        self._cache: dict[int, dict[str, Any]] = {}
        self._rgb_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _one_crop(self, tile_index: int, packed: dict[str, Any], loc: int) -> np.ndarray:
        if tile_index not in self._rgb_cache:
            row = self.tiles.rows[tile_index]
            path = row.get('rgb_path')
            if path and Path(path).is_file():
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
            else:
                rgb = np.full((128, 128, 3), 255, np.uint8)
            self._rgb_cache[tile_index] = rgb
            if len(self._rgb_cache) > 48:
                self._rgb_cache.pop(next(iter(self._rgb_cache)))
        rgb = self._rgb_cache[tile_index]
        xy = packed.get('xy')
        point = xy[loc] if xy is not None and loc < len(xy) else np.zeros(2, np.float32)
        return _imagenet_224(_nucleus_crop(rgb, point))

    def __getitem__(self, index: int) -> dict[str, Any]:
        tile_index, nucleus_id, meta = self.items[index]
        if tile_index not in self._cache:
            self._cache[tile_index] = self.tiles[tile_index]
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        packed = self._cache[tile_index]
        loc_map = packed.get('_loc')
        if loc_map is None:
            loc_map = {int(nid): i for i, nid in enumerate(np.asarray(packed['ids']).reshape(-1))}
            packed['_loc'] = loc_map
        loc = loc_map.get(int(nucleus_id))
        id_miss = loc is None
        width = GEOM_DIM.get(self.geom, 38)
        if id_miss:
            token = np.zeros(width, np.float32)
            dino = np.zeros(384, np.float32)
            crop = np.zeros((3, 224, 224), np.float32) if self.want_crops else None
        else:
            token = packed['tokens'][loc] if len(packed['tokens']) else np.zeros(width, np.float32)
            crop = self._one_crop(tile_index, packed, loc) if self.want_crops else None
            dino = packed['dino'][loc] if len(packed['dino']) else np.zeros(384, np.float32)
        return {'token': token.astype(np.float32), 'label': -1 if id_miss else max(int(meta.get('class_id', 1)) - 1, 0), 'id_miss': id_miss, 'graph_id': packed['graph_id'], 'nucleus_id': nucleus_id, 'roi_id': packed['roi_id'], 'group_id': packed['group_id'], 'dino': dino.astype(np.float32), 'crop': crop, 'index': -1 if id_miss else int(loc)}

def class_weights(rows: list[dict[str, Any]], key: str='class_id', classes: int=6) -> torch.Tensor:
    counts = np.ones(classes, np.float64)
    for row in rows:
        label = int(row.get(key, 1))
        if key == 'class_id':
            label = label - 1
        if 0 <= label < classes:
            counts[label] += 1
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(np.clip(weights, 0, 5.0), dtype=torch.float32)

def lizard_class_weights(split: str='fit') -> torch.Tensor:
    root = LIZARD_DATA
    labels = root / '04_labels_splits_conditional_geometry' / 'nucleus_labels.parquet'
    if not labels.is_file():
        labels = root / '04_labels_splits' / 'nucleus_labels.parquet'
    rows = [row for row in read_parquet(labels) if _role_ok(row, split) and row.get('owned')]
    return class_weights(rows)

def _author_250(array: np.ndarray, augment: bool=False) -> np.ndarray:
    if augment:
        if np.random.rand() < 0.5:
            array = np.fliplr(array)
        k = int(np.random.randint(0, 4))
        if k:
            array = np.rot90(array, k)
    r0 = (array.shape[0] - 224) // 2
    c0 = (array.shape[1] - 224) // 2
    crop = array[r0:r0 + 224, c0:c0 + 224]
    tensor = crop.transpose(2, 0, 1).astype(np.float32)
    return tensor / 127.5 - 1.0
