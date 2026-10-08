from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from . import layout

def _slice(arrays: dict[str, np.ndarray], key: str, index: int) -> np.ndarray:
    ptr = arrays[f'{key}_ptr']
    begin, end = (int(ptr[index]), int(ptr[index + 1]))
    return arrays[key][begin:end]

class ThreeViewLoader:

    def __init__(self, base: Path, tiles_index):
        self.base = Path(base)
        self.tiles = {str(row['graph_id']): dict(row) for row in tiles_index}

    def identity(self, graph_id: str) -> dict[str, Any]:
        row = self.tiles[str(graph_id)]
        return {'graph_id': str(row['graph_id']), 'tile_id': str(row['tile_id']), 'patient_id': str(row['patient_id']), 'slide_id': str(row['slide_id']), 'split': str(row.get('split') or ''), 'rgb_path': str(row['rgb_path'])}

    def rgb_tile(self, graph_id: str) -> dict[str, Any]:
        row = self.identity(graph_id)
        row['dino'] = np.asarray(self.tiles[str(graph_id)]['dino'], dtype=np.float32)
        return row

    def geometry(self, graph_id: str) -> dict[str, Any]:
        row = self.identity(graph_id)
        path = layout.geometry_path(self.base, row['slide_id'])
        with np.load(path, allow_pickle=False) as data:
            ids = [str(item) for item in data['graph_ids'].tolist()]
            index = ids.index(str(graph_id))
            payload = {key: _slice(data, key, index) for key in ('nucleus_id', 'rays', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy')}
            payload['empty'] = bool(data['empty'][index])
        row.update(payload)
        return row

    def scene_graph(self, graph_id: str) -> dict[str, Any]:
        row = self.identity(graph_id)
        path = layout.scene_graph_path(self.base, row['slide_id'])
        with np.load(path, allow_pickle=False) as data:
            ids = [str(item) for item in data['graph_ids'].tolist()]
            index = ids.index(str(graph_id))
            payload = {key: _slice(data, key, index) for key in ('nucleus_id', 'dino', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy', 'nucleus_transform', 'cell_transform', 'edge2d', 'edge3d')}
            begin, end = (int(data['edge_index_ptr'][index]), int(data['edge_index_ptr'][index + 1]))
            payload['edge_index'] = data['edge_index'][:, begin:end]
            payload['empty'] = bool(data['empty'][index])
        row.update(payload)
        return row

    def aligned(self, graph_id: str) -> dict[str, Any]:
        rgb = self.rgb_tile(graph_id)
        geom = self.geometry(graph_id)
        scene = self.scene_graph(graph_id)
        for key in ('graph_id', 'tile_id', 'patient_id'):
            if rgb[key] != geom[key] or rgb[key] != scene[key]:
                raise RuntimeError(f'{key} mismatch across views for {graph_id}')
        return {'rgb_tiles': rgb, 'geometry_tokens': geom, 'scene_graphs': scene}
