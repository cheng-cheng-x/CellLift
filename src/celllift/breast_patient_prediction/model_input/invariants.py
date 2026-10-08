from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from ..protocol import IMAGE_SIZE

class InvariantError(RuntimeError):
    pass

def _require(cond: bool, message: str) -> None:
    if not cond:
        raise InvariantError(message)

def check_rgb(path: Path) -> None:
    from PIL import Image
    text = str(path).lower()
    _require(path.suffix.lower() == '.png', f'rgb_path is not PNG: {path}')
    _require('.svs' not in text, f'rgb_path still points at SVS: {path}')
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'))
    _require(array.dtype == np.uint8, f'RGB dtype {array.dtype}')
    _require(array.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f'RGB shape {array.shape}')

def check_mask_graph(mask: np.ndarray, instances: dict[str, Any], record) -> None:
    _require(mask.dtype == np.int32, f'mask dtype {mask.dtype}')
    ids = [int(item['instance_id']) for item in instances.get('instances', [])]
    unique = [int(v) for v in np.unique(mask) if int(v) > 0]
    _require(sorted(ids) == sorted(unique), 'instance count does not match mask ids')
    _require(list(map(int, np.asarray(record.nucleus_id).reshape(-1))) == sorted(ids), 'graph nucleus_id does not match instances')
    if len(record.xy_um):
        _require(np.isfinite(record.xy_um).all(), 'XY is not finite')
        _require(record.xy_um.shape[1] == 2, 'XY is not [N,2] um')
        _require(np.isfinite(record.rho_um).all() and np.all(record.rho_um > 0), 'rays are not positive um')

def check_empty(rgb: Path, record, scene_payload: dict[str, Any] | None, geometry_nodes: int, tile_dino: np.ndarray) -> None:
    check_rgb(rgb)
    _require(len(record.nucleus_id) == 0, 'empty tile still has nuclei')
    _require(scene_payload is None or len(np.asarray(scene_payload.get('nucleus_id', [])).reshape(-1)) == 0, 'empty tile has scene nodes')
    _require(geometry_nodes == 0, 'empty geometry bag is not empty')
    _require(np.asarray(tile_dino).shape == (384,), f'empty tile missing 384D DINO {np.asarray(tile_dino).shape}')

def check_scene_alignment(record, scene_payload: dict[str, Any]) -> None:
    graph_ids = np.asarray(record.nucleus_id).reshape(-1).astype(np.int64)
    scene_ids = np.asarray(scene_payload['nucleus_id']).reshape(-1).astype(np.int64)
    _require(np.array_equal(graph_ids, scene_ids), 'scene nucleus_id does not match graph')
    n = len(graph_ids)
    _require(len(scene_payload['valid']) == n, 'valid length dropped 2D nodes')
    _require(len(scene_payload['valid_ncr']) == n, 'valid_ncr length dropped 2D nodes')
    _require(tuple(np.asarray(scene_payload['nucleus_transform']).shape[-2:]) == (3, 3), 'nucleus transform is not 3x3')
    _require(tuple(np.asarray(scene_payload['cell_transform']).shape[-2:]) == (3, 3), 'cell transform is not 3x3')

def check_batch_vs_single(modules, model, items, device: str, scratch: Path) -> None:
    import torch
    from celllift.matched_geometry_controls.inference import infer_items
    if len(items) < 2:
        return
    pair = items[:2]
    batch_records = infer_items(modules, model, pair, scratch / 'batch', device, max_graphs=8, max_nodes=8192)
    single_records = []
    for index, item in enumerate(pair):
        single_records.extend(infer_items(modules, model, [item], scratch / f'single_{index}', device, max_graphs=1, max_nodes=8192))
    by_graph = {row['graph_id']: row['path'] for row in batch_records}
    for row in single_records:
        batched = torch.load(by_graph[row['graph_id']], map_location='cpu', weights_only=False)
        single = torch.load(row['path'], map_location='cpu', weights_only=False)
        if not torch.equal(batched['selected_candidate_id'], single['selected_candidate_id']):
            raise InvariantError('selected_candidate_id mismatch between single and batch')
        if not torch.equal(batched['valid'], single['valid']):
            raise InvariantError('valid mismatch between single and batch')
        torch.testing.assert_close(batched['nucleus_transform'], single['nucleus_transform'], rtol=0.0002, atol=5e-06)
        torch.testing.assert_close(batched['cell_transform'], single['cell_transform'], rtol=0.0002, atol=5e-06)
