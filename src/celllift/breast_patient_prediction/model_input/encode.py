from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Sequence
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from celllift.matched_geometry_controls.adapter import build_graph
from celllift.matched_geometry_controls.dino import dino_xy_px, load_rgb
from celllift.matched_geometry_controls.geometry import selected_scene_descriptor_torch
from celllift.matched_geometry_controls.inference import device_limits, graph_batches, load_model
from celllift.matched_geometry_controls.upstream import load_projection_scene
from celllift.morphology_interaction.features import assemble_graph, extract_dino, resolve_observed_xy
from ..io_utils import atomic_json, atomic_npz
from ..protocol import DINO_DIM, DINO_SOURCE_ROOT, DINO_WEIGHT_PATH, TARGET_MPP, ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST, ProjectionScene_SOURCE_ROOT
from . import layout
from .graphs import load_slide_graphs
from .segment import load_slide_masks

def dino_tile_batch_size(device: str='cuda') -> int:
    try:
        import torch
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            return 1
        name = torch.cuda.get_device_name(0)
        if 'A800' in name or 'A100' in name:
            return 16
        if 'auxiliary_ad5736' in name:
            return 4
        return 2
    except Exception:
        return 2

def encode_tile_and_nuclei(encoder, images_hwc: Sequence[np.ndarray], xy_px: Sequence[np.ndarray], device: str='cuda', batch_size: int=8):
    import torch
    if len(images_hwc) != len(xy_px) or not images_hwc:
        raise ValueError('images/coordinate batches must be nonempty and aligned')
    nucleus_out = []
    tile_out = []
    step = max(1, int(batch_size))
    for start in range(0, len(images_hwc), step):
        chunk_images = images_hwc[start:start + step]
        chunk_xy = xy_px[start:start + step]
        images = torch.from_numpy(np.stack([np.ascontiguousarray(image.transpose(2, 0, 1)) for image in chunk_images])).to(device)
        counts = [len(np.asarray(value)) for value in chunk_xy]
        with torch.inference_mode():
            field = encoder(images)
            patch_mean = field.mean(dim=(2, 3))
            maximum = max(counts)
            sampled = None
            if maximum > 0:
                grid = torch.full((len(counts), maximum, 1, 2), -1.0, dtype=torch.float32, device=device)
                for index, value in enumerate(chunk_xy):
                    if not len(value):
                        continue
                    xy = torch.as_tensor(value, dtype=torch.float32, device=device)
                    grid[index, :len(xy), 0] = 2.0 * (xy + float(encoder.source_padding)) / float(encoder.encoded_size) - 1.0
                sampled = torch.nn.functional.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=False)
            for index, count in enumerate(counts):
                if count > 0:
                    feat = sampled[index, :, :count, 0].T.contiguous()
                    nucleus_out.append(feat)
                    tile_out.append(feat.mean(dim=0))
                else:
                    nucleus_out.append(field.new_zeros((0, DINO_DIM)))
                    tile_out.append(patch_mean[index])
        del images
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return (nucleus_out, tile_out)

def load_encoder(device: str='cuda'):
    modules = load_projection_scene(ProjectionScene_SOURCE_ROOT)
    encoder = modules.he_features.DINOv2S14Encoder(Path(DINO_SOURCE_ROOT), Path(DINO_WEIGHT_PATH)).to(device).eval()
    return (modules, encoder)

def _cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value

def infer_selected_scenes(modules, model, items, device: str='cuda') -> dict[str, dict[str, Any]]:
    import torch
    gpu = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else 'CPU'
    max_graphs, max_nodes = device_limits(gpu)
    output: dict[str, dict[str, Any]] = {}
    if not items:
        return output
    with torch.inference_mode():
        for packed in graph_batches(items, max_graphs, max_nodes):
            graph, metadata = modules.data.collate(packed)
            graph = modules.data.move(graph, device)
            proposal = model.propose(graph, scoring=True)
            scene = modules.model.selected(proposal, labels=proposal.labels)
            direct, valid_ncr = selected_scene_descriptor_torch(scene.nucleus, scene.cell)
            ptr = graph.graph_ptr.detach().cpu().numpy()
            labels = proposal.labels.detach().cpu()
            valid = scene.valid.detach().cpu()
            for index, meta in enumerate(metadata):
                begin, end = (int(ptr[index]), int(ptr[index + 1]))
                graph_id = str(meta['graph_id'])
                output[graph_id] = {'graph_id': graph_id, 'nucleus_id': graph.nucleus_id[begin:end].detach().cpu(), 'selected_candidate_id': labels[begin:end], 'valid': valid[begin:end], 'nucleus_center': scene.nucleus.center[begin:end].detach().cpu(), 'nucleus_transform': scene.nucleus.transform[begin:end].detach().cpu(), 'cell_center': scene.cell.center[begin:end].detach().cpu(), 'cell_transform': scene.cell.transform[begin:end].detach().cpu(), 'direct_geometry9': direct[begin:end].detach().cpu(), 'valid_ncr': valid_ncr[begin:end].detach().cpu(), 'metadata': meta}
    return output

def _pack_variable(rows: list[dict[str, Any]], array_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    graph_ids = np.asarray([row['graph_id'] for row in rows], dtype='U160')
    tile_ids = np.asarray([row['tile_id'] for row in rows], dtype='U128')
    empty = np.asarray([bool(row['empty']) for row in rows])
    packed: dict[str, np.ndarray] = {'graph_ids': graph_ids, 'tile_ids': tile_ids, 'empty': empty}
    for key in array_keys:
        pieces = [np.asarray(row[key]) for row in rows]
        counts = np.asarray([0 if item.ndim == 0 else item.shape[0] for item in pieces], np.int64)
        ptr = np.zeros(len(counts) + 1, np.int64)
        ptr[1:] = np.cumsum(counts)
        if pieces and pieces[0].ndim >= 1:
            tail = pieces[0].shape[1:]
            total = int(ptr[-1])
            blob = np.zeros((total, *tail), dtype=pieces[0].dtype) if total else np.zeros((0, *tail), dtype=pieces[0].dtype)
            offset = 0
            for item in pieces:
                n = item.shape[0]
                if n:
                    blob[offset:offset + n] = item
                    offset += n
            packed[key] = blob
            packed[f'{key}_ptr'] = ptr
        else:
            packed[key] = np.asarray(pieces)
    return packed

def infer_slide(job: dict[str, Any], *, device: str='cuda', modules=None, encoder=None, model=None) -> dict[str, Any]:
    import torch
    from concurrent.futures import ThreadPoolExecutor
    torch.set_num_threads(min(4, int(job.get('cpu_threads') or 4)))
    base = Path(job['model_input_root'])
    slide_id = job['slide_id']
    print(f"scene start {slide_id} n={len(job['tiles'])}", flush=True)
    if modules is None or encoder is None:
        modules, encoder = load_encoder(device)
    if model is None:
        model, _ = load_model(modules, ProjectionScene_CHECKPOINT, ProjectionScene_INPUT_MANIFEST, device)
    records = load_slide_graphs(layout.graph_path(base, slide_id))
    masks = load_slide_masks(layout.mask_npz_path(base, slide_id))

    def _load_row(row: dict[str, Any]):
        tile_id = str(row['tile_id'])
        rgb = load_rgb(row['rgb_path'], 'tcga_brca')
        record = records[tile_id]
        return (row, record, rgb, dino_xy_px('tcga_brca', record.xy_px))
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(job['tiles'])))) as pool:
        loaded = list(pool.map(_load_row, job['tiles']))
    order = [(row, record, rgb) for row, record, rgb, _xy in loaded]
    images = [rgb for _row, _record, rgb, _xy in loaded]
    xy_list = [xy for _row, _record, _rgb, xy in loaded]
    nucleus_feats, tile_feats = encode_tile_and_nuclei(encoder, images, xy_list, device=device, batch_size=dino_tile_batch_size(device))
    projection_scene_items = []
    tile_rows = []
    geometry_rows = []
    scene_rows = []
    empty_scenes = []
    for (row, record, rgb), nuc, tile_vec in zip(order, nucleus_feats, tile_feats):
        tile_id = str(row['tile_id'])
        graph_id = str(record.graph_id)
        patient_id = str(row['patient_id'])
        split = str(row.get('split') or '')
        tile_np = np.asarray(tile_vec.detach().cpu(), dtype=np.float32)
        tile_rows.append({'tile_id': tile_id, 'graph_id': graph_id, 'patient_id': patient_id, 'slide_id': slide_id, 'split': split, 'dino': tile_np, 'empty_nuclei': len(record.nucleus_id) == 0, 'n_nodes': int(len(record.nucleus_id))})
        meta = {'graph_id': graph_id, 'tile_id': tile_id, 'patient_id': patient_id, 'slide_id': slide_id, 'split': split, 'wsi_id': slide_id, 'roi_id': tile_id}
        if len(record.nucleus_id) == 0:
            empty_scenes.append(graph_id)
            geometry_rows.append({'graph_id': graph_id, 'tile_id': tile_id, 'empty': True, 'nucleus_id': np.empty(0, np.int64), 'rays': np.empty((0, 36), np.float32), 'node2d': np.empty((0, 38), np.float32), 'node3d': np.empty((0, 12), np.float32), 'include': np.empty(0, np.bool_), 'valid3d': np.empty(0, np.bool_), 'center_xy': np.empty((0, 2), np.float32)})
            scene_rows.append({'graph_id': graph_id, 'tile_id': tile_id, 'empty': True, 'nucleus_id': np.empty(0, np.int64), 'dino': np.empty((0, DINO_DIM), np.float32), 'node2d': np.empty((0, 38), np.float32), 'node3d': np.empty((0, 12), np.float32), 'include': np.empty(0, np.bool_), 'valid3d': np.empty(0, np.bool_), 'center_xy': np.empty((0, 2), np.float32), 'nucleus_transform': np.empty((0, 3, 3), np.float32), 'cell_transform': np.empty((0, 3, 3), np.float32), 'edge_index': np.empty((2, 0), np.int32), 'edge2d': np.empty((0, 4), np.float32), 'edge3d': np.empty((0, 8), np.float32)})
            continue
        graph = build_graph(modules, record, masks[tile_id], nuc, TARGET_MPP, graph_id)
        projection_scene_items.append((graph, meta, record, nuc))
    scenes = infer_selected_scenes(modules, model, [(item[0], item[1]) for item in projection_scene_items], device=device)
    for graph, meta, record, nuc in projection_scene_items:
        graph_id = str(meta['graph_id'])
        payload = scenes[graph_id]
        ids = np.asarray(graph.nucleus_id).reshape(-1).astype(np.int64)
        scene_ids = np.asarray(payload['nucleus_id']).reshape(-1).astype(np.int64)
        if not np.array_equal(ids, scene_ids) or not np.array_equal(ids, np.asarray(record.nucleus_id).reshape(-1).astype(np.int64)):
            raise RuntimeError(f'nucleus_id misaligned for {graph_id}')
        observed = resolve_observed_xy(np.asarray(graph.fitted_center_xy, np.float64), np.asarray(graph.nucleus_xy_um, np.float64))
        dino = extract_dino(graph)
        packed = assemble_graph(payload, np.asarray(record.rho_um, np.float32), ids, observed, ids, dino)
        geometry_rows.append({'graph_id': graph_id, 'tile_id': meta['tile_id'], 'empty': False, 'nucleus_id': ids, 'rays': np.asarray(record.rho_um, np.float32), 'node2d': packed['node2d'], 'node3d': packed['node3d'], 'include': packed['include'], 'valid3d': packed['valid3d'], 'center_xy': packed['center_xy']})
        scene_rows.append({'graph_id': graph_id, 'tile_id': meta['tile_id'], 'empty': False, 'nucleus_id': ids, 'dino': packed['dino'], 'node2d': packed['node2d'], 'node3d': packed['node3d'], 'include': packed['include'], 'valid3d': packed['valid3d'], 'center_xy': packed['center_xy'], 'nucleus_transform': packed['nucleus_transform'], 'cell_transform': packed['cell_transform'], 'edge_index': packed['edge_index'], 'edge2d': packed['edge2d'], 'edge3d': packed['edge3d']})
        scenes[graph_id] = {key: _cpu(value) for key, value in payload.items()}
    dest_scene = layout.scene_path(base, slide_id)
    dest_scene.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'slide_id': slide_id, 'scenes': scenes, 'empty_graph_ids': empty_scenes}, dest_scene)
    geom = _pack_variable(geometry_rows, ('nucleus_id', 'rays', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy'))
    scene_pack = _pack_variable(scene_rows, ('nucleus_id', 'dino', 'node2d', 'node3d', 'include', 'valid3d', 'center_xy', 'nucleus_transform', 'cell_transform', 'edge2d', 'edge3d'))
    edge_pieces = [np.asarray(row['edge_index']) for row in scene_rows]
    edge_counts = np.asarray([item.shape[1] if item.ndim == 2 else 0 for item in edge_pieces], np.int64)
    edge_ptr = np.zeros(len(edge_counts) + 1, np.int64)
    edge_ptr[1:] = np.cumsum(edge_counts)
    edges = np.zeros((2, int(edge_ptr[-1])), np.int32)
    offset = 0
    for item in edge_pieces:
        n = int(item.shape[1]) if item.ndim == 2 else 0
        if n:
            edges[:, offset:offset + n] = item
            offset += n
    scene_pack['edge_index'] = edges
    scene_pack['edge_index_ptr'] = edge_ptr
    atomic_npz(layout.geometry_path(base, slide_id), compressed=True, **geom)
    atomic_npz(layout.scene_graph_path(base, slide_id), compressed=True, **scene_pack)
    tile_path = base / 'features' / f'tile_dino_{layout.shard_name(slide_id)}__{layout._slide_stem(slide_id)}.parquet'
    from ..io_utils import atomic_parquet
    atomic_parquet(tile_path, [{'tile_id': row['tile_id'], 'graph_id': row['graph_id'], 'patient_id': row['patient_id'], 'slide_id': row['slide_id'], 'split': row['split'], 'empty_nuclei': row['empty_nuclei'], 'n_nodes': row['n_nodes'], 'dino': row['dino'].astype(np.float32).tolist()} for row in tile_rows])
    invalid = 0
    nodes = 0
    for row in geometry_rows:
        nodes += int(len(row['nucleus_id']))
        invalid += int((~np.asarray(row['valid3d'], bool)).sum()) if len(row['valid3d']) else 0
    summary = {'slide_id': slide_id, 'stage': 'scene', 'status': 'done', 'n_tiles': len(job['tiles']), 'empty_nuclei': len(empty_scenes), 'nodes': nodes, 'invalid_3d': invalid, 'scene_bytes': dest_scene.stat().st_size if dest_scene.is_file() else 0}
    atomic_json(base / 'logs' / 'status' / f'{dest_scene.stem}.scene.json', summary)
    return {'summary': summary, 'tile_rows': tile_rows, 'scenes': scenes, 'items': projection_scene_items}
