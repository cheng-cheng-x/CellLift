from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass, fields
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
import numpy as np
from .dino import dino_xy_px, encode_graphs, encode_nuclei, load_rgb
from .io_utils import atomic_torch
from .moments import align_moments, all_instance_moments

@dataclass(frozen=True)
class PublicRow:
    graph_id: str
    split: str
    rgb_path: str
    nucleus_mask_path: str
    patient_id: str
    wsi_id: str
    roi_id: str
    fold: int | None
    final_fold: int | None
    label_id: int | None

def normalize_manifest_row(row: Mapping[str, Any], dataset: str) -> PublicRow:
    graph_id = str(row.get('graph_id', row.get('patch_id', row.get('tile_id'))))
    if graph_id in {'None', ''}:
        raise ValueError('manifest row has no graph/patch/tile ID')
    rgb = row.get('rgb_path', row.get('tile_path', row.get('source_path')))
    mask = row.get('nucleus_mask_path', row.get('mask_path'))
    if rgb is None or mask is None:
        raise ValueError(f'manifest paths missing for {graph_id}')
    patient = str(row.get('patient_id', row.get('case_id', graph_id)))
    wsi = str(row.get('wsi_id', patient))
    roi = str(row.get('roi_id', graph_id))
    fold = row.get('validation_fold', row.get('fold'))
    final_fold = row.get('final_validation_fold')
    label = row.get('label_id', row.get('label_7_id', row.get('target')))
    split = row.get('official_split', row.get('split', row.get('split_new', 'train')))
    return PublicRow(graph_id, str(split).lower(), str(rgb), str(mask), patient, wsi, roi, None if fold is None else int(fold), None if final_fold is None else int(final_fold), None if label is None else int(label))

def connected_components(count: int, undirected_src: np.ndarray, undirected_dst: np.ndarray) -> np.ndarray:
    parent = np.arange(count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value
    for a, b in zip(np.asarray(undirected_src, np.int64), np.asarray(undirected_dst, np.int64)):
        ra, rb = (find(int(a)), find(int(b)))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.asarray([find(i) for i in range(count)])
    _, code = np.unique(roots, return_inverse=True)
    return code.astype(np.int64)

def build_graph(modules, record, label_mask: np.ndarray, dino_features, mpp: float, graph_id: str):
    import torch
    moments = align_moments(all_instance_moments(label_mask, mpp), record.nucleus_id)
    edge = np.stack((record.edge_src, record.edge_dst)).astype(np.int64)
    undirected = np.stack((record.undirected_edge_src, record.undirected_edge_dst)).astype(np.int64)
    component = connected_components(len(record.nucleus_id), record.undirected_edge_src, record.undirected_edge_dst)
    base = modules.data.BaseGraph(nucleus_rays_um=torch.as_tensor(record.rho_um, dtype=torch.float32), nucleus_xy_um=torch.as_tensor(record.xy_um, dtype=torch.float32), edge_index=torch.as_tensor(edge, dtype=torch.long), undirected_edge_index=torch.as_tensor(undirected, dtype=torch.long), graph_ptr=torch.tensor([0, len(record.nucleus_id)], dtype=torch.long), nucleus_id=torch.as_tensor(record.nucleus_id, dtype=torch.long), fitted_center_xy=torch.as_tensor(moments.center_xy_um, dtype=torch.float32), fitted_precision_xy=torch.as_tensor(moments.precision_xy_um2_inv, dtype=torch.float32), fitted_root_xy=torch.as_tensor(moments.root_xy_um, dtype=torch.float32), fitted_area_um2=torch.as_tensor(moments.ellipse_area_um2, dtype=torch.float32), ellipse_circle_fallback=torch.zeros(len(record.nucleus_id), dtype=torch.bool), nucleus_component_index=torch.as_tensor(component, dtype=torch.long))
    coeff = modules.geometry.coefficients(base.fitted_root_xy)
    query = modules.reference.reference(base.fitted_root_xy)[1]
    raw = torch.zeros((len(record.nucleus_id), 9, 10), dtype=torch.float32)
    raw[:, :, 0] = query
    graph = modules.data.Graph(**{field.name: getattr(base, field.name) for field in fields(base)}, dino_features=dino_features.to(dtype=torch.float16, device='cpu'), coeff=coeff, graph_uids=(graph_id,), query=query, initial_raw=raw)
    graph.validate()
    return graph

def cache_one(modules, encoder, graph_reader, row: PublicRow, destination: str | Path, dataset: str, mpp: float, device: str='cuda') -> dict[str, Any]:
    record = modules.compatibility_schemas.GraphRecord.from_bytes(graph_reader.get(row.graph_id))
    mask = np.load(row.nucleus_mask_path, mmap_mode='r')
    image = load_rgb(row.rgb_path, dataset)
    shifted = dino_xy_px(dataset, record.xy_px)
    features = encode_nuclei(encoder, image, shifted, device=device)
    graph = build_graph(modules, record, mask, features, mpp, row.graph_id)
    payload = {'graph': graph, 'metadata': row.__dict__, 'dino_dtype': 'float16'}
    atomic_torch(destination, payload)
    return {'graph_id': row.graph_id, 'nodes': len(record.nucleus_id), 'path': str(destination)}

def load_rows(path: str | Path, dataset: str) -> list[PublicRow]:
    import pyarrow.parquet as pq
    rows = pq.read_table(path, partitioning=None).to_pylist()
    return [normalize_manifest_row(row, dataset) for row in rows]

def cache_many(modules, encoder, graph_reader, rows: list[PublicRow], destinations: list[Path], dataset: str, mpp: float, device: str='cuda') -> list[dict[str, Any]]:
    if len(rows) != len(destinations) or not rows:
        raise ValueError('cache_many rows/destinations mismatch')
    records = [modules.compatibility_schemas.GraphRecord.from_bytes(graph_reader.get(row.graph_id)) for row in rows]
    images = [load_rgb(row.rgb_path, dataset) for row in rows]
    coordinates = [dino_xy_px(dataset, record.xy_px) for record in records]
    features = encode_graphs(encoder, images, coordinates, device=device)
    output = []
    for row, destination, record, feature in zip(rows, destinations, records, features):
        mask = np.load(row.nucleus_mask_path, mmap_mode='r')
        graph = build_graph(modules, record, mask, feature, mpp, row.graph_id)
        atomic_torch(destination, {'graph': graph, 'metadata': row.__dict__, 'dino_dtype': 'float16'})
        output.append({'graph_id': row.graph_id, 'nodes': len(record.nucleus_id), 'path': str(destination)})
    return output
