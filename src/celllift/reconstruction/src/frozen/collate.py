from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import fields
from typing import Sequence
from celllift.runtime import torch
from .batch_types import EllipseObservationBatch, NucleusGraphBatch

def _cat_observations(values: Sequence[EllipseObservationBatch], offsets: list[int]):
    result = {}
    for field in fields(EllipseObservationBatch):
        parts = [getattr(value, field.name) for value in values]
        if field.name == 'anchor_node_index':
            parts = [part + offset for part, offset in zip(parts, offsets)]
        result[field.name] = torch.cat(parts, dim=0)
    return EllipseObservationBatch(**result)

def collate_graphs(graphs: Sequence[NucleusGraphBatch]):
    if not graphs:
        raise ValueError('cannot collate an empty graph batch')
    offsets, pointers = ([], [0])
    components = []
    component_offset = 0
    for graph in graphs:
        offsets.append(pointers[-1])
        pointers.append(pointers[-1] + graph.nucleus_id.numel())
        local = graph.nucleus_component_index
        if local is None:
            local = torch.arange(graph.nucleus_id.numel(), dtype=torch.long)
        components.append(local + component_offset)
        component_offset += int(local.max()) + 1 if local.numel() else 0
    values = {}
    for field in fields(NucleusGraphBatch):
        name = field.name
        if name == 'graph_ptr':
            values[name] = torch.tensor(pointers, dtype=torch.long)
        elif name == 'nucleus_component_index':
            values[name] = torch.cat(components)
        elif name in ('edge_index', 'undirected_edge_index'):
            values[name] = torch.cat([getattr(g, name) + o for g, o in zip(graphs, offsets)], dim=1)
        else:
            values[name] = torch.cat([getattr(g, name) for g in graphs])
    return (NucleusGraphBatch(**values), offsets)
