from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass, fields
from typing import Any, Mapping
from celllift.runtime import torch
from torch import Tensor

@dataclass(frozen=True)
class EllipseObservationBatch:
    target_center_normalized: Tensor
    target_root_normalized: Tensor
    anchor_node_index: Tensor
    plane_index: Tensor
    weight: Tensor
    raw_area_um2: Tensor
    representation_dice: Tensor
    circle_fallback: Tensor

    def validate(self, node_count: int) -> None:
        o = self.anchor_node_index.numel()
        if self.target_center_normalized.shape != (o, 2):
            raise ValueError('target_center_normalized must have shape [O,2]')
        if self.target_root_normalized.shape != (o, 2, 2):
            raise ValueError('target_root_normalized must have shape [O,2,2]')
        for name in ('plane_index', 'weight', 'raw_area_um2', 'representation_dice', 'circle_fallback'):
            if getattr(self, name).shape != (o,):
                raise ValueError(f'{name} must have shape [O]')
        if o and (self.anchor_node_index.min() < 0 or self.anchor_node_index.max() >= node_count):
            raise ValueError('observation anchor index is out of range')

@dataclass(frozen=True)
class ProjectionSupervision:
    nucleus: EllipseObservationBatch
    cell: EllipseObservationBatch
    positive_mask: Tensor
    unmatched_mask: Tensor
    confirmed_empty_mask: Tensor
    detection_area_um2: Tensor

    def validate(self, node_count: int) -> None:
        self.nucleus.validate(node_count)
        self.cell.validate(node_count)
        for name in ('positive_mask', 'unmatched_mask', 'confirmed_empty_mask'):
            if getattr(self, name).shape != (node_count, 2):
                raise ValueError(f'{name} must have shape [N,2]')
        if self.detection_area_um2.numel() != 1:
            raise ValueError('detection_area_um2 must be scalar')

def move_observations(value: EllipseObservationBatch, device: torch.device | str) -> EllipseObservationBatch:
    return EllipseObservationBatch(**{f.name: getattr(value, f.name).to(device, non_blocking=True) for f in fields(value)})

def move_supervision(value: ProjectionSupervision, device: torch.device | str) -> ProjectionSupervision:
    return ProjectionSupervision(nucleus=move_observations(value.nucleus, device), cell=move_observations(value.cell, device), positive_mask=value.positive_mask.to(device, non_blocking=True), unmatched_mask=value.unmatched_mask.to(device, non_blocking=True), confirmed_empty_mask=value.confirmed_empty_mask.to(device, non_blocking=True), detection_area_um2=value.detection_area_um2.to(device, non_blocking=True))

@dataclass(frozen=True)
class NucleusGraphBatch:
    nucleus_rays_um: Tensor
    nucleus_xy_um: Tensor
    edge_index: Tensor
    undirected_edge_index: Tensor
    graph_ptr: Tensor
    nucleus_id: Tensor
    fitted_center_xy: Tensor
    fitted_precision_xy: Tensor
    fitted_root_xy: Tensor
    fitted_area_um2: Tensor
    ellipse_circle_fallback: Tensor
    nucleus_component_index: Tensor | None = None

    @classmethod
    def from_mapping(cls, batch: Mapping[str, Any]) -> 'NucleusGraphBatch':
        names = {field.name for field in fields(cls)}
        extra = set(batch) - names
        if extra:
            raise ValueError(f'unexpected/loss-only fields cannot enter moment_shape forward: {sorted(extra)}')
        missing = names - {'nucleus_component_index'} - set(batch)
        if missing:
            raise KeyError(f'missing deterministic middle nucleus fields: {sorted(missing)}')
        result = cls(**dict(batch))
        result.validate()
        return result

    def validate(self) -> None:
        n = self.nucleus_id.numel()
        expected = {'nucleus_id': (n,), 'nucleus_rays_um': (n, 36), 'nucleus_xy_um': (n, 2), 'fitted_center_xy': (n, 2), 'fitted_precision_xy': (n, 2, 2), 'fitted_root_xy': (n, 2, 2), 'fitted_area_um2': (n,), 'ellipse_circle_fallback': (n,)}
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
        for name in ('edge_index', 'undirected_edge_index'):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[0] != 2:
                raise ValueError(f'{name} must have shape [2,E]')
        if self.graph_ptr.ndim != 1:
            raise ValueError('graph_ptr must have shape [G+1]')
        if self.nucleus_component_index is not None and self.nucleus_component_index.shape != (n,):
            raise ValueError('nucleus_component_index must have shape [N]')

    def pin_memory(self) -> 'NucleusGraphBatch':
        return NucleusGraphBatch(**{field.name: getattr(self, field.name).pin_memory() if getattr(self, field.name) is not None else None for field in fields(self)})
