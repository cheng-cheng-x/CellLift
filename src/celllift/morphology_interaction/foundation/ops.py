from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor

def segment_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size,) + tuple(values.shape[1:]))
    if values.shape[0]:
        output.index_add_(0, index, values)
    return output

def segment_count(index: Tensor, size: int, dtype=torch.float32) -> Tensor:
    return segment_sum(torch.ones(index.shape[0], dtype=dtype, device=index.device), index, size)

def segment_mean(values: Tensor, index: Tensor, size: int) -> Tensor:
    total = segment_sum(values, index, size)
    count = segment_count(index, size, values.dtype).clamp_min(1.0)
    return total / count.view((-1,) + (1,) * (values.dim() - 1))

def segment_std(values: Tensor, index: Tensor, size: int) -> Tensor:
    mean = segment_mean(values, index, size)
    centered = values - mean[index]
    variance = segment_mean(centered.square(), index, size).clamp_min(1e-06)
    return torch.sqrt(variance)

def segment_softmax(scores: Tensor, index: Tensor, size: int) -> Tensor:
    if scores.numel() == 0:
        return scores
    if scores.ndim == 1:
        maximum = scores.new_full((size,), -torch.inf)
        maximum = maximum.scatter_reduce(0, index, scores, reduce='amax', include_self=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        exponent = torch.exp(scores - maximum[index])
        return exponent / segment_sum(exponent, index, size).clamp_min(1e-12)[index]
    heads = scores.shape[-1]
    flat = scores.reshape(-1)
    expanded = index.unsqueeze(-1) * heads + torch.arange(heads, device=index.device, dtype=index.dtype)
    return segment_softmax(flat, expanded.reshape(-1), size * heads).view_as(scores)
