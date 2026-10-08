from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from functools import lru_cache
from celllift.runtime import torch

def _support_bisection(ratio, p, steps, tiny):
    lower = torch.zeros_like(ratio)
    upper = torch.ones_like(ratio)
    for _ in range(steps):
        middle = (lower + upper) * 0.5
        derivative = float(p) * 0.5 * middle.pow(float(p) - 1) / (1 - middle.pow(float(p))).clamp_min(tiny).sqrt()
        lower = torch.where(derivative < ratio, middle, lower)
        upper = torch.where(derivative < ratio, upper, middle)
    return lower + upper

@lru_cache(None)
def _compiled():
    return torch.compile(_support_bisection, fullgraph=True, dynamic=True)

def support_bisection(ratio, p, steps, tiny):
    if ratio.is_cuda and os.environ.get('DMR_COMPILE_SUPPORT', '0') == '1':
        return _compiled()(ratio, p, steps, tiny)
    return _support_bisection(ratio, p, steps, tiny)
