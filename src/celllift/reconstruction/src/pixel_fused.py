from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib

@triton.jit
def _inside(BASE, DIR, PLANES, OUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    b0 = tl.load(BASE + 3 * i, mask, 0)
    b1 = tl.load(BASE + 3 * i + 1, mask, 0)
    b2 = tl.load(BASE + 3 * i + 2, mask, 0)
    d0 = tl.load(DIR + 3 * i, mask, 0)
    d1 = tl.load(DIR + 3 * i + 1, mask, 0)
    d2 = tl.load(DIR + 3 * i + 2, mask, 0)
    plane = tl.load(PLANES + i, mask, 0).to(tl.float32)
    lower = (plane - 1.0) * 5.0
    upper = plane * 5.0
    for _ in range(36):
        z = ((lower + upper) * 0.5).to(tl.float64)
        q0 = b0 + d0 * z
        q1 = b1 + d1 * z
        q2 = b2 + d2 * z
        a = tl.abs(q2)
        cube = a * a * a
        sign = tl.where(q2 > 0, 1.0, tl.where(q2 < 0, -1.0, 0.0))
        grad = 2.0 * (q0 * d0 + q1 * d1) + 4.0 * cube * sign * d2
        lower = tl.where(grad < 0, z.to(tl.float32), lower)
        upper = tl.where(grad < 0, upper, z.to(tl.float32))
    z = ((lower + upper) * 0.5).to(tl.float64)
    q0 = b0 + d0 * z
    q1 = b1 + d1 * z
    q2 = b2 + d2 * z
    power = lib.pow(tl.abs(q2), tl.full((BLOCK,), 4.0, tl.float64))
    tl.store(OUT + i, q0 * q0 + q1 * q1 + power <= 1.0, mask)

def inside(base, zdir, planes):
    assert base.dtype == torch.float64 and zdir.dtype == torch.float64
    base = base.contiguous()
    zdir = zdir.contiguous()
    planes = planes.contiguous()
    result = torch.empty(len(base), device=base.device, dtype=torch.bool)
    if len(base):
        _inside[triton.cdiv(len(base), 128),](base, zdir, planes, result, len(base), 128, num_warps=4, enable_fp_fusion=False)
    return result
