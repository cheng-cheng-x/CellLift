from celllift.runtime import resource_path as _public_resource
import math
from functools import lru_cache
import torch
import triton
import triton.language as tl
from celllift.reconstruction.src.frozen.collision import kp_support
from celllift.reconstruction.src.frozen.contact_fused import contact_scale
FACTORS = {'cylinder': 2 * math.pi, 'ellipsoid': 4 * math.pi / 3, 'p4': 8 * math.pi / 5}

def pairs(center, T, valid, method, block=256):
    if method == 'cylinder':
        ext = T[:, :, :2].norm(dim=-1) + T[:, :, 2].abs()
    else:
        ext = kp_support(T, 2 if method == 'ellipsoid' else 4)
    pad = 8 * torch.finfo(center.dtype).eps * (center.abs() + ext + 1)
    lo, hi = (center - ext - pad, center + ext + pad)
    edges = []
    all_ids = torch.arange(len(center), device=center.device)
    for start in range(0, len(center), block):
        ids = all_ids[start:start + block]
        hit = (lo[ids, None] <= hi[None]).all(-1) & (lo[None] <= hi[ids, None]).all(-1)
        hit &= valid[ids, None] & valid[None] & (ids[:, None] < all_ids[None])
        a, b = torch.where(hit)
        if len(a):
            edges.append(torch.stack((ids[a], b)))
    return torch.cat(edges, 1) if edges else torch.empty((2, 0), device=center.device, dtype=torch.long)

def ellipse_contact(Q1, Q2, delta):
    lo = delta.new_zeros(len(delta))
    hi = torch.ones_like(lo)

    def f(t):
        Q = Q1 * (1 - t[:, None, None]) + Q2 * t[:, None, None]
        x, y = delta.unbind(1)
        a = Q[:, 0, 0]
        b = Q[:, 0, 1]
        d = Q[:, 1, 1]
        return t * (1 - t) * (d * x * x - 2 * b * x * y + a * y * y) / (a * d - b * b)
    for _ in range(48):
        l = (2 * lo + hi) / 3
        r = (lo + 2 * hi) / 3
        less = f(l) < f(r)
        lo = torch.where(less, l, lo)
        hi = torch.where(less, hi, r)
    return f((lo + hi) / 2).clamp_min(0).sqrt()

def contact_pen(center, T, edge, coaxial=False, chunk=8192):
    i, j = edge
    values = []
    if coaxial:
        Q = T[:, :2, :2] @ T[:, :2, :2].transpose(-1, -2)
    for start in range(0, len(i), chunk):
        a, b = (i[start:start + chunk], j[start:start + chunk])
        if coaxial:
            alpha = ellipse_contact(Q[a], Q[b], center[b, :2] - center[a, :2])
        else:
            alpha, _ = contact_scale(center[a], T[a], center[b], T[b], 4)
        values.append((1 - alpha).clamp_min(0))
    return torch.cat(values) if values else center.new_empty(0)

def max_penalty(n, edge, pen):
    out = pen.new_zeros(n)
    if len(pen):
        out.scatter_reduce_(0, edge[0], pen, reduce='amax', include_self=True)
        out.scatter_reduce_(0, edge[1], pen, reduce='amax', include_self=True)
    return out

@lru_cache(None)
def unit_points(method, reps=4, samples=512, device='cuda'):
    u = torch.stack([torch.quasirandom.SobolEngine(3, scramble=True, seed=42 + 1009 * r).draw(samples, dtype=torch.float64) for r in range(reps)]).to(device)
    signed = 2 * u[..., 0] - 1
    if method == 'cylinder':
        t = signed
        radius = torch.ones_like(t)
    else:
        p = 2 if method == 'ellipsoid' else 4
        y = signed.abs()
        lo = torch.zeros_like(y)
        hi = torch.ones_like(y)
        for _ in range(40):
            mid = (lo + hi) / 2
            below = ((p + 1) * mid - mid.pow(p + 1)) / p < y
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        t = (lo + hi) / 2 * signed.sign()
        radius = (1 - t.pow(p)).clamp_min(0).sqrt()
    radius *= u[..., 2].sqrt()
    theta = 2 * math.pi * u[..., 1]
    return torch.stack((radius * theta.cos(), radius * theta.sin(), t), -1).contiguous()

def csr(edge, n):
    a = torch.cat((edge[0], edge[1]))
    b = torch.cat((edge[1], edge[0]))
    order = a.argsort(stable=True)
    counts = torch.bincount(a, minlength=n)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return (ptr, b[order])

@triton.jit
def _counts(C, T, I, PTR, NEIGH, U, OUT, S: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr):
    body = tl.program_id(0)
    point = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = point < S
    q0 = tl.load(U + point * 3, mask, 0.0)
    q1 = tl.load(U + point * 3 + 1, mask, 0.0)
    q2 = tl.load(U + point * 3 + 2, mask, 0.0)
    x = tl.load(C + body * 3) + tl.load(T + body * 9) * q0 + tl.load(T + body * 9 + 1) * q1 + tl.load(T + body * 9 + 2) * q2
    y = tl.load(C + body * 3 + 1) + tl.load(T + body * 9 + 3) * q0 + tl.load(T + body * 9 + 4) * q1 + tl.load(T + body * 9 + 5) * q2
    z = tl.load(C + body * 3 + 2) + tl.load(T + body * 9 + 6) * q0 + tl.load(T + body * 9 + 7) * q1 + tl.load(T + body * 9 + 8) * q2
    count = tl.full((BLOCK,), 1, tl.int32)
    begin = tl.load(PTR + body)
    end = tl.load(PTR + body + 1)
    for k in range(begin, end):
        j = tl.load(NEIGH + k)
        dx = x - tl.load(C + j * 3)
        dy = y - tl.load(C + j * 3 + 1)
        dz = z - tl.load(C + j * 3 + 2)
        a = tl.load(I + j * 9) * dx + tl.load(I + j * 9 + 1) * dy + tl.load(I + j * 9 + 2) * dz
        b = tl.load(I + j * 9 + 3) * dx + tl.load(I + j * 9 + 4) * dy + tl.load(I + j * 9 + 5) * dz
        c = tl.load(I + j * 9 + 6) * dx + tl.load(I + j * 9 + 7) * dy + tl.load(I + j * 9 + 8) * dz
        if P == 0:
            hit = (a * a + b * b <= 1.0) & (tl.abs(c) <= 1.0)
        elif P == 2:
            hit = a * a + b * b + c * c <= 1.0
        else:
            hit = a * a + b * b + c * c * (c * c) <= 1.0
        count += hit.to(tl.int32)
    tl.store(OUT + body * S + point, count, mask)

def multiplicity(center, T, edge, points):
    n = len(center)
    S = points.numel() // 3
    ptr, neighbors = csr(edge, n)
    inv = torch.linalg.inv(T).contiguous()
    out = torch.empty((n, S), device=center.device, dtype=torch.int32)
    return (out, ptr, neighbors, inv)

def volume_overlap(center, T, valid, edge, method, reps=4, samples=512, return_counts=False):
    u = unit_points(method, reps, samples, str(center.device))
    count, ptr, neigh, inv = multiplicity(center, T, edge, u)
    _counts[len(center), triton.cdiv(reps * samples, 256)](center.contiguous(), T.contiguous(), inv, ptr, neigh, u, count, reps * samples, 0 if method == 'cylinder' else 2 if method == 'ellipsoid' else 4, 256, num_warps=4, enable_fp_fusion=False)
    vol = FACTORS[method] * torch.linalg.det(T).abs() * valid
    weight = vol / vol.sum().clamp_min(1e-30)
    value = 1 - 1 / count.double().reshape(len(center), reps, samples)
    result = (value.mean(-1) * weight[:, None]).sum(0)
    half = (value[:, :, :samples // 2].mean(-1) * weight[:, None]).sum(0)
    x = dict(volume_overlap=float(result.mean()), volume_overlap_replicates=result.cpu().tolist(), volume_overlap_half_replicates=half.cpu().tolist(), total_cell_volume=float(vol.sum()), pairs=len(edge[0]))
    return (x, count) if return_counts else x
