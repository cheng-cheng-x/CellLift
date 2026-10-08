from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from .collision import constraints, kp_support

@triton.jit
def _value(x, y, z, alpha, cx, cy, cz, r0, r1, r2, r3, r4, r5, r6, r7, r8, P: tl.constexpr):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    w0 = (r0 * dx + r1 * dy + r2 * dz) / alpha
    w1 = (r3 * dx + r4 * dy + r5 * dz) / alpha
    w2 = (r6 * dx + r7 * dy + r8 * dz) / alpha
    zpower = w2 * w2
    for _ in tl.static_range(1, 1 if P == 2 else 2 if P == 4 else 3):
        zpower = zpower * zpower
    return alpha * (w0 * w0 + w1 * w1 + zpower - 1.0)

@triton.jit
def _terms(x, y, z, alpha, cx, cy, cz, r0, r1, r2, r3, r4, r5, r6, r7, r8, mu, P: tl.constexpr):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    w0 = (r0 * dx + r1 * dy + r2 * dz) / alpha
    w1 = (r3 * dx + r4 * dy + r5 * dz) / alpha
    w2 = (r6 * dx + r7 * dy + r8 * dz) / alpha
    z2 = w2 * w2
    if P == 2:
        zp = z2
        zm1 = w2
        zm2 = tl.full(w2.shape, 1.0, tl.float64)
    elif P == 4:
        zp = z2 * z2
        zm1 = z2 * w2
        zm2 = z2
    else:
        z4 = z2 * z2
        zp = z4 * z4
        zm1 = z4 * z2 * w2
        zm2 = z4 * z2
    g = w0 * w0 + w1 * w1 + zp
    f = alpha * (g - 1.0)
    j0 = 2.0 * w0 * r0 + 2.0 * w1 * r3 + P * zm1 * r6
    j1 = 2.0 * w0 * r1 + 2.0 * w1 * r4 + P * zm1 * r7
    j2 = 2.0 * w0 * r2 + 2.0 * w1 * r5 + P * zm1 * r8
    j3 = g - 1.0 - (2.0 * w0 * w0 + 2.0 * w1 * w1 + P * zm1 * w2)
    multiplier = -mu / f
    normal = mu / (f * f)
    a = 2.0 * multiplier / alpha
    b = P * (P - 1) * zm2 * multiplier / alpha
    h00 = a * (r0 * r0 + r3 * r3) + b * r6 * r6 + normal * j0 * j0
    h01 = a * (r0 * r1 + r3 * r4) + b * r6 * r7 + normal * j0 * j1
    h02 = a * (r0 * r2 + r3 * r5) + b * r6 * r8 + normal * j0 * j2
    h03 = -a * (r0 * w0 + r3 * w1) - b * r6 * w2 + normal * j0 * j3
    h11 = a * (r1 * r1 + r4 * r4) + b * r7 * r7 + normal * j1 * j1
    h12 = a * (r1 * r2 + r4 * r5) + b * r7 * r8 + normal * j1 * j2
    h13 = -a * (r1 * w0 + r4 * w1) - b * r7 * w2 + normal * j1 * j3
    h22 = a * (r2 * r2 + r5 * r5) + b * r8 * r8 + normal * j2 * j2
    h23 = -a * (r2 * w0 + r5 * w1) - b * r8 * w2 + normal * j2 * j3
    h33 = a * (w0 * w0 + w1 * w1) + b * w2 * w2 + normal * j3 * j3
    return (f, multiplier * j0, multiplier * j1, multiplier * j2, multiplier * j3, h00, h01, h02, h03, h11, h12, h13, h22, h23, h33)

@triton.jit
def _newton(g0, g1, g2, g3, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9):
    d0 = lib.sqrt(tl.maximum(h0, 1e-12))
    d1 = lib.sqrt(tl.maximum(h4, 1e-12))
    d2 = lib.sqrt(tl.maximum(h7, 1e-12))
    d3 = lib.sqrt(tl.maximum(h9, 1e-12))
    eps = 7.105427357601002e-15
    a00 = h0 / d0 / d0 + eps
    a01 = h1 / d0 / d1
    a02 = h2 / d0 / d2
    a03 = h3 / d0 / d3
    a11 = h4 / d1 / d1 + eps
    a12 = h5 / d1 / d2
    a13 = h6 / d1 / d3
    a22 = h7 / d2 / d2 + eps
    a23 = h8 / d2 / d3
    a33 = h9 / d3 / d3 + eps
    l10 = a01 / a00
    l20 = a02 / a00
    l30 = a03 / a00
    q1 = a11 - l10 * l10 * a00
    l21 = (a12 - l20 * l10 * a00) / q1
    l31 = (a13 - l30 * l10 * a00) / q1
    q2 = a22 - l20 * l20 * a00 - l21 * l21 * q1
    l32 = (a23 - l30 * l20 * a00 - l31 * l21 * q1) / q2
    q3 = a33 - l30 * l30 * a00 - l31 * l31 * q1 - l32 * l32 * q2
    y0 = -g0 / d0
    y1 = -g1 / d1 - l10 * y0
    y2 = -g2 / d2 - l20 * y0 - l21 * y1
    y3 = -g3 / d3 - l30 * y0 - l31 * y1 - l32 * y2
    s3 = y3 / q3
    s2 = y2 / q2 - l32 * s3
    s1 = y1 / q1 - l21 * s2 - l31 * s3
    s0 = y0 / a00 - l10 * s1 - l20 * s2 - l30 * s3
    return (s0 / d0, s1 / d1, s2 / d2, s3 / d3)

@triton.jit
def _iterate(C, I, X, SIZE, P: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < SIZE
    cx0 = tl.load(C + row * 6, mask, 0.0)
    cy0 = tl.load(C + row * 6 + 1, mask, 0.0)
    cz0 = tl.load(C + row * 6 + 2, mask, 0.0)
    cx1 = tl.load(C + row * 6 + 3, mask, 0.0)
    cy1 = tl.load(C + row * 6 + 4, mask, 0.0)
    cz1 = tl.load(C + row * 6 + 5, mask, 0.0)
    r0 = tl.load(I + row * 18 + 0, mask, 1.0)
    r1 = tl.load(I + row * 18 + 1, mask, 0.0)
    r2 = tl.load(I + row * 18 + 2, mask, 0.0)
    r3 = tl.load(I + row * 18 + 3, mask, 0.0)
    r4 = tl.load(I + row * 18 + 4, mask, 1.0)
    r5 = tl.load(I + row * 18 + 5, mask, 0.0)
    r6 = tl.load(I + row * 18 + 6, mask, 0.0)
    r7 = tl.load(I + row * 18 + 7, mask, 0.0)
    r8 = tl.load(I + row * 18 + 8, mask, 1.0)
    s0 = tl.load(I + row * 18 + 9, mask, 1.0)
    s1 = tl.load(I + row * 18 + 10, mask, 0.0)
    s2 = tl.load(I + row * 18 + 11, mask, 0.0)
    s3 = tl.load(I + row * 18 + 12, mask, 0.0)
    s4 = tl.load(I + row * 18 + 13, mask, 1.0)
    s5 = tl.load(I + row * 18 + 14, mask, 0.0)
    s6 = tl.load(I + row * 18 + 15, mask, 0.0)
    s7 = tl.load(I + row * 18 + 16, mask, 0.0)
    s8 = tl.load(I + row * 18 + 17, mask, 1.0)
    x = tl.load(X + row * 4, mask, 0.0)
    y = tl.load(X + row * 4 + 1, mask, 0.0)
    z = tl.load(X + row * 4 + 2, mask, 0.0)
    alpha = tl.load(X + row * 4 + 3, mask, 1.0)
    mu = tl.full((BLOCK,), 1.0, tl.float64)
    for _barrier in range(10):
        done = ~mask
        iteration = 0
        while (iteration < 32) & (tl.sum((~done).to(tl.int32), 0) > 0):
            af, ag0, ag1, ag2, ag3, ah0, ah1, ah2, ah3, ah4, ah5, ah6, ah7, ah8, ah9 = _terms(x, y, z, alpha, cx0, cy0, cz0, r0, r1, r2, r3, r4, r5, r6, r7, r8, mu, P)
            bf, bg0, bg1, bg2, bg3, bh0, bh1, bh2, bh3, bh4, bh5, bh6, bh7, bh8, bh9 = _terms(x, y, z, alpha, cx1, cy1, cz1, s0, s1, s2, s3, s4, s5, s6, s7, s8, mu, P)
            g0 = ag0 + bg0
            g1 = ag1 + bg1
            g2 = ag2 + bg2
            g3 = 1.0 + ag3 + bg3
            done = done | (lib.sqrt(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3) < 1e-09)
            sx, sy, sz, sa = _newton(g0, g1, g2, g3, ah0 + bh0, ah1 + bh1, ah2 + bh2, ah3 + bh3, ah4 + bh4, ah5 + bh5, ah6 + bh6, ah7 + bh7, ah8 + bh8, ah9 + bh9)
            directional = g0 * sx + g1 * sy + g2 * sz + g3 * sa
            old = alpha - mu * (lib.log(-af) + lib.log(-bf))
            rate = tl.full((BLOCK,), 1.0, tl.float64)
            accepted = done
            nx = x
            ny = y
            nz = z
            na = alpha
            line = 0
            while (line < 40) & (tl.sum((~accepted).to(tl.int32), 0) > 0):
                tx = x + rate * sx
                ty = y + rate * sy
                tz = z + rate * sz
                ta = alpha + rate * sa
                positive = ta > 0
                tx = tl.where(positive, tx, x)
                ty = tl.where(positive, ty, y)
                tz = tl.where(positive, tz, z)
                ta = tl.where(positive, ta, alpha)
                f0 = _value(tx, ty, tz, ta, cx0, cy0, cz0, r0, r1, r2, r3, r4, r5, r6, r7, r8, P)
                f1 = _value(tx, ty, tz, ta, cx1, cy1, cz1, s0, s1, s2, s3, s4, s5, s6, s7, s8, P)
                value = ta - mu * (lib.log(tl.maximum(-f0, 2.2250738585072014e-308)) + lib.log(tl.maximum(-f1, 2.2250738585072014e-308)))
                ok = positive & (f0 < 0) & (f1 < 0) & (value <= old + 0.0001 * rate * directional)
                take = ok & ~accepted
                nx = tl.where(take, tx, nx)
                ny = tl.where(take, ty, ny)
                nz = tl.where(take, tz, nz)
                na = tl.where(take, ta, na)
                accepted = accepted | ok
                rate = tl.where(accepted, rate, rate * 0.5)
                line += 1
            x = nx
            y = ny
            z = nz
            alpha = na
            iteration += 1
        mu = mu * 0.1
    tl.store(X + row * 4, x, mask)
    tl.store(X + row * 4 + 1, y, mask)
    tl.store(X + row * 4 + 2, z, mask)
    tl.store(X + row * 4 + 3, alpha, mask)

@torch.no_grad()
def contact_scale(center_i, transform_i, center_j, transform_j, p, *, check=True):
    if p not in (2, 4, 8):
        raise ValueError('fused contact implements fixed p2/p4/p8 only')
    delta = center_j - center_i
    ids = torch.where(delta.square().sum(-1) > 0)[0]
    alpha = delta.new_zeros(len(delta))
    lower = alpha.clone()
    gaps = alpha.clone()
    residuals = alpha.clone()
    if not len(ids):
        return (alpha, dict(lower=lower, gap=gaps, stationarity=residuals))
    scale = 0.5 * (transform_i[ids].norm(dim=(-1, -2)) + transform_j[ids].norm(dim=(-1, -2)))
    distance = delta[ids].norm(dim=-1)
    alpha_scale = distance / scale
    centers = torch.stack((torch.zeros_like(delta[ids]), delta[ids] / distance[:, None]), 1).double().contiguous()
    transforms = (torch.stack((transform_i[ids], transform_j[ids]), 1) / scale[:, None, None, None]).double()
    inverses = torch.linalg.inv(transforms).contiguous()
    x = centers.mean(1)
    u = torch.einsum('nbij,nbj->nbi', inverses, x[:, None] - centers)
    initial = 2 * (u[..., :2].norm(dim=-1) + u[..., 2].abs()).amax(-1) + 1
    primal = torch.cat((x, initial[:, None]), -1).contiguous()
    _iterate[triton.cdiv(len(ids), 32),](centers, inverses, primal, len(ids), p, 32, num_warps=1, enable_fp_fusion=False)
    f, jac, _ = constraints(primal, centers, inverses, p, derivatives=True)
    gram = jac @ jac.transpose(-1, -2)
    multiplier = torch.linalg.solve(gram, -jac[..., 3])
    normal = multiplier[:, 0, None] * jac[:, 0, :3]
    transformed = torch.einsum('nbji,nj->nbi', transforms, normal)
    bound = ((centers[:, 1] - centers[:, 0]) * normal).sum(-1) / kp_support(transformed, p).sum(-1)
    gap = primal[:, 3] - bound
    residual = (primal.new_tensor((0.0, 0.0, 0.0, 1.0)) + (multiplier[..., None] * jac).sum(1)).norm(dim=-1)
    tolerance = 2e-05 if center_i.dtype == torch.float32 else 2e-07
    if check and ((~torch.isfinite(primal)).any() or (f > 1e-10).any() or (gap > tolerance * (1 + primal[:, 3])).any() or (gap < -tolerance).any() or (multiplier < 0).any() or (residual > 0.002).any()):
        raise RuntimeError(f'fused contact certificate failed: gap={gap.max().item():.6g}, residual={residual.max().item():.6g}')
    alpha[ids] = (primal[:, 3] * alpha_scale).to(alpha.dtype)
    lower[ids] = (bound * alpha_scale).to(lower.dtype)
    gaps[ids] = (gap * alpha_scale).to(gaps.dtype)
    residuals[ids] = residual.to(residuals.dtype)
    return (alpha, dict(lower=lower, gap=gaps, stationarity=residuals, ids=ids, scale=scale, distance=distance, primal=primal, multiplier=multiplier))
