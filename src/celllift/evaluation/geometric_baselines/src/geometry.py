from celllift.runtime import resource_path as _public_resource
import math
import torch
CONSTANTS = {'cylinder': (2 * math.pi, 0.25, 1 / 3), 'ellipsoid': (4 * math.pi / 3, 0.2, 0.2), 'p4': (8 * math.pi / 5, 2 / 9, 5 / 21)}

def coefficients(root, method):
    v, a, b = CONSTANTS[method]
    root = root.double()
    Q = root @ root.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(Q)
    det = torch.linalg.det(root).abs()
    lo = (a * ev[:, 1] / (9 * b)).sqrt()
    hi = torch.minimum((9 * a * ev[:, 0] / b).sqrt(), 800 / (v * det))
    valid = (ev[:, 0] > 0) & (ev[:, 1] <= 9 * ev[:, 0]) & (lo <= hi)
    safe = lambda x: torch.where(valid, x, torch.ones_like(x))
    return dict(Q=Q, det=det, ev=ev, lo=safe(lo), hi=safe(hi), valid=valid)

def decode(cf, parameters, method):
    vp, a, b = CONSTANTS[method]
    k, l, g = [torch.as_tensor(x, device=cf['det'].device, dtype=torch.float64) for x in parameters]
    h = (k * cf['det'].sqrt()).maximum(cf['lo']).minimum(cf['hi'])
    lmax = torch.minimum((4000 / (vp * cf['det'] * h)).sqrt(), (4000 * 6 * math.sqrt(b) / (vp * cf['det'] * (a * cf['ev'][:, 1]).sqrt())).pow(1 / 3))
    lam = l.clamp_min(1).minimum(lmax.clamp_min(1))
    low = torch.maximum(h, lam * (a * cf['ev'][:, 1] / (36 * b)).sqrt())
    high = torch.minimum(lam * (36 * a * cf['ev'][:, 0] / b).sqrt(), 4000 / (vp * cf['det'] * lam.square()))
    hc = (g.clamp_min(1) * h).maximum(low).minimum(high)
    return dict(h=h, lam=lam, hc=hc, valid=cf['valid'])

def stats(cf, geo, method, kind):
    vp, a, b = CONSTANTS[method]
    h = geo['h'] if kind == 'nucleus' else geo['hc']
    lam = torch.ones_like(h) if kind == 'nucleus' else geo['lam']
    ev = torch.cat((a * cf['ev'] * lam[:, None].square(), b * h[:, None].square()), 1)
    axes = ev.sqrt()
    logs = axes.log()
    scale = math.log(2 if kind == 'nucleus' else 3)
    return dict(volume=vp * cf['det'] * lam.square() * h, ratio=(ev.max(1).values / ev.min(1).values).sqrt(), prior=0.5 * ((logs - logs.mean(1, keepdim=True)) / scale).square().mean(1))

def slab_factor(h, plane, method, extend=False):
    d = torch.where(plane == 1, torch.zeros_like(h), torch.full_like(h, 2.5))
    exists = h > d
    if method == 'cylinder':
        factor = torch.ones_like(h)
    else:
        p = 2 if method == 'ellipsoid' else 4
        ratio = d / h
        if extend:
            ratio = ratio.clamp_max(1 - 1e-06)
        factor = (1 - ratio.pow(p)).clamp_min(0).sqrt()
    factor = torch.where(exists | extend, factor, torch.zeros_like(factor))
    return (factor, exists, (d - h).clamp_min(0))

def contact(Q1, Q2, delta, iterations=48):
    Q1 = Q1.double()
    Q2 = Q2.double()
    delta = delta.double()
    lo = delta.new_zeros(len(delta))
    hi = torch.ones_like(lo)

    def f(t):
        Q = Q1 * (1 - t[:, None, None]) + Q2 * t[:, None, None]
        x, y = delta.unbind(1)
        aa = Q[:, 0, 0]
        bb = Q[:, 0, 1]
        dd = Q[:, 1, 1]
        return t * (1 - t) * (dd * x * x - 2 * bb * x * y + aa * y * y) / (aa * dd - bb * bb)
    for _ in range(iterations):
        l = (2 * lo + hi) / 3
        r = (lo + 2 * hi) / 3
        less = f(l) < f(r)
        lo = torch.where(less, l, lo)
        hi = torch.where(less, hi, r)
    return f((lo + hi) / 2).clamp_min(0).sqrt()

def contact_pairs(center, Q, valid, chunk=256):
    ext = Q.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    pairs = []
    n = len(center)
    for start in range(0, n, chunk):
        ids = torch.arange(start, min(start + chunk, n), device=center.device)
        other = torch.arange(n, device=center.device)
        hit = ((center[ids, None] - center[None, :]).abs() <= ext[ids, None] + ext[None, :] + 1e-07).all(-1)
        hit &= valid[ids, None] & valid[None, :] & (ids[:, None] < other[None, :])
        a, b = torch.where(hit)
        if len(a):
            pairs.append(torch.stack((ids[a], b)))
    return torch.cat(pairs, 1) if pairs else torch.empty((2, 0), device=center.device, dtype=torch.long)
