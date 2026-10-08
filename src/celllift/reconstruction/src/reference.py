from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from celllift.runtime import torch
from .geometry import A, B, coefficients

@torch.no_grad()
def weighted_quantile(x, w, prob):
    order = x.argsort(dim=-1, stable=True)
    x = x.gather(1, order)
    w = w.gather(1, order)
    c = (w.cumsum(1) - 0.5 * w) / w.sum(1, keepdim=True).clamp_min(1e-100)
    c = torch.cat((c.new_zeros(len(c), 1), c, c.new_ones(len(c), 1)), 1)
    x = torch.cat((x[:, :1], x, x[:, -1:]), 1)
    p = prob[None, :].expand(len(x), -1).contiguous()
    idx = torch.searchsorted(c.contiguous(), p).clamp(1, c.shape[1] - 1)
    l = idx - 1
    f = (p - c.gather(1, l)) / (c.gather(1, idx) - c.gather(1, l)).clamp_min(1e-100)
    return x.gather(1, l) + f * (x.gather(1, idx) - x.gather(1, l))

@torch.no_grad()
def reference(root, steps=96, knots=129, k=9):
    outputs = []
    for part in root.split(64):
        cf = coefficients(part)
        lo = (cf.maximum / (9 * B)).sqrt().clamp_min(1e-06)
        Rmax = (9 * B * cf.h0.square() / cf.maximum).clamp_min(1).pow(1 / 3)
        rmax = (1 - 1 / Rmax).clamp_min(0).pow(0.25)
        hi = torch.minimum(cf.h0, (9 * cf.minimum * Rmax / B).sqrt()).maximum(lo)
        t = (torch.arange(steps, device=part.device, dtype=torch.float64) + 0.5) / steps
        dh = (hi.log() - lo.log()) / steps
        h = (lo.log()[:, None] + t[None, :] * (hi.log() - lo.log())[:, None]).exp()

        def weight(R, hh):
            ev = torch.stack((cf.minimum[:, None] * R, cf.maximum[:, None] * R, B * hh.square()), -1)
            good = (ev.amax(-1) <= 9 * ev.amin(-1) * (1 + 1e-12)) & (R * hh <= cf.h0[:, None] * (1 + 1e-12))
            axes = 0.5 * ev.log()
            prior = 0.5 * ((axes - axes.mean(-1, keepdim=True)) / math.log(2)).square().mean(-1)
            return good * torch.exp(-prior)
        wf = weight(torch.ones_like(h), h) * dh[:, None]
        fullmass = 5 * wf.sum(1)
        rr = (rmax[:, None] * t[None, :])[:, :, None].expand(-1, -1, steps).reshape(len(part), -1)
        hh = h[:, None, :].expand(-1, steps, -1).reshape(len(part), -1)
        R = 1 / (1 - rr.pow(4))
        wc = weight(R, hh) * hh * (rmax / steps * dh)[:, None]
        zc = -rr * hh
        xic = -1 + torch.log1p(-(rr / rmax.clamp_min(1e-100)[:, None]).clamp_max(1 - 1e-15))
        zf = (5 * t)[None, :].expand(len(part), -1)
        xf = 2 * zf / 5 - 1
        w = torch.cat((wc, fullmass[:, None].expand(-1, steps) / steps, wc), 1)
        z = torch.cat((zc, zf, 5 - zc), 1)
        x = torch.cat((xic, xf, -xic), 1)
        bad = (w.sum(1) == 0) | ~cf.valid
        w[bad] = 0
        w[bad, wc.shape[1]:wc.shape[1] + steps] = 1
        zq = weighted_quantile(z, w, torch.linspace(0, 1, knots, device=part.device, dtype=torch.float64))
        xq = weighted_quantile(x, w, (torch.arange(k, device=part.device, dtype=torch.float64) + 0.5) / k)
        zq = (zq + 5 - zq.flip(1)) / 2
        zq[:, knots // 2] = 2.5
        xq = (xq - xq.flip(1)) / 2
        outputs.append((zq.float(), xq.float(), (fullmass / w.sum(1).clamp_min(1e-100)).float()))
    return tuple((torch.cat([o[j] for o in outputs]) for j in range(3)))

@torch.no_grad()
def phase(centers, knots):
    idx = torch.searchsorted(knots.contiguous(), centers.contiguous()).clamp(1, knots.shape[1] - 1)
    lo = idx - 1
    f = (centers - knots.gather(1, lo)) / (knots.gather(1, idx) - knots.gather(1, lo)).clamp_min(1e-08)
    return ((lo + f) / (knots.shape[1] - 1)).clamp(0, 1)
