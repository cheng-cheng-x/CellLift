from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from .reference import reference
from .geometry import A, B, VP, coefficients, nucleus_geometry, Anchor, CellInterval, body_volume

def logit_fraction(x):
    return torch.logit(x.clamp(0.0001, 1 - 0.0001))

@torch.no_grad()
def initialize(root, cf=None):
    cf = coefficients(root) if cf is None else cf
    _, query, _ = reference(root)
    n, k = query.shape
    ids = torch.arange(n, device=root.device).repeat_interleave(k)
    c = cf.select(ids)
    u = query.flatten().double()
    rmax = (1 - (9 * B * c.h0.square() / c.maximum).clamp_min(1).pow(-1 / 3)).pow(0.25)
    r = rmax * -torch.expm1(-(u.abs() - 1).clamp_min(0))
    R = 1 / (1 - r.pow(4))
    hlo = (R * c.maximum / (9 * B)).sqrt()
    hhi = torch.minimum(c.h0 / R, (9 * R * c.minimum / B).sqrt())
    hc = (R * (c.minimum * c.maximum).sqrt() / B).sqrt()
    hc = hc.maximum(hlo + 0.0001 * (hhi - hlo)).minimum(hhi - 0.0001 * (hhi - hlo))
    tlo = R * c.maximum / 9
    thi = torch.minimum(R * c.minimum, B * (c.h0 / R).square())
    t = (torch.maximum(tlo, B * hc.square() / 9) + torch.minimum(thi, B * hc.square())) / 2
    raw = torch.zeros(n * k, 10, device=root.device, dtype=torch.float64)
    raw[:, 0] = u
    raw[:, 4] = logit_fraction((t - tlo) / (thi - tlo).clamp_min(1e-30))
    low = t / B
    high = torch.minimum(9 * t / B, (c.h0 / R).square())
    raw[:, 3] = logit_fraction((hc.square() - low) / (high - low).clamp_min(1e-30))
    raw[~c.valid, 1:] = 0
    body, _ = nucleus_geometry(Anchor(root.new_zeros((n * k, 2)), root[ids]), raw[:, :5], coeff=c)
    interval = CellInterval.apply(body.transform)
    raw[:, 5] = logit_fraction(-interval[:, 0] / (interval[:, 1] - interval[:, 0]).clamp_min(1e-30))
    raw[:, 6] = 0
    raw[~c.valid, 1:] = 0
    return dict(query=query, raw=raw.reshape(n, k, 10).float())
