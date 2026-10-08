from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import math
from celllift.runtime import torch
from .frozen.collision import kp_support
from .frozen.geometry import projection_support, build_sheared_geometry
from .frozen.compatibility_kkt import compatibility_projection_support_kkt

@dataclass
class Anchor:
    center: torch.Tensor
    root: torch.Tensor
    height: float = 5.0

@dataclass
class Body:
    center: torch.Tensor
    transform: torch.Tensor
    p: int
    layout: str

    def select(self, index):
        return Body(self.center[index], self.transform[index], self.p, self.layout[index] if isinstance(self.layout, torch.Tensor) else self.layout)

def constants(p):
    return (2 * math.pi * p / (p + 1), (p / (2 * (2 * p + 1)),) * 2 + ((p + 1) / (3 * (p + 3)),))

def body_volume(body):
    return constants(body.p)[0] * torch.linalg.det(body.transform).abs()

def rms_axes(body):
    _, cp = constants(body.p)
    cov = body.transform * body.transform.new_tensor(cp) @ body.transform.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(cov)
    return ev.clamp_min(torch.finfo(ev.dtype).tiny).sqrt()

def axial_halfheight(body):
    return kp_support(body.transform[..., 2, :], body.p)

def physical_stats(body):
    vp, cp = constants(body.p)
    cov = body.transform * body.transform.new_tensor(cp) @ body.transform.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(cov)
    axes = ev.clamp_min(torch.finfo(ev.dtype).tiny).sqrt()
    return dict(volume=vp * torch.linalg.det(body.transform).abs(), covariance=cov, axes=axes, aspect=axes[..., -1] / axes[..., 0], axial_halfheight=kp_support(body.transform[..., 2, :], body.p))

def disk(raw):
    return raw / (1 + raw.square().sum(-1, keepdim=True)).sqrt()

def decode_geometry(anchor, branch, parameters, p=4, volume_cap=800.0):
    m, L, H = (anchor.center, anchor.root, anchor.height)
    raw = parameters
    vp, _ = constants(p)
    if branch == 'moment':
        g = build_sheared_geometry(m, L, raw, p, moment_directions=2048, support_steps=32)
        return Body(g.center, g.transform, p, 'upper')
    T = raw.new_zeros((len(raw), 3, 3))
    if branch == 'full':
        phase = raw[:, 0].tanh()
        c = H / 2 * (1 + phase)
        k = H / 2 * (1 - phase.abs())[:, None] * disk(raw[:, 1:3])
        h = volume_cap / (vp * torch.linalg.det(L).abs()) * raw[:, 3].sigmoid()
        T[:, :2, :2] = L
        T[:, 2, :2] = k
        T[:, 2, 2] = h
        return Body(torch.cat((m, c[:, None]), -1), T, p, 'lower')
    if branch not in ('lower', 'upper'):
        raise ValueError(branch)
    r = raw[:, 0].sigmoid()
    R = 1 / (1 - r.pow(p))
    u = (p / 2 * r.pow(p - 1) * R)[:, None] * disk(raw[:, 1:3])
    h = volume_cap / (vp * torch.linalg.det(L).abs() * R) * raw[:, 3].sigmoid()
    sign = 1.0 if branch == 'lower' else -1.0
    boundary = 0.0 if branch == 'lower' else H
    Lu = torch.einsum('nij,nj->ni', L, u)
    T[:, :2, :2] = L * R.sqrt()[:, None, None]
    T[:, :2, 2] = -Lu
    T[:, 2, 2] = sign * h
    center = torch.cat((m + Lu * r[:, None], (boundary - sign * r * h)[:, None]), -1)
    return Body(center, T, p, 'upper')

def conditional_cell(nucleus, raw, volume_cap=4000.0):
    vn = physical_stats(nucleus)['volume']
    budget = (volume_cap / vn).log().clamp_min(0)
    xy = 0.5 * budget * raw[:, 0].sigmoid()
    z = (budget - 2 * xy) * raw[:, 1].sigmoid()
    D = torch.stack((xy.exp(), xy.exp(), z.exp()), -1)
    w = raw[:, 2:]
    denom = 1 + w[:, :2].square().sum(-1) + w[:, 2].abs().pow(nucleus.p)
    d = torch.cat((w[:, :2] / denom.sqrt()[:, None], (w[:, 2] / denom.pow(1 / nucleus.p))[:, None]), -1)
    d = d * (D.amin(-1) - 1)[:, None]
    center = nucleus.center + torch.einsum('nij,nj->ni', nucleus.transform, d)
    return Body(center, nucleus.transform * D[:, None, :], nucleus.p, nucleus.layout)

def project_slab(body, slab, directions, steps=40):
    T, c = (body.transform, body.center.to(body.transform.dtype))
    directions = directions.to(T.dtype)
    slab = slab.to(T.dtype)
    if slab.ndim == 1:
        slab = slab[None].expand(len(c), -1)
    if isinstance(body.layout, torch.Tensor):
        out = c.new_empty((len(c), len(directions)))
        exists = torch.zeros(len(c), device=c.device, dtype=torch.bool)
        for code, layout in enumerate(('lower', 'upper')):
            ids = torch.where(body.layout == code)[0]
            if len(ids):
                values, ok = project_slab(Body(c[ids], T[ids], body.p, layout), slab[ids], directions, steps)
                out = out.index_copy(0, ids, values)
                exists[ids] = ok
        return (out, exists)
    if body.layout == 'lower':
        r = compatibility_projection_support_kkt(T[:, :2, :2], c[:, :2], c[:, 2], T[:, 2, :2], T[:, 2, 2].abs(), slab, directions, body.p, bisection_steps=steps, maximization_steps=steps, interior_tolerance=0.0)
        return (r.support, r.has_interior)
    out, exists, _ = projection_support(T[:, :2, :2], c[:, :2], c[:, 2], T[:, :2, 2] / T[:, 2, 2, None], T[:, 2, 2].abs().pow(-body.p), slab, directions, body.p, steps=steps)
    return (torch.where(exists[:, None], out, torch.full_like(out, float('nan'))), exists)

def positive_projection(body, slab, directions):
    h = axial_halfheight(body)
    c = body.center[:, 2]
    below = (slab[:, 0] - c - h).clamp_min(0)
    above = (c - h - slab[:, 1]).clamp_min(0)
    gap = below + above
    empty = (slab[:, 0] >= c + h) | (slab[:, 1] <= c - h)
    inset = h * 0.001
    shift = torch.where(slab[:, 0] >= c + h, -below - inset, torch.where(slab[:, 1] <= c - h, above + inset, torch.zeros_like(h)))
    translation = torch.stack((torch.zeros_like(shift), torch.zeros_like(shift), -shift), -1)
    transported = Body(body.center + translation, body.transform, body.p, body.layout)
    support, exists = project_slab(transported, slab, directions)
    boundary = torch.where(~exists)[0]
    if len(boundary):
        selected = transported.select(boundary)
        delta = torch.where(selected.center[:, 2] > slab[boundary, 1], -inset[boundary], inset[boundary])
        displacement = torch.stack((torch.zeros_like(delta), torch.zeros_like(delta), delta), -1)
        extended = Body(selected.center + displacement, selected.transform, selected.p, selected.layout)
        values, _ = project_slab(extended, slab[boundary], directions)
        support = support.index_copy(0, boundary, values)
    return (support, gap.to(support.dtype), ~empty)

def ellipsoid_reference(body, slab, directions):
    if body.p != 2:
        raise ValueError('p2 only')
    Q = body.transform @ body.transform.transpose(-1, -2)
    hz = Q[:, 2, 2].sqrt()
    slope = Q[:, :2, 2] / Q[:, 2, 2, None]
    S = Q[:, :2, :2] - Q[:, :2, 2, None] * Q[:, None, 2, :2] / Q[:, 2, 2, None, None]
    a = torch.einsum('ki,nij,kj->nk', directions, S, directions).clamp_min(0).sqrt()
    b = slope @ directions.T * hz[:, None]
    tau = b / (a.square() + b.square()).sqrt()
    lower = (slab[:, 0] - body.center[:, 2]) / hz
    upper = (slab[:, 1] - body.center[:, 2]) / hz
    exists = (lower < 1) & (upper > -1)
    tau = tau.maximum(lower[:, None]).minimum(upper[:, None]).clamp(-1, 1)
    out = body.center[:, :2] @ directions.T + b * tau + a * (1 - tau.square()).clamp_min(0).sqrt()
    return (torch.where(exists[:, None], out, torch.full_like(out, float('nan'))), exists)
