from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import math
from celllift.runtime import torch
from .projection_geometry import Anchor, Body, physical_stats, project_slab, positive_projection, constants, body_volume, rms_axes
A = 2 / 9
B = 5 / 21
VP = 8 * math.pi / 5

@dataclass
class Coefficients:
    root: torch.Tensor
    pmat: torch.Tensor
    amat: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    h0: torch.Tensor
    valid: torch.Tensor
    pvec: torch.Tensor | None = None
    avec: torch.Tensor | None = None

    def __post_init__(self):
        if self.pvec is None:
            self.pvec = torch.linalg.eigh(self.pmat).eigenvectors
        if self.avec is None:
            self.avec = torch.linalg.eigh(self.amat).eigenvectors

    def select(self, ix):
        return Coefficients(**{k: getattr(self, k)[ix] for k in self.__dataclass_fields__})

@torch.no_grad()
def coefficients(root):
    L = root.double()
    pm = A * L.transpose(-1, -2) @ L
    am = A * L @ L.transpose(-1, -2)
    ev = torch.linalg.eigvalsh(am)
    h0 = 800 / (VP * torch.linalg.det(L).abs())
    valid = (ev[:, 0] > 0) & (ev[:, 1] / 9 <= torch.minimum(ev[:, 0], B * h0.square()))
    return Coefficients(L, pm, am, ev[:, 0], ev[:, 1], h0, valid)

def open_lerp(low, high, raw):
    value = low + (high - low) * raw.sigmoid()
    with torch.no_grad():
        rounded = value.maximum(torch.nextafter(low, high)).minimum(torch.nextafter(high, low))
    return value + (rounded - value).detach()

def nucleus_geometry(anchor, raw, full_only=False, coeff=None):
    cf = coefficients(anchor.root) if coeff is None else coeff
    n = len(raw)
    dtype = raw.dtype
    rraw = raw.double()
    center = torch.cat((anchor.center.double(), torch.full_like(rraw[:, :1], 2.5)), -1)
    transform = torch.eye(3, device=raw.device, dtype=torch.float64)[None].repeat(n, 1, 1)
    layout = torch.zeros(n, device=raw.device, dtype=torch.long)
    ix = torch.where(cf.valid)[0]
    if not len(ix):
        return (Body(center.to(dtype), transform.to(dtype), 4, layout), cf.valid)
    x = rraw[ix]
    cc = cf.select(ix)
    L = cc.root
    loc = x[:, 0].tanh() if full_only else x[:, 0]
    cap = loc.abs() > 1
    eps = torch.where(loc < 0, 1.0, -1.0).to(x)
    rmax = (-torch.expm1(-torch.log(9 * B * cc.h0.square() / cc.maximum) / 3)).clamp_min(0).pow(0.25)
    r = rmax * -torch.expm1(-(loc.abs() - 1).clamp_min(0))
    R = 1 / (1 - r.pow(4))
    hv = cc.h0 / R
    low = R * cc.maximum / 9
    high = torch.minimum(R * cc.minimum, B * hv.square())
    regular = high > torch.nextafter(low, torch.full_like(low, float('inf')))
    tau = open_lerp(low, high, x[:, 4])
    q = x.new_zeros((len(x), 2))
    h = (low / B).sqrt()
    ids = torch.where(regular)[0]
    if len(ids):
        tp = tau[ids]
        eye = torch.eye(2, device=x.device, dtype=x.dtype)[None]
        iscap = cap[ids]
        factor = torch.where(iscap, torch.ones_like(tp), torch.full_like(tp, A / B))
        basis = torch.where(iscap[:, None, None], cc.avec[ids], cc.pvec[ids])
        eig = torch.stack((cc.minimum[ids], cc.maximum[ids]), -1) * torch.where(iscap, R[ids], torch.ones_like(tp))[:, None]
        minus = factor[:, None] * tp[:, None] / (eig - tp[:, None])
        plus = factor[:, None] * tp[:, None] / (tp[:, None] - eig / 9)
        margin = 2.5 * (1 - loc[ids].abs()).clamp_min(0)
        J = torch.where(iscap[:, None, None], (2 * r[ids].pow(3) * R[ids])[:, None, None] * L[ids], margin[:, None, None] * eye)
        w = x[ids, 1:3]
        jw = (J @ w[..., None]).squeeze(-1)
        spectral = (basis.transpose(-1, -2) @ jw[..., None]).squeeze(-1).square()
        qm = (spectral * minus).sum(-1)
        qp = (spectral * plus).sum(-1)
        volume_gap = (B * hv[ids].square() - tp) / B
        gauge = torch.stack((w.square().sum(-1), (qm + qp) / (8 * tp / B), qm / volume_gap), -1).amax(-1)
        qq = jw / (1 + gauge).sqrt()[:, None]
        hlo = tp / B + qm / (1 + gauge)
        hhi = torch.minimum(9 * tp / B - qp / (1 + gauge), hv[ids].square())
        hh = open_lerp(hlo, hhi, x[ids, 3]).sqrt()
        q = q.index_copy(0, ids, qq)
        h = h.index_copy(0, ids, hh)
    T = x.new_zeros((len(x), 3, 3))
    T[:, :2, :2] = R.sqrt()[:, None, None] * L
    T[:, 2, 2] = h
    T[:, 2, :2] = torch.where(cap[:, None], 0.0, q)
    T[:, :2, 2] = torch.where(cap[:, None], q, 0.0)
    xy = anchor.center[ix].double() - torch.where(cap[:, None], eps[:, None] * r[:, None] * q, 0.0)
    z = torch.where(cap, torch.where(loc < 0, 0.0, 5.0) - eps * r * h, 2.5 * (1 + loc))
    center = center.index_copy(0, ix, torch.cat((xy, z[:, None]), -1))
    transform = transform.index_copy(0, ix, T)
    layout[ix] = cap.long()
    return (Body(center.to(dtype), transform.to(dtype), 4, layout), cf.valid)

def log_aspect(T, logt):
    D = torch.stack((torch.ones_like(logt), torch.ones_like(logt), logt.exp()), -1)
    U = T * D[:, None, :]
    cp = U.new_tensor([A, A, B])
    ev = torch.linalg.eigvalsh(U * cp @ U.transpose(-1, -2))
    return 0.5 * (ev[:, -1].log() - ev[:, 0].log())

class CellInterval(torch.autograd.Function):

    @staticmethod
    def forward(ctx, T):
        u = T.double()
        v = VP * torch.linalg.det(u).abs()
        budget = (4000 / v).log()
        volume_lo = -0.5 * budget
        volume_hi = budget
        hit_lo = log_aspect(u, volume_lo) > math.log(6)
        hit_hi = log_aspect(u, volume_hi) > math.log(6)
        endpoints = torch.cat((volume_lo, volume_hi))
        active = torch.where(torch.cat((hit_lo, hit_hi)))[0]
        if len(active):
            owner = active % len(u)
            w = u[owner]
            xy = w[:, :, :2] * A @ w[:, :, :2].transpose(-1, -2)
            z = w[:, :, 2:] * B @ w[:, :, 2:].transpose(-1, -2)
            outside = endpoints[active].clone()
            inside = torch.zeros_like(outside)
            for _ in range(32):
                mid = (outside + inside) / 2
                ev = torch.linalg.eigvalsh(xy + (2 * mid).exp()[:, None, None] * z)
                bad = 0.5 * (ev[:, -1].log() - ev[:, 0].log()) > math.log(6)
                outside = torch.where(bad, mid, outside)
                inside = torch.where(bad, inside, mid)
            endpoints = endpoints.index_copy(0, active, inside)
        lo, hi = (endpoints[:len(u)], endpoints[len(u):])
        ctx.save_for_backward(u, lo, hi, hit_lo, hit_hi)
        ctx.dtype = T.dtype
        return torch.stack((lo, hi), -1).to(T.dtype)

    @staticmethod
    def backward(ctx, gradient):
        u, lo, hi, hit_lo, hit_hi = ctx.saved_tensors
        with torch.enable_grad():
            T = u.detach().requires_grad_(True)
            budget = (4000 / (VP * torch.linalg.det(T).abs())).log()
            result = torch.zeros_like(T)
            for j, (endpoint, active, analytic) in enumerate(((lo, hit_lo, -0.5 * budget), (hi, hit_hi, budget))):
                e = endpoint.detach().requires_grad_(True)
                value = log_aspect(T, e)
                dt, de = torch.autograd.grad(value.sum(), (T, e), retain_graph=True)
                da = torch.autograd.grad(analytic.sum(), T, retain_graph=True)[0]
                derivative = torch.where(active[:, None, None], -dt / de[:, None, None], da)
                result = result + derivative * gradient[:, j, None, None].double()
        return result.to(ctx.dtype)

def conditional_cell(nucleus, raw):
    interval = CellInterval.apply(nucleus.transform)
    logt = open_lerp(interval[:, 0], interval[:, 1], raw[:, 0])
    v = body_volume(nucleus)
    budget = (4000 / v).log()
    logs = open_lerp((-logt).clamp_min(0), (budget - logt) / 3, raw[:, 1])
    D = torch.stack((logs.exp(), logs.exp(), (logs + logt).exp()), -1)
    w = raw[:, 2:]
    den = 1 + w[:, :2].square().sum(-1) + w[:, 2].pow(4)
    shift = torch.cat((w[:, :2] / den.sqrt()[:, None], w[:, 2, None] / den.pow(0.25)[:, None]), -1) * (D.amin(-1) - 1)[:, None]
    center = nucleus.center.double() + (nucleus.transform.double() @ shift.double()[..., None]).squeeze(-1)
    return Body(center, nucleus.transform * D[:, None, :], 4, nucleus.layout)

@dataclass
class Scene:
    nucleus: Body
    cell: Body
    valid: torch.Tensor
    raw: torch.Tensor

def decode_feasible(anchor, nucleus_raw, cell_raw, full_only=False, coeff=None):
    nucleus, valid = nucleus_geometry(anchor, nucleus_raw, full_only, coeff)
    cell = conditional_cell(nucleus, cell_raw)
    return Scene(nucleus, cell, valid, torch.cat((nucleus_raw, cell_raw), -1))
