from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from .frozen.collision import contact_scale as reference_contact, constraints, aabb_pairs

def contact_scale(ci, ti, cj, tj, p):
    if not ci.is_cuda:
        return reference_contact(ci, ti, cj, tj, p)
    from .frozen.contact_fused import contact_scale as solve
    alpha, diag = solve(ci, ti, cj, tj, p)
    if 'ids' not in diag:
        return ((ci.sum(-1) + cj.sum(-1) + ti.sum((-1, -2)) + tj.sum((-1, -2))) * 0, diag)
    ids = diag['ids']
    scale = diag['scale']
    distance = diag['distance']
    primal = diag['primal']
    mult = diag['multiplier']
    delta = cj[ids] - ci[ids]
    centers = torch.stack((torch.zeros_like(delta), delta / distance[:, None]), 1).double()
    inverse = torch.linalg.inv((torch.stack((ti[ids], tj[ids]), 1) / scale[:, None, None, None]).double())
    live = constraints(primal, centers, inverse, p)
    correction = distance / scale * (mult * (live - live.detach())).sum(-1)
    alpha = alpha.index_add(0, ids, correction.to(alpha.dtype))
    return (alpha, {k: v for k, v in diag.items() if k in ('lower', 'gap', 'stationarity')})

def collision_sum(body, valid, chunk=8192):
    pairs = aabb_pairs(body.center, body.transform, torch.zeros(len(valid), device=valid.device, dtype=torch.long), valid, body.p)
    total = body.transform.sum() * 0
    hits = torch.zeros_like(valid)
    penetrations = []
    for i, j in pairs.split(chunk, dim=1):
        if not len(i):
            continue
        a, _ = contact_scale(body.center[i], body.transform[i], body.center[j], body.transform[j], body.p)
        pen = torch.relu(1 - a)
        total = total + pen.square().sum()
        hit = pen > 1e-05
        hits[i[hit]] = True
        hits[j[hit]] = True
        penetrations.append(pen.detach())
    return (total, dict(cell_collision_rate=hits.sum() / valid.sum().clamp_min(1), colliding=hits, penetration=torch.cat(penetrations) if penetrations else body.center.new_empty(0)))
