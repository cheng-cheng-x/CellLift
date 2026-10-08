from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch

def kp_support(vector, p, steps=40):
    a = vector[..., :2].norm(dim=-1)
    b = vector[..., 2].abs()
    with torch.no_grad():
        lo, hi = (torch.zeros_like(a), torch.ones_like(a))
        tiny = torch.finfo(a.dtype).tiny
        for _ in range(steps):
            t = (lo + hi) * 0.5
            derivative = 0.5 * p * a * t.pow(p - 1) / (1 - t.pow(p)).clamp_min(tiny).sqrt()
            lo = torch.where(derivative < b, t, lo)
            hi = torch.where(derivative < b, hi, t)
        t = (lo + hi) * 0.5
        t = torch.where(b == 0, torch.zeros_like(t), t)
        t = torch.where(a == 0, torch.ones_like(t), t)
    return a * (1 - t.pow(p)).clamp_min(0).sqrt() + b * t

def constraints(primal, centers, inverses, p, *, derivatives=False):
    x, alpha = (primal[:, :3], primal[:, 3])
    u = torch.einsum('nbij,nbj->nbi', inverses, x[:, None] - centers)
    w = u / alpha[:, None, None]
    g = w[..., :2].square().sum(-1) + w[..., 2].pow(p)
    value = alpha[:, None] * (g - 1)
    if not derivatives:
        return value
    gradient = torch.cat((2 * w[..., :2], (p * w[..., 2].pow(p - 1))[..., None]), -1)
    hd = torch.cat((torch.full_like(w[..., :2], 2), (p * (p - 1) * w[..., 2].pow(p - 2))[..., None]), -1)
    gx = torch.einsum('nbji,nbj->nbi', inverses, gradient)
    ga = g - 1 - (gradient * w).sum(-1)
    jac = torch.cat((gx, ga[..., None]), -1)
    xx = inverses.transpose(-1, -2) @ (hd[..., :, None] * inverses) / alpha[:, None, None, None]
    xa = -torch.einsum('nbji,nbj->nbi', inverses, hd * w) / alpha[:, None, None]
    aa = (hd * w.square()).sum(-1) / alpha[:, None]
    hess = primal.new_zeros((len(primal), 2, 4, 4))
    hess[..., :3, :3] = xx
    hess[..., :3, 3] = xa
    hess[..., 3, :3] = xa
    hess[..., 3, 3] = aa
    return (value, jac, hess)

def _solve(centers, transforms, p):
    inverses = torch.linalg.inv(transforms)
    x = centers.mean(1)
    u = torch.einsum('nbij,nbj->nbi', inverses, x[:, None] - centers)
    alpha = 2 * (u[..., :2].norm(dim=-1) + u[..., 2].abs()).amax(-1) + 1
    primal = torch.cat((x, alpha[:, None]), -1)
    eye = torch.eye(4, device=x.device, dtype=x.dtype)
    objective_gradient = primal.new_tensor((0.0, 0.0, 0.0, 1.0))
    mus = (1.0, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07)
    if primal.dtype == torch.float64:
        mus += (1e-08, 1e-09)
    for mu in mus:
        for _ in range(32):
            f, jac, hess = constraints(primal, centers, inverses, p, derivatives=True)
            multiplier = -mu / f
            grad = objective_gradient + (multiplier[..., None] * jac).sum(1)
            if (grad.norm(dim=-1) < (1e-06 if primal.dtype == torch.float32 else 1e-09)).all():
                break
            hessian = (multiplier[..., None, None] * hess + (mu / f.square())[..., None, None] * jac[..., :, None] * jac[..., None, :]).sum(1)
            diagonal = hessian.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            balanced = hessian / diagonal[:, :, None] / diagonal[:, None, :]
            balanced = 0.5 * (balanced + balanced.transpose(-1, -2))
            step = torch.linalg.solve_ex(balanced + 32 * torch.finfo(primal.dtype).eps * eye, -grad / diagonal, check_errors=False).result / diagonal
            directional = (grad * step).sum(-1)
            old = primal[:, 3] - mu * torch.log(-f).sum(-1)
            rate = torch.ones_like(alpha)
            accepted = torch.zeros_like(alpha, dtype=torch.bool)
            next_primal = primal.clone()
            for _ in range(40):
                trial = primal + rate[:, None] * step
                positive = trial[:, 3] > 0
                safe = torch.where(positive[:, None], trial, primal)
                tf = constraints(safe, centers, inverses, p)
                feasible = positive & (tf < 0).all(-1)
                value = safe[:, 3] - mu * torch.log((-tf).clamp_min(torch.finfo(tf.dtype).tiny)).sum(-1)
                ok = feasible & (value <= old + 0.0001 * rate * directional)
                take = ok & ~accepted
                next_primal = torch.where(take[:, None], safe, next_primal)
                accepted |= ok
                rate = torch.where(accepted, rate, rate * 0.5)
                if accepted.all():
                    break
            primal = next_primal
    f, jac, _ = constraints(primal, centers, inverses, p, derivatives=True)
    gram = jac @ jac.transpose(-1, -2)
    multiplier = torch.linalg.solve_ex(gram, -jac[..., 3], check_errors=False).result
    normal = multiplier[:, 0, None] * jac[:, 0, :3]
    transformed = torch.einsum('nbji,nj->nbi', transforms, normal)
    denominator = kp_support(transformed, p).sum(-1)
    lower = ((centers[:, 1] - centers[:, 0]) * normal).sum(-1) / denominator
    gap = primal[:, 3] - lower
    stationarity = (objective_gradient + (multiplier[..., None] * jac).sum(1)).norm(dim=-1)
    return (primal, multiplier, lower, gap, stationarity)

def contact_scale(center_i, transform_i, center_j, transform_j, p, *, check=True):
    delta = center_j - center_i
    different = delta.square().sum(-1) > 0
    ids = torch.where(different)[0]
    alpha = delta.sum(-1) * 0 + (transform_i.sum((-1, -2)) + transform_j.sum((-1, -2))) * 0
    lower = torch.zeros_like(alpha)
    gaps = torch.zeros_like(alpha)
    residuals = torch.zeros_like(alpha)
    if not ids.numel():
        return (alpha, dict(lower=lower, gap=gaps, stationarity=residuals))
    with torch.no_grad():
        scale = 0.5 * (transform_i[ids].norm(dim=(-1, -2)) + transform_j[ids].norm(dim=(-1, -2)))
        distance = delta[ids].norm(dim=-1)
        alpha_scale = distance / scale
        centers = torch.stack((torch.zeros_like(delta[ids]), delta[ids] / distance[:, None]), 1)
        transforms = torch.stack((transform_i[ids], transform_j[ids]), 1) / scale[:, None, None, None]
        primal, multiplier, bound, gap, residual = _solve(centers.double(), transforms.double(), p)
        tolerance = 2e-05 if center_i.dtype == torch.float32 else 2e-07
        finite = torch.isfinite(primal).all() & torch.isfinite(multiplier).all() & torch.isfinite(bound).all() & torch.isfinite(gap).all() & torch.isfinite(residual).all()
        if check and (not finite or (gap > tolerance * (1 + primal[:, 3])).any() or (gap < -tolerance).any() or (multiplier < 0).any() or (residual > 0.002).any()):
            raise RuntimeError(f'contact primal-dual gap failed: max={gap.max().item():.6g}')
    live_centers = torch.stack((torch.zeros_like(delta[ids]), delta[ids] / distance[:, None]), 1).double()
    live_inverses = torch.linalg.inv((torch.stack((transform_i[ids], transform_j[ids]), 1) / scale[:, None, None, None]).double())
    live_f = constraints(primal, live_centers, live_inverses, p)
    result = alpha_scale * (primal[:, 3] + (multiplier * (live_f - live_f.detach())).sum(-1))
    alpha = alpha.index_copy(0, ids, result.to(alpha.dtype))
    return (alpha, dict(lower=lower.index_copy(0, ids, (bound * alpha_scale).to(lower.dtype)), gap=gaps.index_copy(0, ids, (gap * alpha_scale).to(gaps.dtype)), stationarity=residuals.index_copy(0, ids, residual.to(residuals.dtype))))

def aabb_pairs(centers, transforms, graph_index, valid, p, *, block=256):
    with torch.no_grad():
        extent = kp_support(transforms, p)
        padding = 8 * torch.finfo(centers.dtype).eps * (centers.abs() + extent + 1)
        lower, upper = (centers - extent - padding, centers + extent + padding)
        pairs = []
        index = torch.arange(len(centers), device=centers.device)
        for graph in torch.unique(graph_index).tolist():
            members = index[graph_index == graph]
            glower, gupper = (lower[members], upper[members])
            for start in range(0, len(members), block):
                rows = members[start:start + block]
                overlap = (lower[rows, None] <= gupper[None]).all(-1)
                overlap &= (glower[None] <= upper[rows, None]).all(-1)
                overlap &= rows[:, None] < members[None]
                overlap &= valid[rows, None] & valid[members][None]
                a, b = torch.where(overlap)
                pairs.append(torch.stack((rows[a], members[b])))
        if not pairs:
            return index.new_empty((2, 0))
        result = torch.cat(pairs, 1)
        order = torch.argsort(result[0] * len(centers) + result[1])
        return result[:, order]

def collision_loss(centers, transforms, graph_index, valid, p, *, chunk=8192):
    pairs = aabb_pairs(centers, transforms, graph_index, valid, p)
    loss = transforms.sum() * 0
    max_gap = transforms.new_zeros(())
    hits = torch.zeros(len(centers), device=centers.device, dtype=torch.bool)
    penetrations = []
    for edges in pairs.split(chunk, dim=1):
        if not edges.numel():
            continue
        i, j = edges
        alpha, diag = contact_scale(centers[i], transforms[i], centers[j], transforms[j], p)
        penetration = torch.relu(1 - alpha)
        loss = loss + penetration.square().sum()
        active = penetration.detach() > 1e-05
        hits[i[active]] = True
        hits[j[active]] = True
        max_gap = torch.maximum(max_gap, diag['gap'].max())
        penetrations.append(penetration.detach())
    return (loss / valid.sum().clamp_min(1), dict(cell_collision_rate=hits.sum() / valid.sum().clamp_min(1), max_contact_gap=max_gap, penetration=torch.cat(penetrations) if penetrations else transforms.new_empty(0), pair_count=pairs.shape[1]))
