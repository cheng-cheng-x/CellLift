from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from .geometry import project_slab

@torch.no_grad()
def pixel_inside(body, inverse, owner, xy, planes):
    inv = inverse[owner]
    dxy = xy - body.center[owner, :2]
    base = torch.einsum('nij,nj->ni', inv[:, :, :2], dxy) - inv[:, :, 2] * body.center[owner, 2, None]
    zdir = inv[:, :, 2]
    lower = (planes.to(xy.dtype) - 1) * 5
    upper = planes.to(xy.dtype) * 5
    for _ in range(36):
        z = (lower + upper) / 2
        q = base + zdir * z[:, None]
        grad = 2 * (q[:, :2] * zdir[:, :2]).sum(-1) + body.p * q[:, 2].abs().pow(body.p - 1) * q[:, 2].sign() * zdir[:, 2]
        lower = torch.where(grad < 0, z, lower)
        upper = torch.where(grad < 0, upper, z)
    q = base + zdir * ((lower + upper) / 2)[:, None]
    return q[:, :2].square().sum(-1) + q[:, 2].abs().pow(body.p) <= 1

@torch.no_grad()
def raw_rows(body, prepared, valid, chunk=131072, mpp=0.46):
    owner, plane = (prepared['obs_owner'], prepared['plane'])
    count = len(owner)
    if not count:
        return []
    inverse = torch.linalg.inv(body.transform)
    unique, rev = torch.unique(owner * 3 + plane, return_inverse=True)
    nodes, planes = (unique // 3, unique % 3)
    slab = torch.stack(((planes - 1) * 5, planes * 5), -1).to(body.center.dtype)
    dirs = body.center.new_tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    support, exists = project_slab(body.select(nodes), slab, dirs)
    active = exists & valid[nodes]
    support = torch.where(active[:, None], support, torch.zeros_like(support))
    low = torch.floor(-support[:, 2:] / mpp - 0.5).long() - 1
    high = torch.ceil(support[:, :2] / mpp - 0.5).long() + 1
    extent = (high - low + 1).clamp_min(0)
    sizes = extent.prod(-1) * active
    offsets = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
    predicted = torch.zeros(len(unique), device=owner.device, dtype=torch.long)
    sums = body.center.new_zeros((len(unique), 2), dtype=torch.float64)
    for start in range(0, int(offsets[-1]), chunk):
        flat = torch.arange(start, min(start + chunk, int(offsets[-1])), device=owner.device)
        obs = torch.searchsorted(offsets[1:], flat, right=True)
        off = flat - offsets[obs]
        x = low[obs, 0] + off % extent[obs, 0]
        y = low[obs, 1] + off // extent[obs, 0]
        xy = (torch.stack((x, y), -1).float() + 0.5) * mpp
        inside = pixel_inside(body, inverse, nodes[obs], xy, planes[obs])
        predicted.index_add_(0, obs, inside.long())
        sums.index_add_(0, obs, xy.double() * inside[:, None])
    intersection = torch.zeros(count, device=owner.device, dtype=torch.long)
    for start in range(0, len(prepared['observation']), chunk):
        obs = prepared['observation'][start:start + chunk]
        xy = (torch.floor(prepared['xy'][start:start + chunk] / mpp) + 0.5) * mpp
        hit = pixel_inside(body, inverse, owner[obs], xy, plane[obs]) & valid[owner[obs]]
        intersection.index_add_(0, obs, hit.long())
    pc = predicted[rev]
    tc = prepared['target_count']
    centroid = sums[rev] / pc[:, None].clamp_min(1)
    if ((intersection > pc) | (intersection > tc)).any():
        raise RuntimeError('raster count inconsistency')
    dice = 2 * intersection.double() / (pc + tc).clamp_min(1)
    error = (centroid - prepared['target_centroid']).norm(dim=-1)
    rows = []
    for i in range(count):
        rows.append(dict(node=int(owner[i]), plane=int(plane[i]), dice=float(dice[i]), centroid_error_um=float(error[i]) if pc[i] > 0 else None, relative_area_error=float((pc[i] - tc[i]).abs() / tc[i].clamp_min(1)), predicted_area_um2=float(pc[i] * mpp * mpp), target_area_px=int(tc[i]), empty_prediction=not bool(pc[i]), geometry_valid=bool(valid[owner[i]])))
    return rows
