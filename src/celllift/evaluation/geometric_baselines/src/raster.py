from celllift.runtime import resource_path as _public_resource
import torch
from .geometry import slab_factor

@torch.no_grad()
def raw_rows(center, root, h, lam, method, prepared, valid, chunk=131072):
    owner = prepared['obs_owner']
    plane = prepared['plane']
    count = len(owner)
    if not count:
        return []
    unique, rev = torch.unique(owner * 3 + plane, return_inverse=True)
    node = unique // 3
    planes = unique % 3
    factor, exists, _ = slab_factor(h[node], planes, method)
    scale = factor * lam[node]
    active = exists & valid[node]
    Q = root @ root.transpose(-1, -2)
    inv = torch.linalg.inv(root)
    ext = Q.diagonal(dim1=-2, dim2=-1).sqrt()[node] * scale[:, None]
    low = torch.floor((center[node] - ext) / 0.46 - 0.5).long() - 1
    high = torch.ceil((center[node] + ext) / 0.46 - 0.5).long() + 1
    size = (high - low + 1).clamp_min(0)
    amount = size.prod(1) * active
    offset = torch.cat((amount.new_zeros(1), amount.cumsum(0)))
    predicted = torch.zeros(len(unique), device=owner.device, dtype=torch.long)
    sums = center.new_zeros(len(unique), 2)

    def inside(nodes, xy, sc):
        q = (inv[nodes] @ (xy.double() - center[nodes])[:, :, None])[:, :, 0]
        return q.square().sum(1) <= sc.square()
    for start in range(0, int(offset[-1]), chunk):
        flat = torch.arange(start, min(start + chunk, int(offset[-1])), device=owner.device)
        o = torch.searchsorted(offset[1:], flat, right=True)
        idx = flat - offset[o]
        xy = (torch.stack((low[o, 0] + idx % size[o, 0], low[o, 1] + idx // size[o, 0]), 1).float() + 0.5) * 0.46
        hit = inside(node[o], xy, scale[o])
        predicted.index_add_(0, o, hit.long())
        sums.index_add_(0, o, xy.double() * hit[:, None])
    intersection = torch.zeros(count, device=owner.device, dtype=torch.long)
    for start in range(0, len(prepared['observation']), chunk):
        o = prepared['observation'][start:start + chunk]
        xy = (torch.floor(prepared['xy'][start:start + chunk] / 0.46) + 0.5) * 0.46
        hit = inside(owner[o], xy, scale[rev[o]]) & active[rev[o]]
        intersection.index_add_(0, o, hit.long())
    pc = predicted[rev]
    tc = prepared['target_count']
    assert ((intersection <= pc) & (intersection <= tc)).all()
    centroid = sums[rev] / pc[:, None].clamp_min(1)
    dice = 2 * intersection.double() / (pc + tc).clamp_min(1)
    error = (centroid - prepared['target_centroid']).norm(dim=1)
    values = torch.stack((owner.double(), plane.double(), dice, error, (pc - tc).abs() / tc.clamp_min(1), (pc * 0.46 ** 2).double(), tc.double(), (pc == 0).double(), valid[owner].double()), 1).cpu().numpy()
    return [dict(node=int(x[0]), plane=int(x[1]), dice=float(x[2]), centroid_error_um=float(x[3]) if not x[7] else None, relative_area_error=float(x[4]), predicted_area_um2=float(x[5]), target_area_px=int(x[6]), empty_prediction=bool(x[7]), geometry_valid=bool(x[8])) for x in values]
