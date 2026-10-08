from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from celllift.reconstruction.src.geometry import Body, project_slab

@torch.no_grad()
def pixel_inside(body, inverse, owner, xy, planes):
    inv = inverse[owner].double()
    center = body.center[owner].double()
    dxy = xy.double() - center[:, :2]
    base = torch.einsum('nij,nj->ni', inv[:, :, :2], dxy) - inv[:, :, 2] * center[:, 2, None]
    zdir = inv[:, :, 2]
    from .pixel_fused import inside
    assert body.p == 4
    return inside(base, zdir, planes)

@torch.no_grad()
def raw_rows(body, prepared, valid, chunk=131072, mpp=0.46):
    owner, plane = (prepared['obs_owner'], prepared['plane'])
    count = len(owner)
    if not count:
        return []
    body = Body(body.center.double(), body.transform.double(), body.p, body.layout)
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
    columns = torch.stack((owner.double(), plane.double(), dice, error, ((pc - tc).abs() / tc.clamp_min(1)).double(), (pc * mpp * mpp).double(), tc.double(), (pc == 0).double(), valid[owner].double()), -1).cpu().numpy()
    return [dict(node=int(x[0]), plane=int(x[1]), dice=float(x[2]), centroid_error_um=float(x[3]) if not bool(x[7]) else None, relative_area_error=float(x[4]), predicted_area_um2=float(x[5]), target_area_px=int(x[6]), empty_prediction=bool(x[7]), geometry_valid=bool(x[8])) for x in columns]
