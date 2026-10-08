from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from .geometry import Body, positive_projection
from .frozen.geometry import unit_circle
from .losses import student_t

def weighted_observations(geo, g, s, k, node_weight, lookup, chunk):
    dirs = unit_circle(36, device=node_weight.device, dtype=torch.float32)
    projection = geo.raw.sum() * 0
    gap_total = projection
    for kind in ('nucleus', 'cell'):
        obs = getattr(s, kind)
        owner = obs.anchor_node_index
        keep = g.coeff.valid[owner] & (obs.weight > 0)
        if kind == 'nucleus':
            keep &= obs.plane_index != 1
        ob = torch.where(keep)[0]
        row, candidate = torch.where(node_weight[owner[ob]] > 0)
        selected = ob[row]
        own = owner[selected]
        flat = own * k + candidate
        for start in range(0, len(flat), chunk):
            ss = slice(start, start + chunk)
            oid = selected[ss]
            a = own[ss]
            j = candidate[ss]
            root = g.fitted_root_xy[a].float()
            center = g.fitted_center_xy[a].float()
            tc = (root @ obs.target_center_normalized[oid, :, None]).squeeze(-1)
            tr = root @ obs.target_root_normalized[oid]
            target = tc @ dirs.T + torch.einsum('nji,kj->nki', tr, dirs).norm(dim=-1)
            body = getattr(geo, kind).select(lookup[flat[ss]])
            body = Body(body.center - torch.cat((center, torch.zeros_like(center[:, :1])), -1), body.transform, 4, body.layout)
            plane = obs.plane_index[oid].float()
            slab = torch.stack(((plane - 1) * 5, plane * 5), -1)
            pred, gap, _ = positive_projection(body, slab, dirs)
            w = node_weight[a, j] * obs.weight[oid]
            projection = projection + (student_t(pred - target).mean(1) * w).sum()
            gap_total = gap_total + (student_t(gap) * w).sum()
    return (projection, gap_total)
