from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import math
import numpy as np
from celllift.runtime import torch
from .geometry import Body, rms_axes, positive_projection
from .frozen.geometry import unit_circle
from .frozen.collision import kp_support
from .contact import contact_scale
from .losses import student_t

@dataclass
class Tables:
    prior: torch.Tensor
    i: torch.Tensor
    j: torch.Tensor
    contact: torch.Tensor
    valid: torch.Tensor
    colors: list
    color_plans: list | None = None

    def conditional_color(self, labels, color_index):
        ids = self.colors[color_index]
        if self.color_plans is None:
            return self.conditional(labels)[ids]
        left, right, left_owner, right_owner = self.color_plans[color_index]
        out = self.prior.new_zeros(len(ids), self.prior.shape[1])
        out.index_add_(0, left_owner, self.contact[left, :, labels[self.j[left]]])
        out.index_add_(0, right_owner, self.contact[right, labels[self.i[right]], :])
        return out

    def conditional(self, labels):
        n, k = self.prior.shape
        out = self.prior.new_zeros(n, k)
        if len(self.i):
            e = torch.arange(len(self.i), device=self.i.device)
            out.index_add_(0, self.i, self.contact[e, :, labels[self.j]])
            out.index_add_(0, self.j, self.contact[e, labels[self.i], :])
        return out

    def energy_nodes(self, unary, labels, weight=10.0):
        result = unary.gather(1, labels[:, None]).squeeze(1).clone()
        if len(self.i):
            e = torch.arange(len(self.i), device=self.i.device)
            result.index_add_(0, self.i, weight * self.contact[e, labels[self.i], labels[self.j]])
        return result

def greedy_colors(i, j, n, device, with_plans=False):
    adjacency = [[] for _ in range(n)]
    left = i.cpu().numpy()
    right = j.cpu().numpy()
    for a, b in zip(left.tolist(), right.tolist()):
        adjacency[a].append(b)
        adjacency[b].append(a)
    color = np.full(n, -1, dtype=np.int32)
    for a, neighbors in enumerate(adjacency):
        used = {int(color[b]) for b in neighbors if color[b] >= 0}
        c = 0
        while c in used:
            c += 1
        color[a] = c
    groups = [np.where(color == c)[0] for c in range(int(color.max()) + 1)]
    colors = [torch.as_tensor(ids, device=device) for ids in groups]
    if not with_plans:
        return colors
    rank = np.empty(n, dtype=np.int64)
    for ids in groups:
        rank[ids] = np.arange(len(ids))
    plans = []
    for c in range(len(groups)):
        le = np.where(color[left] == c)[0]
        re = np.where(color[right] == c)[0]
        plans.append(tuple((torch.as_tensor(v, device=device) for v in (le, re, rank[left[le]], rank[right[re]]))))
    return (colors, plans)

@torch.no_grad()
def make_tables(scene, n, k, ptr, chunk=4096):
    prior = scene.raw.new_zeros(n * k)
    for kind, scale in [('nucleus', math.log(2)), ('cell', math.log(3))]:
        axes = rms_axes(getattr(scene, kind)).log()
        prior += 0.5 * ((axes - axes.mean(-1, keepdim=True)) / scale).square().mean(-1)
    prior = prior.reshape(n, k)
    valid = scene.valid.reshape(n, k)[:, 0]
    prior *= valid[:, None]
    c = scene.cell.center.reshape(n, k, 3).float()
    T = scene.cell.transform.reshape(n, k, 3, 3)
    ext = kp_support(T, 4)
    padding = 8 * torch.finfo(c.dtype).eps * (c.abs() + ext + 1)
    lower = (c - ext - padding).amin(1)
    upper = (c + ext + padding).amax(1)
    pairs = []
    bounds = ptr.cpu().tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        other = torch.arange(lo, hi, device=c.device)
        for start in range(lo, hi, 256):
            owner = torch.arange(start, min(start + 256, hi), device=c.device)
            hit = (lower[owner, None] <= upper[None, other]).all(-1) & (lower[None, other] <= upper[owner, None]).all(-1)
            hit &= valid[owner, None] & valid[None, other] & (owner[:, None] < other[None, :])
            a, b = torch.where(hit)
            if len(a):
                pairs.append(torch.stack([owner[a], other[b]]))
    pair = torch.cat(pairs, 1) if pairs else torch.empty((2, 0), dtype=torch.long, device=c.device)
    i, j = pair
    cost = c.new_zeros(len(i), k, k)
    total = len(i) * k * k
    for start in range(0, total, chunk):
        ids = torch.arange(start, min(start + chunk, total), device=c.device)
        e = ids // (k * k)
        a = ids // k % k
        b = ids % k
        c1, c2 = (c[i[e], a], c[j[e], b])
        t1, t2 = (T[i[e], a], T[j[e], b])
        extent = ext[i[e], a] + ext[j[e], b] + padding[i[e], a] + padding[j[e], b]
        hit = ((c1 - c2).abs() <= extent).all(-1)
        values = c.new_zeros(len(ids))
        if hit.any():
            exact = scene.cell.center.reshape(n, k, 3)
            alpha, _ = contact_scale(exact[i[e[hit]], a[hit]], t1[hit], exact[j[e[hit]], b[hit]], t2[hit], 4)
            values[hit] = (1 - alpha).clamp_min(0).to(values.dtype)
        cost.view(-1)[ids] = values
    colors, plans = greedy_colors(i, j, n, c.device, with_plans=True)
    return Tables(prior, i, j, cost, valid, colors, plans)

@torch.no_grad()
def select(tables, unary, initial=None, weight=10.0, sweeps=5):
    labels = unary.argmin(-1) if initial is None else initial.clone()
    for _ in range(sweeps):
        changes = torch.zeros((), device=unary.device, dtype=torch.long)
        for color_index, ids in enumerate(tables.colors):
            local = unary[ids] + weight * tables.conditional_color(labels, color_index)
            best = local.argmin(-1)
            improve = local.gather(1, best[:, None])[:, 0] < local.gather(1, labels[ids, None])[:, 0] - 1e-07
            labels[ids] = torch.where(improve, best, labels[ids])
            changes += improve.sum()
        if int(changes) == 0:
            break
    return labels

def _accumulate_observations(out, which, values):
    for start in range(0, len(which), 8192):
        out.index_add_(0, which[start:start + 8192], values[start:start + 8192])

@torch.no_grad()
def observation_tables(scene, g, s, k, chunk=8192):
    if chunk < 8192 or chunk % 8192:
        raise ValueError('observation solve block must be a multiple of 8192')
    n = len(g.nucleus_id)
    out = {key: scene.raw.new_zeros(n * k) for key in ('projection', 'positive_gap', 'nucleus_positive', 'nucleus_empty', 'cell_positive', 'cell_empty')}
    dirs = unit_circle(36, device=scene.raw.device, dtype=scene.raw.dtype)
    valid = scene.valid.reshape(n, k)[:, 0]
    for kind in ('nucleus', 'cell'):
        body = getattr(scene, kind)
        obs = getattr(s, kind)
        keep = valid[obs.anchor_node_index] & (obs.weight > 0)
        if kind == 'nucleus':
            keep &= obs.plane_index != 1
        all_ids = torch.where(keep)[0]
        targets = []
        for start in range(0, len(all_ids), 8192):
            ids = all_ids[start:start + 8192]
            owner = obs.anchor_node_index[ids]
            root = g.fitted_root_xy[owner].float()
            tc = (root @ obs.target_center_normalized[ids, :, None]).squeeze(-1)
            tr = root @ obs.target_root_normalized[ids]
            targets.append(tc @ dirs.T + torch.einsum('nji,kj->nki', tr, dirs).norm(dim=-1))
        target_table = torch.cat(targets) if targets else scene.raw.new_empty((0, len(dirs)))
        for start in range(0, len(all_ids) * k, chunk):
            flat = torch.arange(start, min(start + chunk, len(all_ids) * k), device=dirs.device)
            ids = all_ids[flat // k]
            owner = obs.anchor_node_index[ids]
            which = owner * k + flat % k
            center = g.fitted_center_xy[owner].float()
            target = target_table[flat // k]
            sub = body.select(which)
            sub = Body(sub.center - torch.cat([center, torch.zeros_like(center[:, :1])], -1), sub.transform, 4, sub.layout)
            plane = obs.plane_index[ids].to(dirs.dtype)
            slab = torch.stack([(plane - 1) * 5, plane * 5], -1)
            pred, gap, exists = positive_projection(sub, slab, dirs)
            adjacent = (obs.plane_index[ids] != 1).to(pred.dtype)
            out[kind + '_positive'].index_add_(0, which, adjacent)
            out[kind + '_empty'].index_add_(0, which, (~exists).to(pred.dtype) * adjacent)
            _accumulate_observations(out['projection'], which, student_t(pred - target).mean(-1) * obs.weight[ids])
            _accumulate_observations(out['positive_gap'], which, student_t(gap) * obs.weight[ids])
    return {key: value.reshape(n, k) for key, value in out.items()}

def node_norm(g, batch_graphs=None):
    sizes = g.graph_ptr[1:] - g.graph_ptr[:-1]
    group = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), sizes)
    return 1 / sizes[group].float() / float(len(sizes) if batch_graphs is None else batch_graphs)

@torch.no_grad()
def contact_metrics(tables, labels):
    hit = torch.zeros_like(tables.valid)
    deep = torch.zeros_like(hit)
    if len(tables.i):
        e = torch.arange(len(tables.i), device=labels.device)
        pen = tables.contact[e, labels[tables.i], labels[tables.j]]
        flag = pen > 1e-05
        large = pen > 0.1
        hit[tables.i[flag]] = True
        hit[tables.j[flag]] = True
        deep[tables.i[large]] = True
        deep[tables.j[large]] = True
    return dict(nodes=len(labels), valid=tables.valid.sum(), collided=hit.sum(), deep_collision=deep.sum())
