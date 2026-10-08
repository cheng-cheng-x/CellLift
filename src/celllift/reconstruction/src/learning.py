from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from celllift.runtime import torch
from .candidate_tables import observation_tables, select
from .geometry import rms_axes
from .contact import contact_scale
from .weighted_observations import weighted_observations

def cached_contact(t, labels):
    edge = torch.arange(len(t.i), device=t.i.device)
    return t.contact[edge, labels[t.i], labels[t.j]]

def contact_penalty(alpha, weight=10.0):
    return weight * (1 - alpha).clamp_min(0)

@torch.no_grad()
def metrics(p, g, obs, labels):
    t = p.tables
    n = len(labels)
    k = p.raw.shape[1]
    ids = torch.arange(n, device=labels.device)
    pen = cached_contact(t, labels)
    sizes = g.graph_ptr[1:] - g.graph_ptr[:-1]
    group = torch.repeat_interleave(torch.arange(len(sizes), device=labels.device), sizes)
    ng = len(sizes)
    agg = lambda x: x.new_zeros(ng).index_add(0, group, x)
    hit = torch.zeros_like(t.valid)
    core = hit.clone()
    flag = pen > 1e-05
    deep = pen > 0.02
    hit[t.i[flag]] = True
    hit[t.j[flag]] = True
    core[t.i[deep]] = True
    core[t.j[deep]] = True
    edgegroup = group[t.i]
    depth = pen.new_zeros(ng).index_add(0, edgegroup, (pen - 0.02).clamp_min(0))
    counts = pen.new_zeros(ng).index_add(0, edgegroup, deep.float())
    raw = agg((obs['projection'] + obs['positive_gap'] + t.prior)[ids, labels]) / sizes
    linear = pen.new_zeros(ng).index_add(0, edgegroup, pen)
    vals = dict(actual_energy=raw + 10 * linear / sizes, fixed_monitor=raw + 10 * depth / sizes, raw_energy=raw, projection=agg(obs['projection'][ids, labels]) / sizes, positive_gap=agg(obs['positive_gap'][ids, labels]) / sizes, prior=agg(t.prior[ids, labels]) / sizes, collision=agg(hit.float()) / agg(t.valid.float()).clamp_min(1), core_collision=agg(core.float()) / agg(t.valid.float()).clamp_min(1), core_depth=depth / sizes, core_pairs=counts, core_pair_density=counts / sizes, coverage=agg(t.valid.float()) / sizes)
    z = p.geometry.nucleus.center.reshape(n, k, 3)[ids, labels, 2].float()
    h = p.geometry.nucleus.transform.reshape(n, k, 3, 3)[ids, labels, 2, 2]
    nv = agg(t.valid.float()).clamp_min(1)
    zm = agg(z * t.valid) / nv
    vals['z_std'] = (agg((z - zm[group]).square() * t.valid) / nv).sqrt()
    vals['height'] = agg(h * t.valid) / nv
    for kind in ('nucleus', 'cell'):
        positive = agg(obs[kind + '_positive'][ids, labels])
        empty = agg(obs[kind + '_empty'][ids, labels])
        vals[kind + '_empty'] = empty / positive.clamp_min(1)
    return {k: float(x.sum()) for k, x in vals.items()} | dict(graphs=ng)

def objective(model, p, g, sup, config, batch_graphs=None, live=True):
    n, k = p.raw.shape[:2]
    t = p.tables
    dev = p.raw.device
    sizes = g.graph_ptr[1:] - g.graph_ptr[:-1]
    ng = len(sizes)
    nb = ng if batch_graphs is None else batch_graphs
    group = torch.repeat_interleave(torch.arange(ng, device=dev), sizes)
    ids = torch.arange(n, device=dev)
    normal = 1 / sizes.float() / nb
    aggregate = lambda x: x.new_zeros(ng).index_add(0, group, x)
    with torch.no_grad():
        obs = observation_tables(p.geometry, g, sup, k, config['observation_chunk'])
        unary = obs['projection'] + obs['positive_gap'] + t.prior
        if k == 1:
            best = torch.zeros(n, device=dev, dtype=torch.long)
            pool = [best]
        else:
            best = select(t, (p.online_scores.detach() + t.prior) * t.valid[:, None], weight=config['contact_weight'], sweeps=config['training_selection_sweeps'])
            pool = [best]
            local = ids - g.graph_ptr[group]
            for shift in (0, 3, 6):
                initial = (local + shift) % k
                pool.append(initial)
                pool.append(select(t, unary, initial, weight=config['contact_weight'], sweeps=config['training_selection_sweeps']))
        p.labels = best
        p.search = dict(kind='same_energy_scene_support', scenes=len(pool))
        m = len(pool)
        labels = torch.stack(pool)
        unique = torch.ones(m, ng, device=dev, dtype=torch.bool)
        bounds = g.graph_ptr.cpu().tolist()
        for gi, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            for a in range(1, m):
                unique[a, gi] = ~(labels[:a, lo:hi][:, t.valid[lo:hi]] == labels[a, lo:hi][t.valid[lo:hi]]).all(1).any()
        edge = torch.arange(len(t.i), device=dev)
        codes = torch.stack([edge * k * k + y[t.i] * k + y[t.j] for y in pool])
        code, inverse = torch.unique(codes.flatten(), sorted=True, return_inverse=True)
        inverse = inverse.reshape(m, -1)
        e = code // (k * k)
        ca = code // k % k
        cb = code % k
        left = t.i[e] * k + ca
        right = t.j[e] * k + cb
        penalties = config['contact_weight'] * t.contact[e, ca, cb]
        energies = []
        for a, y in enumerate(pool):
            E = aggregate(unary[ids, y])
            E.index_add_(0, group[t.i], penalties[inverse[a]])
            energies.append(E)
        energies = torch.stack(energies)
        tau = config['scene_temperature']
        logits = (-energies / tau).masked_fill(~unique, -torch.inf)
        q = logits.softmax(0)
        free = -tau * (torch.logsumexp(logits, 0) - unique.sum(0).float().log())
        nw = p.raw.new_zeros(n, k)
        for a, y in enumerate(pool):
            nw[ids, y] += q[a, group] * normal[group]
        weights = torch.stack([q[a, group[t.i]] * normal[group[t.i]] for a in range(m)])
        ew = p.raw.new_zeros(len(code)).index_add(0, inverse.flatten(), weights.flatten())
        measured = metrics(p, g, obs, best)
        measured.update(scene_free_energy=float((free * normal).sum() * nb), unique_scenes=float(unique.sum()), active_pairs=float((ew > 0).sum()), contact_weight=float(config['contact_weight']) * ng)
        constant = torch.stack([aggregate(t.prior[ids, y]) + (E - aggregate(unary[ids, y])) for y, E in zip(pool, energies)])
    score = p.raw.sum() * 0
    if p.online_scores is not None:
        pred = torch.stack([aggregate(p.online_scores[ids, y] * t.valid) + constant[a] for a, y in enumerate(pool)])
        logp = torch.log_softmax((-pred / tau).masked_fill(~unique, -torch.inf), 0)
        score = (tau * (q * (q.clamp_min(1e-30).log() - logp.masked_fill(~unique, 0))).sum(0) * normal).sum()
    measured['online_score_loss'] = float(score.detach() * nb)
    if not live:
        return ((free * normal).sum() + config['online_score_weight'] * score, measured)
    active_ids = torch.where((nw * t.valid[:, None]).flatten() > 0)[0]
    if not len(active_ids):
        active_ids = ids * k
    geo = model.decode_subset(g, p.raw, active_ids)
    lookup = torch.full((n * k,), -1, device=dev, dtype=torch.long)
    lookup[active_ids] = torch.arange(len(active_ids), device=dev)
    projection, gap = weighted_observations(geo, g, sup, k, nw * t.valid[:, None], lookup, config['support_chunk'])
    prior = p.raw.new_zeros(len(active_ids))
    for kind, scale in [('nucleus', math.log(2)), ('cell', math.log(3))]:
        axes = rms_axes(getattr(geo, kind)).log()
        prior = prior + 0.5 * ((axes - axes.mean(-1, keepdim=True)) / scale).square().mean(-1)
    compact = (prior * geo.valid * nw.flatten()[active_ids]).sum()
    constraint = p.raw.sum() * 0
    active = torch.where(ew > 0)[0]
    for start in range(0, len(active), config['contact_chunk']):
        ix = active[start:start + config['contact_chunk']]
        a = lookup[left[ix]]
        b = lookup[right[ix]]
        c = geo.cell
        alpha, _ = contact_scale(c.center[a], c.transform[a], c.center[b], c.transform[b], 4)
        constraint = constraint + (contact_penalty(alpha.float(), config['contact_weight']) * ew[ix]).sum()
    weighted = projection + gap + compact + constraint
    freevalue = (free * normal).sum()
    loss = weighted + (freevalue - weighted.detach())
    measured['live_queries'] = len(active_ids)
    measured['all_queries'] = n * k
    return (loss + config['online_score_weight'] * score, measured)
