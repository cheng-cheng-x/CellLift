from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import time
import numpy as np
from celllift.runtime import torch

def lexarg(count, depth, unary):
    keep = count == count.amin(-1, keepdim=True)
    d = depth.masked_fill(~keep, float('inf')).amin(-1, keepdim=True)
    keep &= depth <= d + 1e-08
    return unary.masked_fill(~keep, float('inf')).argmin(-1)

class Selector:

    def __init__(self, t):
        self.t = t
        self.n, self.k = t.prior.shape
        self.pot = torch.stack(((t.contact > 0.02).float(), (t.contact - 0.02).clamp_min(0)), -1)
        a = t.i.cpu().tolist()
        b = t.j.cpu().tolist()
        adj = [set() for _ in range(self.n)]
        for i, j in zip(a, b):
            adj[i].add(j)
            adj[j].add(i)
        groups = []
        blocked = []
        for e, (i, j) in enumerate(zip(a, b)):
            group = next((g for g, s in enumerate(blocked) if i not in s and j not in s), len(groups))
            if group == len(groups):
                groups.append([])
                blocked.append(set())
            groups[group].append(e)
            blocked[group].update(adj[i] | adj[j] | {i, j})
        self.single = [self.plan(ids.cpu().tolist(), a, b) for ids in t.colors]
        self.pair = []
        for es in groups:
            ids = [a[e] for e in es] + [b[e] for e in es]
            self.pair.append((torch.tensor(es, device=t.i.device), self.plan(ids, a, b)))

    def plan(self, ids, a, b):
        rank = {v: r for r, v in enumerate(ids)}
        le = [e for e, v in enumerate(a) if v in rank]
        re = [e for e, v in enumerate(b) if v in rank]
        arrays = [ids, le, re, [rank[a[e]] for e in le], [rank[b[e]] for e in re]]
        return tuple((torch.tensor(x, device=self.t.i.device, dtype=torch.long) for x in arrays))

    def conditional(self, labels, plan):
        ids, le, re, lo, ro = plan
        t = self.t
        out = self.pot.new_zeros(len(ids), self.k, 2)
        out.index_add_(0, lo, self.pot[le, :, labels[t.j[le]], :])
        out.index_add_(0, ro, self.pot[re, labels[t.i[re]], :, :])
        return out

    def key(self, labels, unary):
        t = self.t
        e = torch.arange(len(t.i), device=labels.device)
        p = self.pot[e, labels[t.i], labels[t.j]].sum(0)
        return (int(p[0]), float(p[1]), float(unary.gather(1, labels[:, None]).sum()))

    def choose(self, c, d, u, current):
        choice = lexarg(c, d, u)
        r = torch.arange(len(choice), device=choice.device)
        old = (c[r, current], d[r, current], u[r, current])
        new = (c[r, choice], d[r, choice], u[r, choice])
        better = (new[0] < old[0]) | (new[0] == old[0]) & ((new[1] < old[1] - 1e-08) | (torch.abs(new[1] - old[1]) <= 1e-08) & (new[2] < old[2] - 1e-07))
        return (torch.where(better, choice, current), better.sum())

    @torch.no_grad()
    def solve(self, unary, initials, sweeps=4, graph_ptr=None):
        started = time.time()
        best = None
        best_key = None
        records = []
        solutions = []
        for initial in initials:
            labels = initial.clone()
            first = self.key(labels, unary)
            for sweep in range(sweeps):
                changes = torch.zeros((), device=labels.device, dtype=torch.long)
                for plan in self.single:
                    ids = plan[0]
                    p = self.conditional(labels, plan)
                    labels[ids], changed = self.choose(p[..., 0], p[..., 1], unary[ids], labels[ids])
                    changes += changed
                for es, plan in self.pair:
                    t = self.t
                    i = t.i[es]
                    j = t.j[es]
                    m = len(es)
                    p = self.conditional(labels, plan)
                    left = p[:m] - self.pot[es, :, labels[j], :]
                    right = p[m:] - self.pot[es, labels[i], :, :]
                    joint = left[:, :, None, :] + right[:, None, :, :] + self.pot[es]
                    u = unary[i, :, None] + unary[j, None, :]
                    selected, changed = self.choose(joint[..., 0].flatten(1), joint[..., 1].flatten(1), u.flatten(1), labels[i] * self.k + labels[j])
                    labels[i] = selected // self.k
                    labels[j] = selected % self.k
                    changes += changed
                if int(changes) == 0:
                    break
            solutions.append(labels.clone())
            key = self.key(labels, unary)
            records.append(dict(before=first, after=key, sweeps=sweep + 1))
            if best_key is None or key < best_key:
                best_key = key
                best = labels.clone()
        if graph_ptr is not None and len(graph_ptr) > 2:
            group = torch.repeat_interleave(torch.arange(len(graph_ptr) - 1, device=unary.device), graph_ptr[1:] - graph_ptr[:-1])
            ng = len(graph_ptr) - 1
            keys = []
            edge = torch.arange(len(self.t.i), device=unary.device)
            for lab in solutions:
                potential = self.pot[edge, lab[self.t.i], lab[self.t.j]]
                per = unary.new_zeros(ng, 3)
                per[:, :2].index_add_(0, group[self.t.i], potential)
                per[:, 2].index_add_(0, group, unary.gather(1, lab[:, None])[:, 0])
                keys.append(per)
            keys = torch.stack(keys, 1)
            which = lexarg(keys[:, :, 0], keys[:, :, 1], keys[:, :, 2])
            stack = torch.stack(solutions, 1)
            best = stack[torch.arange(self.n, device=unary.device), which[group]]
            best_key = self.key(best, unary)
        return (best, dict(starts=records, key=best_key, feasible=best_key[0] == 0, seconds=time.time() - started, pair_groups=len(self.pair)))

def verify(Tables, device='cuda'):
    from celllift.reconstruction.src.candidate_tables import greedy_colors
    i = torch.tensor([0, 0, 1], device=device)
    j = torch.tensor([1, 2, 2], device=device)
    torch.manual_seed(42)
    cost = torch.rand(3, 3, 3, device=device) * 0.04
    colors, plans = greedy_colors(i, j, 3, device, True)
    t = Tables(torch.zeros(3, 3, device=device), i, j, cost, torch.ones(3, device=device, dtype=torch.bool), colors, plans)
    s = Selector(t)
    labels = torch.tensor([0, 1, 2], device=device)
    u = torch.rand(3, 3, device=device)
    for es, plan in s.pair:
        assert len(es) == 1
        e = int(es[0])
        a = int(i[e])
        b = int(j[e])
        p = s.conditional(labels, plan)
        local = p[:1, :, None, :] - s.pot[es, :, labels[j[es]], :][:, :, None, :] + p[1:, None, :, :] - s.pot[es, labels[i[es]], :, :][:, None, :, :] + s.pot[es]
        for x in range(3):
            for y in range(3):
                test = labels.clone()
                test[a] = x
                test[b] = y
                delta = torch.tensor(s.key(test, u)[:2], device=device) - torch.tensor(s.key(labels, u)[:2], device=device)
                assert torch.allclose(delta, local[0, x, y] - local[0, labels[a], labels[b]], atol=1e-06)
    i = torch.tensor([0], device=device)
    j = torch.tensor([1], device=device)
    colors, plans = greedy_colors(i, j, 2, device, True)
    t = Tables(torch.zeros(2, 2, device=device), i, j, torch.eye(2, device=device)[None] * 0.1, torch.ones(2, device=device, dtype=torch.bool), colors, plans)
    s = Selector(t)
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    got, info = s.solve(u, [torch.tensor([0, 1], device=device)])
    assert got.tolist() == [1, 0] and info['feasible']
    return dict(block_delta_matches_enumeration=True, simultaneous_swap=True)
