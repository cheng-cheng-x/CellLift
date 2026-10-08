from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import torch
from torch import nn
torch.use_deterministic_algorithms(True)
from .data import Graph
from .frozen.camera_ring import CameraRingMPNN
from .geometry import Anchor, Body, Scene, decode_feasible, project_slab
from .frozen.geometry import unit_circle
from .candidate_tables import make_tables
from .selector import Selector
from .scorer import Scorer

class MessageLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(259, 128), nn.GELU(), nn.Linear(128, 128))
        self.update = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())

    def forward(self, h, g):
        i, j = g.edge_index
        delta = (g.nucleus_xy_um[i] - g.nucleus_xy_um[j]) / 60
        m = self.message(torch.cat((h[i], h[j], delta, delta.norm(dim=-1, keepdim=True)), -1))
        agg = torch.zeros_like(h).index_add(0, j, m)
        degree = torch.bincount(j, minlength=len(h)).clamp_min(1)[:, None]
        return h + self.update(torch.cat((h, agg / degree), -1))

@dataclass
class Proposal:
    raw: torch.Tensor
    geometry: Scene
    tables: object
    token: torch.Tensor
    scores: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    search: dict | None = None
    online_scores: torch.Tensor | None = None

class ConditionalSceneNetwork(nn.Module):

    def __init__(self, manifest, config, single=False):
        super().__init__()
        self.config = config
        self.k = 1 if single else 9
        self.backbone = CameraRingMPNN(manifest['ray_mean_um'], manifest['ray_std_um'])
        self.dino = nn.Linear(384, 128)
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        self.messages = nn.ModuleList([MessageLayer(), MessageLayer()])
        self.query_embed = None if single else nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 128))
        self.nucleus_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 5 if single else 4))
        self.cell_head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 5))
        for head in (self.nucleus_head, self.cell_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if not single:
            nn.init.zeros_(self.query_embed[-1].weight)
            nn.init.zeros_(self.query_embed[-1].bias)
        self.online_head = None if single else nn.Sequential(nn.Linear(152, 128), nn.GELU(), nn.Linear(128, 1))
        if self.online_head is not None:
            nn.init.zeros_(self.online_head[-1].weight)
            nn.init.zeros_(self.online_head[-1].bias)
        self.scorer = None

    def encode(self, g):
        if type(g) is not Graph:
            raise TypeError('Only middle-input Graph is accepted')
        g.validate()
        h = self.backbone(g.nucleus_rays_um, g.nucleus_xy_um, g.edge_index)
        h = self.fuse(torch.cat((h, self.dino(g.dino_features.float())), -1))
        for layer in self.messages:
            h = layer(h, g)
        return h

    def decode_subset(self, g, raw, flat_ids):
        owner = flat_ids // self.k
        flat = raw.reshape(-1, 10)[flat_ids]
        scenes = []
        for start in range(0, len(flat), self.config['decode_chunk']):
            ids = owner[start:start + self.config['decode_chunk']]
            x = flat[start:start + self.config['decode_chunk']].float()
            scenes.append(decode_feasible(Anchor(g.fitted_center_xy[ids].float(), g.fitted_root_xy[ids].float()), x[:, :5], x[:, 5:], coeff=g.coeff.select(ids)))

        def body(kind):
            bs = [getattr(s, kind) for s in scenes]
            return Body(torch.cat([x.center for x in bs]), torch.cat([x.transform for x in bs]), 4, torch.cat([x.layout for x in bs]))
        return Scene(body('nucleus'), body('cell'), torch.cat([s.valid for s in scenes]), flat)

    def decode_raw(self, g, raw):
        return self.decode_subset(g, raw, torch.arange(raw.shape[0] * self.k, device=raw.device))

    def propose(self, g, scoring=False):
        h = self.encode(g)
        if self.k == 1:
            raw = torch.cat((self.nucleus_head(h), self.cell_head(h)), -1)[:, None, :]
        else:
            q = g.query.float()
            f = torch.stack((q, q.square(), q.sin(), q.cos()), -1)
            c = h[:, None, :] + self.query_embed(f)
            raw = torch.cat((q[:, :, None], self.nucleus_head(c), self.cell_head(c)), -1)
        with torch.no_grad():
            geo = self.decode_raw(g, raw)
            t = make_tables(geo, len(h), self.k, g.graph_ptr, self.config['table_chunk'])
        p = Proposal(raw, geo, t, h)
        if self.online_head is not None and (not scoring):
            descriptor = self.online_descriptor(g, geo)
            p.online_scores = self.online_head(torch.cat((h[:, None, :].expand(-1, self.k, -1), descriptor), -1)).squeeze(-1)
        if scoring:
            if self.scorer is None:
                raise RuntimeError('Compatibility stage has not completed')
            p.scores = self.scorer(self.features(g, p))
            p.labels, p.search = solve_scene(t, (-p.scores.log_softmax(-1) + t.prior) * t.valid[:, None], g.graph_ptr, self.config)
        return p

    def transfer_shape(self, state):
        if self.k != 9:
            raise ValueError('Only the candidate stage receives shape weights')
        target = self.state_dict()
        copied = []
        for key, value in state.items():
            if key.startswith(('query_embed.', 'online_head.', 'scorer.')):
                continue
            if key in ('nucleus_head.2.weight', 'nucleus_head.2.bias'):
                value = value[1:]
            if key not in target or target[key].shape != value.shape:
                raise ValueError('Stage transfer mismatch: ' + key)
            target[key] = value
            copied.append(key)
        expected = [key for key in target if not key.startswith(('query_embed.', 'online_head.', 'scorer.'))]
        if set(copied) != set(expected):
            raise ValueError('Incomplete shape-stage transfer')
        self.load_state_dict(target)
        return dict(copied_tensors=len(copied), removed='nucleus_head final row 0 (position)', query_output_zero=bool((self.query_embed[-1].weight == 0).all() and (self.query_embed[-1].bias == 0).all()))

    @torch.no_grad()
    def online_descriptor(self, g, scene):
        n = len(g.nucleus_id)
        k = self.k
        scale = torch.linalg.det(g.fitted_root_xy.float()).abs().sqrt().clamp_min(1e-06)
        origin = torch.cat((g.fitted_center_xy, scale.new_full((n, 1), 2.5)), -1)
        parts = []
        for kind in ('nucleus', 'cell'):
            body = getattr(scene, kind)
            parts.extend(((body.center.reshape(n, k, 3) - origin[:, None, :]) / scale[:, None, None], body.transform.reshape(n, k, 9) / scale[:, None, None]))
        return torch.where(g.coeff.valid[:, None, None], torch.cat(parts, -1).float(), torch.zeros((n, k, 24), device=scale.device))

    @torch.no_grad()
    def descriptor(self, g, scene):
        n = len(g.nucleus_id)
        k = self.k
        scale = torch.linalg.det(g.fitted_root_xy.float()).abs().sqrt().clamp_min(1e-06)
        origin = torch.cat((g.fitted_center_xy, scale.new_full((n, 1), 2.5)), -1)
        owner = torch.arange(n, device=scale.device).repeat_interleave(k)
        dirs = unit_circle(36, device=scale.device, dtype=torch.float32)
        features = []
        for kind in ('nucleus', 'cell'):
            body = getattr(scene, kind)
            features.extend(((body.center.reshape(n, k, 3) - origin[:, None, :]) / scale[:, None, None], body.transform.reshape(n, k, 9) / scale[:, None, None]))
            relative = Body(body.center - torch.cat((g.fitted_center_xy[owner], scale.new_zeros(n * k, 1)), -1), body.transform, 4, body.layout)
            for plane in range(3):
                support, exists = project_slab(relative, dirs.new_tensor([(plane - 1) * 5, plane * 5]), dirs)
                features.extend((torch.where(exists[:, None], support, torch.zeros_like(support)).reshape(n, k, 36) / scale[:, None, None], exists.reshape(n, k, 1).float()))
        out = torch.cat(features, -1).float()
        return torch.where(g.coeff.valid[:, None, None], out, torch.zeros_like(out))

    @torch.no_grad()
    def features(self, g, p):
        scale = torch.linalg.det(g.fitted_root_xy.float()).abs().sqrt().clamp_min(1e-06)
        return dict(node=torch.cat((p.token.detach(), g.dino_features.float(), g.nucleus_rays_um.float() / scale[:, None]), -1), candidate=self.descriptor(g, p.geometry), edge_index=g.edge_index, valid=g.coeff.valid)

    def attach_scorer(self, normalization):
        for p in self.parameters():
            p.requires_grad_(False)
        self.scorer = Scorer(548, 246, normalization).to(next(self.parameters()).device)

    def forward(self, graph):
        return selected(self.propose(graph, scoring=True))

def starts(unary, ptr, k=9):
    n = len(unary)
    group = torch.repeat_interleave(torch.arange(len(ptr) - 1, device=ptr.device), ptr[1:] - ptr[:-1])
    local = torch.arange(n, device=ptr.device) - ptr[group]
    return [unary.argmin(-1)] + [(local + s) % k for s in (0, 3, 6)]

@torch.no_grad()
def solve_scene(t, unary, ptr, config, return_pool=False):
    solver = Selector(t)
    initial = starts(unary, ptr)
    pool = []
    reports = []
    for x in initial:
        y, report = solver.solve(unary, [x], config['selection_sweeps'], graph_ptr=ptr)
        pool.append(y)
        reports.append(report)
    sizes = ptr[1:] - ptr[:-1]
    group = torch.repeat_interleave(torch.arange(len(sizes), device=ptr.device), sizes)
    e = torch.arange(len(t.i), device=ptr.device)
    keys = []
    for y in pool:
        potential = solver.pot[e, y[t.i], y[t.j]]
        key = unary.new_zeros(len(sizes), 3)
        key[:, :2].index_add_(0, group[t.i], potential)
        key[:, 2].index_add_(0, group, unary.gather(1, y[:, None])[:, 0])
        keys.append(key)
    from .selector import lexarg
    keys = torch.stack(keys, 1)
    which = lexarg(keys[:, :, 0], keys[:, :, 1], keys[:, :, 2])
    stack = torch.stack(pool, 1)
    labels = stack[torch.arange(len(unary), device=ptr.device), which[group]]
    report = dict(starts=reports, key=solver.key(labels, unary))
    return (labels, report, pool) if return_pool else (labels, report)

def selected(p, k=9, labels=None):
    labels = p.labels if labels is None else labels
    ids = torch.arange(len(labels), device=labels.device) * k + labels
    s = p.geometry
    return Scene(s.nucleus.select(ids), s.cell.select(ids), s.valid[ids], s.raw[ids])
