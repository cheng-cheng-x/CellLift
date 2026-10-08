from celllift.runtime import resource_path as _public_resource
from .common import *
import sys
sys.path.insert(0, str(PUBLIC))
import torch, pandas as pd
from types import SimpleNamespace
from functools import lru_cache

class Adapter:
    dataset = 'tcga_crc_msi'
    task = 'msi'
    classes = 2

    def __init__(self, device='cuda'):
        from celllift.morphology_interaction.route_b.models import RouteB
        self.device = device
        self.entries = {e['graph_id']: e for e in read(DATA / 'v16c_parallel_routes_v1/tcga_crc_msi/parallel_v1/scene_index.json')['graphs']}
        self.rows = []
        grouped = {}
        for gid, e in self.entries.items():
            grouped.setdefault(str(e['metadata']['patient_id']), []).append(gid)
        for split in ('train', 'val', 'test'):
            source = DATA / f'official_split_full_v16c/splits/tcga_crc_msi/{split}_rows.json'
            if not source.exists():
                continue
            for r in read(source):
                self.rows.append(dict(r, split=split, cluster_id=str(r['patient_id']), graph_ids=grouped.get(str(r['patient_id']), [])))
        units = read(DATA / 'official_split_full_v16c/splits/tcga_crc_msi/units.json')
        if len(units) != 1:
            raise ValueError('Ambiguous CRC official unit')
        self.training_ids = set(map(str, units[0]['fit']))
        self.models = {}
        self.saved = {}
        self.paths = {}
        self.weights = {}
        root = RESULT / 'official_split_full_v16c/official_split_full_v16c_seed42'
        for key, arm in [('H2', 'B2'), ('H3', 'B3')]:
            p = root / f'routes/b/tcga_crc_msi/jobs/{arm}/seed_42/outer_00/best.pt'
            s = torch.load(p, map_location=device, weights_only=False)
            m = RouteB(classes=1, arm=arm, bag=True).to(device)
            m.load_state_dict(s['model'])
            m.eval().requires_grad_(False)
            self.models[key] = m
            self.paths[key] = p
            self.weights[key] = s
            self.saved[key] = {}
            for split, name in [('test', 'test_predictions.parquet'), ('val', 'predictions.parquet')]:
                q = root / f'tcga_crc_msi/route/{arm}/official/seed_42/fold_00' / name
                if q.exists():
                    for r in pd.read_parquet(q).to_dict('records'):
                        self.saved[key][split, str(r['sample_id'])] = np.asarray(r['probability'])

    @lru_cache(maxsize=256)
    def graph(self, gid):
        from celllift.morphology_interaction.data import SceneGraph
        with np.load(self.entries[gid]['path']) as z:
            return SceneGraph(gid, {k: z[k] for k in z.files})

    def input(self, r):
        graphs = [self.graph(g) for g in r['graph_ids']]
        rich = []
        centers = []
        for g in graphs:
            z = float(np.median(g.nucleus_center[g.valid3d, 2])) if g.valid3d.any() else 0
            rich.append(np.concatenate([g.node3d, g.nucleus_transform.reshape(-1, 9), g.cell_transform.reshape(-1, 9), g.cell_center - g.nucleus_center, g.nucleus_center[:, 2:3] - z], 1))
            centers.append(np.full(len(g.node3d), z, np.float32))
        return dict(rich=np.nan_to_num(np.concatenate(rich)), two=np.concatenate([g.node2d for g in graphs]), valid=np.concatenate([g.valid3d & g.include for g in graphs]), image=np.concatenate([g.dino for g in graphs]), groups=np.concatenate([np.full(len(g.node3d), i, int) for i, g in enumerate(graphs)]), graphs=graphs, row=r, zcenter=np.concatenate(centers))

    def forward(self, r):
        return Forward(self, self.input(r))

class Forward:

    def __init__(self, a, data):
        from celllift.morphology_interaction.data import Sample, build_batch
        self.adapter = a
        self.data = data
        self.rich = torch.as_tensor(data['rich'], device=a.device)
        self.batches = {}
        self.constants = {}
        r = data['row']
        cache = SimpleNamespace(dataset=a.dataset, get=lambda gid: a.graph(gid))
        sample = Sample(r['sample_id'], r['graph_ids'], r['label_id'], 0, r['cluster_id'], np.ones(1, np.float32))
        for key, arm in [('H2', 'B2'), ('H3', 'B3')]:
            s = a.weights[key]
            self.batches[key] = build_batch(cache, [sample], arm=arm, device=a.device, **{k: s[k] for k in ('mean', 'std', 'fill3d', 'edge_mean2d', 'edge_std2d', 'edge_mean3d', 'edge_std3d')})
        b = self.batches['H3']
        s = a.weights['H3']
        self.mean = torch.as_tensor(s['mean'][38:], device=a.device)
        self.std = torch.as_tensor(s['std'][38:], device=a.device)
        self.em = torch.as_tensor(s['edge_mean3d'], device=a.device)
        self.es = torch.as_tensor(s['edge_std3d'], device=a.device)
        self.q2 = torch.as_tensor(np.concatenate([g.edge2d[:, 1] for g in data['graphs']]), device=a.device)
        self.z = torch.as_tensor(data['zcenter'], device=a.device)
        self.xy = b.nucleus_center[:, :2]
        self.valid = b.valid3d
        self.src, self.dst = b.edge_index
        self.edgevalid = self.valid[self.src] & self.valid[self.dst]
        with torch.no_grad():
            self.baseline2 = self._logits(a.models['H2'](self.batches['H2'])['logits'])
        m = a.models['H3']
        with torch.no_grad():
            coord, assign = m.assign(b.dino)
            appear = m.appear_embed(coord)
            include = b.include.to(assign.dtype).unsqueeze(-1)
            self.assign = assign * include
            mass = assign.new_zeros((int(b.n_tiles), m.k))
            mass.index_add_(0, b.node_tile, self.assign)
            count = assign.new_zeros((int(b.n_tiles), 1))
            count.index_add_(0, b.node_tile, include)
            self.frac = mass / count.clamp_min(1.0)
            self.denom = mass.unsqueeze(-1).clamp_min(1.0)
            u = assign.new_zeros((int(b.n_tiles), m.k, appear.shape[-1]))
            u.index_add_(0, b.node_tile, self.assign.unsqueeze(-1) * appear.unsqueeze(1))
            self.appearance_proto = u / self.denom
            self.appearance_interaction = m.U(self.appearance_proto)
            expected = torch.softmax(self._logits(m(b)['logits']), -1)
            actual = torch.softmax(self(self.rich), -1)
            self.cache_identity_error = float((expected - actual).abs().max())
            if self.cache_identity_error > 1e-05:
                raise ValueError('Cached fixed appearance path changes original probabilities')

    def _logits(self, z):
        return torch.stack([torch.zeros_like(z), z], -1)

    def edges(self, x):
        s, t = (self.src, self.dst)
        w = x.new_tensor([4 / 18, 4 / 18, 5 / 21], dtype=torch.float64)
        nt = x[:, 12:21].reshape(-1, 3, 3).double()
        ct = x[:, 21:30].reshape(-1, 3, 3).double()
        nc = nt * w @ nt.transpose(-1, -2)
        cc = ct * w @ ct.transpose(-1, -2)
        center = torch.cat([self.xy, x[:, 33:34] + self.z[:, None]], 1).double()
        cell = center + x[:, 30:33].double()

        def spacing(c, cov):
            delta = c[t] - c[s]
            dist = torch.linalg.vector_norm(delta, dim=1)
            u = delta / dist[:, None].clamp_min(1e-12)
            left = torch.einsum('bi,bij,bj->b', u, cov[s], u).clamp_min(0).add(1e-12).sqrt()
            right = torch.einsum('bi,bij,bj->b', u, cov[t], u).clamp_min(0).add(1e-12).sqrt()
            return (dist / (left + right)).float()
        qn = spacing(center, nc)
        qc = spacing(cell, cc)
        radius = (3 * x[:, 0].double().exp() / (4 * np.pi)).pow(1 / 3)
        dz = ((center[t, 2] - center[s, 2]).abs() / (radius[s] + radius[t]).clamp_min(1e-06)).float()
        raw = torch.stack([dz, qn, qc, qn - self.q2, x[t, 0] - x[s, 0], x[t, 4] - x[s, 4], x[t, 1] - x[s, 1], x[t, 5] - x[s, 5]], -1)
        out = torch.where(self.edgevalid[:, None], (raw - self.em) / self.es, torch.zeros_like(raw))
        return torch.nan_to_num(out).clamp(-8, 8)

    def __call__(self, x, trace=False, arm='H3', pathway='consistent', edge_override=None):
        if x.ndim == 3:
            return torch.cat([self(t, arm=arm, pathway=pathway) for t in x], 0)
        if arm == 'H2':
            return self.baseline2
        original = self.batches['H3']
        b = SimpleNamespace(**original.__dict__)
        node = torch.where(self.valid[:, None], (x[:, :12] - self.mean) / self.std, torch.zeros_like(x[:, :12]))
        node = torch.where(b.include[:, None], node, torch.zeros_like(node))
        node = torch.nan_to_num(node).clamp(-8, 8)
        b.node = torch.cat([b.node[:, :38], node if pathway != 'edge_only' else b.node[:, 38:]], -1)
        if edge_override is not None:
            e = torch.where(self.edgevalid[:, None], (edge_override - self.em) / self.es, torch.zeros_like(edge_override))
            b.edge = torch.cat([b.edge[:, :4], torch.nan_to_num(e).clamp(-8, 8)], -1)
        else:
            b.edge = torch.cat([b.edge[:, :4], self.edges(x) if pathway != 'node_only' else b.edge[:, 4:]], -1)
        m = self.adapter.models['H3']
        geometry = m.geom_embed(m.geometry_token(b))
        v = self.assign.new_zeros((int(b.n_tiles), m.k, geometry.shape[-1]))
        v.index_add_(0, b.node_tile, self.assign.unsqueeze(-1) * geometry.unsqueeze(1))
        v = v / self.denom
        interaction = self.appearance_interaction * m.V(v)
        evidence = torch.cat((self.frac.unsqueeze(-1), self.appearance_proto, v, interaction), -1)
        tile = m.cell_head(evidence).sum(1)
        from celllift.morphology_interaction.foundation.ops import segment_softmax, segment_sum
        score = m.tile_score(tile).squeeze(-1)
        weight = segment_softmax(score, b.tile_bag, int(b.n_bags))
        bag = segment_sum(weight.unsqueeze(-1) * m.tile_value(tile), b.tile_bag, int(b.n_bags))
        logits = self._logits(m.head(bag).reshape(-1))
        if trace:
            return (logits, dict(geometry_encoding=geometry, prototype_geometry=v, appearance_geometry_interaction=interaction, tile_summary=tile, patient_bag=bag, output=logits))
        return logits
