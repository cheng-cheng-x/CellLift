from celllift.runtime import resource_path as _public_resource
from .common import *
import torch
import pandas as pd
from functools import lru_cache
from types import SimpleNamespace
import sys
sys.path.insert(0, str(PUBLIC))

class Adapter:
    dataset = 'bracs'
    task = 'roi7'
    classes = 7

    def __init__(self, device='cuda'):
        from celllift.benchmark_prediction.train_geometry import DualSet
        self.device = device
        self.entries = {e['graph_id']: e for e in read(DATA / 'v16c_parallel_routes_v1/bracs/parallel_v1/scene_index.json')['graphs']}
        self.rows = []
        for split in ('train', 'val', 'test'):
            for r in read(DATA / f'official_split_full_v16c/splits/bracs/{split}_rows.json'):
                r = dict(r, split=split, cluster_id=str(r['wsi_id']))
                r['graph_ids'] = [k for k, e in self.entries.items() if str(e.get('metadata', {}).get('roi_id')) == str(r['sample_id'])]
                self.rows.append(r)
        self.models = {}
        self.saved = {}
        self.paths = {}
        for arm, width in [('H2', 38), ('H3', 12)]:
            root = RESULT / f'official_split_full_v16c/official_split_full_v16c_seed42/bracs/feature/{arm}/official/seed_42/fold_00'
            p = root / 'best.pt'
            s = torch.load(p, map_location=device, weights_only=False)
            if arm == 'H3':
                units = read(DATA / 'official_split_full_v16c/splits/bracs/units.json')
                if len(units) != 1:
                    raise ValueError('Ambiguous official BRACS training unit')
                self.training_ids = set(map(str, units[0]['fit']))
            geom = DualSet(width, 1, 32, 'deepsets').to(device)
            head = torch.nn.Sequential(torch.nn.Linear(416, 128), torch.nn.SiLU(), torch.nn.Dropout(0.1), torch.nn.Linear(128, 7)).to(device)
            geom.load_state_dict(s['geom'])
            head.load_state_dict(s['head'])
            geom.eval().requires_grad_(False)
            head.eval().requires_grad_(False)
            self.models[arm] = (geom, head)
            self.paths[arm] = p
            self.saved[arm] = {}
            for split, name in [('val', 'predictions.parquet'), ('test', 'test_predictions.parquet')]:
                if (root / name).exists():
                    for r in pd.read_parquet(root / name).to_dict('records'):
                        self.saved[arm][split, str(r['sample_id'])] = np.asarray(r['probability'], float)

    @lru_cache(maxsize=16)
    def graph(self, gid):
        with np.load(self.entries[gid]['path']) as z:
            return {k: z[k] for k in ('node2d', 'node3d', 'include', 'valid3d', 'dino')}

    def input(self, r):
        two = []
        rich = []
        valid = []
        images = []
        groups = []
        rgb = []
        for i, gid in enumerate(r['graph_ids']):
            g = self.graph(gid)
            inc = g['include'].astype(bool)
            if not inc.any():
                inc = np.ones(len(inc), bool)
            x = g['node3d'][inc].copy()
            v = g['valid3d'][inc].astype(bool)
            x[~v] = 0
            two.append(np.nan_to_num(g['node2d'][inc]))
            rich.append(np.nan_to_num(x))
            valid.append(v)
            d = g['dino'][inc].astype(np.float32)
            images.append(d)
            rgb.append(d.mean(0) if len(d) else np.zeros(384, np.float32))
            groups.extend([i] * len(d))
        if not rgb:
            raise ValueError('No original cached graph in ROI')
        return dict(two=np.concatenate(two), rich=np.concatenate(rich), valid=np.concatenate(valid), image=np.concatenate(images), groups=np.asarray(groups), rgb=np.mean(np.stack(rgb), 0))

    def forward(self, r):
        return Forward(self, self.input(r))

class Forward:

    def __init__(self, a, data):
        self.adapter = a
        self.data = data
        self.rich = torch.as_tensor(data['rich'], device=a.device, dtype=torch.float32)
        self.two = torch.as_tensor(data['two'], device=a.device, dtype=torch.float32)
        self.rgb = torch.as_tensor(data['rgb'], device=a.device, dtype=torch.float32)
        self.empty = torch.zeros((len(self.rich), 1), device=a.device)

    def __call__(self, x, trace=False, arm='H3'):
        if x.ndim == 3:
            return torch.cat([self(t, trace=False, arm=arm) for t in x], 0)
        geom, head = self.adapter.models[arm]
        node = x if arm == 'H3' else self.two
        out = geom(node, self.empty)
        fused = torch.cat([out, self.rgb])
        logit = head(fused)[None]
        if trace:
            return (logit, {'geometry_encoding': geom.enc_a(node), 'geometry_pool': out, 'image_geometry_fusion': fused, 'output': logit})
        return logit
