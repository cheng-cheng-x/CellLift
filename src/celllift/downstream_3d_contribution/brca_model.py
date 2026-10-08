from celllift.runtime import resource_path as _public_resource
from .common import *
import sys
sys.path.insert(0, str(PUBLIC))
import torch
import pandas as pd

class Adapter:
    dataset = 'tcga_brca'

    def __init__(self, task, device='cuda'):
        from celllift.breast_patient_prediction.routes.bags import build_bags, SlideCache
        from celllift.breast_patient_prediction.routes.arms import FusionMIL
        from celllift.breast_patient_prediction.routes.scale import NodePooledScale
        self.task = task
        self.device = device
        self.base, self.rows, _ = build_bags(task, need_image=True)
        for r in self.rows:
            r.update(sample_id=r['patient_id'], cluster_id=r['patient_id'], label_id=r['label'])
        self.classes = self.rows[0]['classes']
        self.cache = SlideCache(self.base, 'geometry', max_slides=2)
        self.scene_cache = SlideCache(self.base, 'scene', max_slides=1)
        self.models = {}
        self.saved = {}
        self.paths = {}
        self.scales = {}
        self.caps = {}
        self.chunks = {}
        self.thresholds = {}
        for arm in ('H2', 'H3'):
            root = RESULT / f'tcga_brca_downstream_v1/brca_downstream_routes_v2_seed42/{task}/{arm}/seed42'
            self.paths[arm] = root / 'checkpoint.pt'
            s = torch.load(self.paths[arm], map_location=device, weights_only=False)
            m = FusionMIL(self.classes, 'g2' if arm == 'H2' else 'g3').to(device)
            m.load_state_dict(s['model'])
            m.eval().requires_grad_(False)
            self.models[arm] = m
            self.scales[arm] = NodePooledScale.from_dict(read(root / 'geometry_scale.json'))
            config = read(root / 'metrics.json')
            self.caps[arm] = config.get('node_cap')
            self.chunks[arm] = int(config.get('chunk', 8))
            if os.environ.get('DOWNSTREAM_GRAPH_CHUNK'):
                self.chunks[arm] = min(self.chunks[arm], int(os.environ['DOWNSTREAM_GRAPH_CHUNK']))
            self.thresholds[arm] = config.get('threshold') if config.get('threshold') is not None else 0.5
            self.saved[arm] = {}
            for r in pd.read_parquet(root / 'predictions.parquet').to_dict('records'):
                p = np.asarray(r['prob']).reshape(-1)
                if len(p) == 1:
                    p = np.array([1 - p[0], p[0]])
                self.saved[arm][r['split'], r['patient_id']] = p
        self.classes = max(2, self.classes)

    def calibration_input(self, r, image=False):
        return self.input(r, load_image=image)

    def input(self, r, load_image=True):
        from celllift.breast_patient_prediction.routes.bags import load_tile_geometry
        from celllift.breast_patient_prediction.routes.geometry import isolate_tokens, truncate_by_nucleus_id
        chunks = []
        rich = []
        two = []
        valid = []
        image = []
        groups = []
        for i, tile in enumerate(r['tiles']):
            g = load_tile_geometry(self.cache, tile)
            tokens = {}
            for arm in ('H2', 'H3'):
                tokens[arm] = isolate_tokens(g['rays'], g['node3d'], g['include'], g['valid3d'], 'g2' if arm == 'H2' else 'g3', nucleus_id=g['nucleus_id'], node_cap=self.caps[arm], scale=self.scales[arm])
            tokens['tile'] = tile
            chunks.append(tokens)
            t = tokens['H3']
            n = len(t['nucleus'])
            rich.append(np.concatenate([t['nucleus'][:, 36:], t['cell'][:, 36:]], 1))
            keep = truncate_by_nucleus_id(g['nucleus_id'], self.caps['H3'])
            two.append(g['node2d'][keep] if len(g['node2d']) else np.zeros((1, 38), np.float32))
            valid.append((g['include'] & g['valid3d'] & np.isfinite(g['node3d'][:, :9]).all(1))[keep] if len(g['node2d']) else np.zeros(1, bool))
            if len(g['node2d']) and load_image:
                scene = self.scene_cache.get(tile['slide_id'], tile['graph_id'])
                if not np.array_equal(scene['nucleus_id'], g['nucleus_id']):
                    raise ValueError('Node DINO / geometry instance order differs')
                image.append(scene['dino'][keep])
            else:
                image.append(np.zeros((len(keep) if len(g['node2d']) else 1, 384), np.float32))
            groups.extend([i] * n)
        return dict(rich=np.concatenate(rich), two=np.concatenate(two), valid=np.concatenate(valid), image=np.concatenate(image), groups=np.asarray(groups), chunks=chunks, dino=r['dino_global'])

    def forward(self, r):
        return Forward(self, self.input(r))

class Forward:
    integration_batch_size = 2
    perturbation_batch_size = 4

    def __init__(self, a, data):
        from celllift.breast_patient_prediction.routes.geometry import pad_sets
        self.adapter = a
        self.data = data
        self.rich = torch.as_tensor(data['rich'], device=a.device)
        self.packed = {}
        self.bounds = []
        self.gradient_budget = int(torch.cuda.mem_get_info(self.rich.device)[0] * 0.2) if self.rich.is_cuda else 0
        self.image = {}
        self.offset = np.r_[0, np.cumsum([len(t['H3']['nucleus']) for t in data['chunks']])]
        self.gather = {}
        with torch.no_grad():
            for arm in ('H2', 'H3'):
                self.image[arm] = a.models[arm].image(torch.as_tensor(data['dino'], device=a.device)).detach()
                self.packed[arm] = []
                for start in range(0, len(data['chunks']), a.chunks[arm]):
                    stop = min(len(data['chunks']), start + a.chunks[arm])
                    self.packed[arm].append((start, stop, pad_sets([t[arm] for t in data['chunks'][start:stop]], a.device)))
            for start, stop, p in self.packed['H3']:
                width = p['nucleus_tokens'].shape[1]
                ix = np.zeros((stop - start, width), np.int64)
                valid = np.zeros_like(ix, bool)
                for j in range(start, stop):
                    n = self.offset[j + 1] - self.offset[j]
                    ix[j - start, :n] = np.arange(self.offset[j], self.offset[j + 1])
                    valid[j - start, :n] = True
                self.gather[start, stop] = (torch.as_tensor(ix, device=a.device), torch.as_tensor(valid, device=a.device))

    def __call__(self, x, trace=False, arm='H3'):
        if x.ndim == 3 and arm == 'H3' and (not trace):
            return self.batched(x)
        if x.ndim == 3:
            return torch.cat([self(t, arm=arm) for t in x], 0)
        m = self.adapter.models[arm]
        geom = []
        for start, stop, original in self.packed[arm]:
            p = dict(original)
            if arm == 'H3':
                ix, valid = self.gather[start, stop]
                z = torch.where(valid[:, :, None], x[ix], 0)
                zeros = z.new_zeros(*z.shape[:2], 36)
                p['nucleus_tokens'] = torch.cat([zeros, z[:, :, :5]], -1)
                p['cell_tokens'] = torch.cat([zeros, z[:, :, 5:]], -1)
            if x.requires_grad and arm == 'H3' and self.needs_checkpoint(1):
                from torch.utils.checkpoint import checkpoint

                def encode(nuc, cell, nmask, cmask):
                    return m.encode_geom(dict(nucleus_tokens=nuc, cell_tokens=cell, nucleus_mask=nmask, cell_mask=cmask))
                geom.append(checkpoint(encode, p['nucleus_tokens'], p['cell_tokens'], p['nucleus_mask'], p['cell_mask'], use_reentrant=False))
            else:
                geom.append(m.encode_geom(p))
        geometry = torch.cat(geom)
        fused = m.fuse(torch.cat([self.image[arm], geometry], -1))
        logits, _ = m.mil(fused)
        if logits.ndim == 1:
            logits = logits[None]
        if logits.shape[-1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], -1)
        if trace:
            return (logits, dict(geometry_encoding=geometry, image_geometry_fusion=fused, output=logits))
        return logits

    def needs_checkpoint(self, batch):
        return os.environ.get('DOWNSTREAM_FORCE_CHECKPOINT') == '1' or len(self.rich) * batch * 64 * 4 * 12 > self.gradient_budget

    def batched(self, x):
        b = len(x)
        m = self.adapter.models['H3']
        parts = []
        for start, stop, original in self.packed['H3']:
            width = original['nucleus_tokens'].shape[1]
            ix, valid = self.gather[start, stop]
            z = torch.where(valid[None, :, :, None], x[:, ix], 0)
            zeros = z.new_zeros(*z.shape[:3], 36)
            p = dict(nucleus_tokens=torch.cat([zeros, z[:, :, :, :5]], -1).reshape(b * (stop - start), width, 41), cell_tokens=torch.cat([zeros, z[:, :, :, 5:]], -1).reshape(b * (stop - start), width, 41))
            for key in ('nucleus_mask', 'cell_mask'):
                p[key] = original[key][None].expand(b, *original[key].shape).reshape(b * (stop - start), width)
            if x.requires_grad and self.needs_checkpoint(b):
                from torch.utils.checkpoint import checkpoint

                def encode(nuc, cell, nmask, cmask):
                    return m.encode_geom(dict(nucleus_tokens=nuc, cell_tokens=cell, nucleus_mask=nmask, cell_mask=cmask))
                encoded = checkpoint(encode, p['nucleus_tokens'], p['cell_tokens'], p['nucleus_mask'], p['cell_mask'], use_reentrant=False)
            else:
                encoded = m.encode_geom(p)
            parts.append(encoded.reshape(b, stop - start, -1))
        geometry = torch.cat(parts, 1)
        fused = m.fuse(torch.cat([self.image['H3'][None].expand(b, -1, -1), geometry], -1))
        logits, _ = m.mil(fused)
        if logits.shape[-1] == 1:
            logits = torch.cat([torch.zeros_like(logits), logits], -1)
        if not getattr(self, 'batch_equivalence_checked', False):
            with torch.no_grad():
                serial = self(x[0].detach())
                error = float((torch.softmax(serial, -1) - torch.softmax(logits[:1].detach(), -1)).abs().max())
            if error > 1e-05:
                raise ValueError(f'Batched complete-bag probability mismatch: {error}')
            self.batch_equivalence_checked = True
            self.batch_equivalence_error = error
        return logits
