from celllift.runtime import resource_path as _public_resource
from .common import *
import sys, argparse
sys.path.insert(0, str(PUBLIC))
import torch
from scipy.special import softmax

class Adapter:

    def __init__(self, dataset, device='cuda'):
        from celllift.core_and_nucleus_prediction.data import ArvanitiWindowDataset, LizardTileDataset
        from celllift.core_and_nucleus_prediction.models import ImageNodeExpert
        self.dataset = dataset
        self.task = 'nucleus6' if dataset == 'lizard' else 'local'
        self.device = device
        self.classes = 6 if dataset == 'lizard' else 4
        self.datasets = {}
        self.rows = []
        self.models = {}
        self.paths = {}
        self.saved = {}
        for split in ('fit', 'val', 'test'):
            d = LizardTileDataset(split, geom='a3', load_dino=True) if dataset == 'lizard' else ArvanitiWindowDataset(split, geom='g23', supervised=split == 'fit', load_dino=True, load_rgb=False, cache_items=False)
            self.datasets[split] = d
            for i, r in enumerate(d.rows):
                self.rows.append(dict(sample_id=r['graph_id'], graph_id=r['graph_id'], split=split, index=i, cluster_id=str(r.get('group_id') or r.get('core_id')), label_id=int(r.get('label_id', -1))))
        if dataset == 'arvaniti':
            import pandas as pd
            labels = pd.read_parquet(DATA / 'model_input_v1/arvaniti/04_labels_splits_v2/supervised_windows.parquet').set_index('graph_id')
            for r in self.rows:
                if r['graph_id'] in labels.index:
                    s = labels.loc[r['graph_id']]
                    r['label_id'] = int(s.label_id_p1 if r['split'] == 'test' else s.label_id)
        for dim, arm in [(38, 'LA2' if dataset == 'lizard' else 'A2'), (47, 'LA3' if dataset == 'lizard' else 'AS')]:
            root = RESULT / 'arvaniti_lizard_full_v16c_seed42_fix1' / dataset / 'interact' / arm / 'seed42'
            p = root / 'model.pt'
            m = ImageNodeExpert(dim, 7 if dataset == 'lizard' and dim == 47 else 3, classes=self.classes, pool=dataset == 'arvaniti').to(device)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=False))
            m.eval().requires_grad_(False)
            key = 'H2' if dim == 38 else 'H3'
            self.models[key] = m
            self.paths[key] = p
            self.saved[key] = {}
            for split in ('val', 'test'):
                p = root / (split + ('_infer_logits.npz' if dataset == 'arvaniti' else '_logits.npz'))
                if not p.exists():
                    continue
                with np.load(p, allow_pickle=True) as z:
                    probs = softmax(z['logits'].astype(np.float64), axis=-1)
                    if dataset == 'arvaniti':
                        self.saved[key].update({(split, str(g)): v for g, v in zip(z['graph_id'], probs)})
                    else:
                        self.saved[key].update({(split, str(g), int(n)): v for g, n, v in zip(z['graph_id'], z['nucleus_id'], probs)})

    def input(self, r):
        from celllift.core_and_nucleus_prediction.data import _load_geometry, _align
        ds = self.datasets[r['split']]
        item = dict(ds[r['index']])
        if self.dataset == 'arvaniti':
            item['label'] = r['label_id']
        packed = ds._geom(r['graph_id']) if self.dataset == 'arvaniti' else _load_geometry(ds.reader, ds.input_root, ds.scene_root, ds.residual_root, r['graph_id'])
        scene = packed['scene']
        ids = packed['ids']
        valid = np.isin(ids, scene['nucleus_id']) if scene is not None else np.zeros(len(ids), bool)
        if scene is not None and 'valid' in scene:
            valid &= _align(ids, scene, 'valid', 1).reshape(-1).astype(bool)
        centers = _align(ids, dict(nucleus_id=scene['nucleus_id'], x3=scene['nucleus_center']), 'x3', 3) if scene is not None and 'nucleus_center' in scene else np.zeros((len(ids), 3), np.float32)
        valid &= np.isfinite(centers).all(1) & np.isfinite(packed['x3']).all(1)
        zcenter = float(np.median(centers[valid, 2])) if valid.any() else 0
        rich = np.concatenate([packed['x3'], centers[:, 2:3] - zcenter], 1)
        return dict(rich=rich, two=packed['x2'], valid=valid, image=item['dino'], groups=np.zeros(len(ids), int), item=item, centers=centers, zcenter=zcenter, ids=ids)

    def forward(self, r):
        return Forward(self, self.input(r))

class Forward:

    def __init__(self, a, data):
        self.adapter = a
        self.data = data
        self.rich = torch.as_tensor(data['rich'], device=a.device)
        self.two = torch.as_tensor(data['two'], device=a.device)
        item = data['item']
        self.src = torch.as_tensor(item['src'], device=a.device, dtype=torch.long)
        self.dst = torch.as_tensor(item['dst'], device=a.device, dtype=torch.long)
        self.edge = torch.as_tensor(item['edge'], device=a.device)
        self.xy = torch.as_tensor(data['centers'][:, :2], device=a.device)
        self.dino = torch.as_tensor(data['image'], device=a.device)
        self.appear = {}
        with torch.no_grad():
            for arm, m in a.models.items():
                self.appear[arm] = m.appear(self.dino).detach()

    def __call__(self, x, trace=False, arm='H3', pathway='consistent', edge_override=None):
        if x.ndim == 3:
            return torch.stack([self(t, arm=arm, pathway=pathway) for t in x])
        m = self.adapter.models[arm]
        node = self.two if arm == 'H2' else torch.cat([self.two, x[:, :9] if pathway != 'edge_only' else self.rich[:, :9]], -1)
        edge = self.edge
        if self.adapter.dataset == 'lizard' and arm == 'H3':
            state = self.rich if pathway == 'node_only' else x
            center = torch.cat([self.xy, state[:, 9:10] + self.data['zcenter']], -1)
            delta = center[self.dst] - center[self.src]
            edge = torch.cat([edge, delta, torch.linalg.vector_norm(delta, dim=1, keepdim=True)], -1)
            if edge_override is not None:
                edge = torch.cat([self.edge, edge_override], -1)
        geom = m.morph(node)
        fused = m.fuse(torch.cat([self.appear[arm], geom], -1))
        local = m.relation.encode(fused, edge, self.src, self.dst)
        if self.adapter.dataset == 'lizard':
            logits = m.relation.head(local)
        elif not len(local):
            logits = m.window(local.new_zeros((1, m.window[0].in_features)))[0]
        else:
            logits = m.window(torch.cat([local.mean(0), torch.log1p(local.new_tensor(float(len(local)))).reshape(1)])[None])[0]
        if trace:
            return (logits, dict(geometry_encoding=geom, fusion=fused, message_passing=local, output=logits))
        return logits

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['predict', 'perturb'])
    p.add_argument('--dataset', choices=['arvaniti', 'lizard'], required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(4)
    a = Adapter(args.dataset)
    folder = OUT / a.dataset / a.task / 'model'
    sha = {k: digest(v) for k, v in a.paths.items()}
    write(folder / 'input_paths.json', dict(protocol='fix1', checkpoint_sha256=sha, direct_3d=['direct_geometry9'] + (['nucleus_center edge deltaXYZ and distance'] if a.dataset == 'lizard' else []), shape_matrix_direct=False, edge_graph_fixed=True))
    matcher = sensitivity = None
    if args.stage == 'perturb':
        from .matching import Matcher
        matcher = Matcher(a)
        sensitivity = Matcher(a, image=True)
    for i, r in enumerate(a.rows):
        if r['split'] != args.split or stable(r['sample_id']) % args.shards != args.shard:
            continue
        path = folder / args.stage / args.split / f"{stable(r['sample_id']):012x}.json"
        if path.exists():
            continue
        f = a.forward(r)
        item = f.data['item']
        owned = item['owned'].astype(bool) if a.dataset == 'lizard' else np.ones(1, bool)
        if a.dataset == 'lizard' and (not owned.any()):
            write(path, dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], split=args.split, status='not_applicable', reason='graph has no owned prediction targets', n_owned=0))
            continue
        with torch.no_grad():
            p3 = torch.softmax(f(f.rich), -1).cpu().numpy()
            p2 = torch.softmax(f(f.rich, arm='H2'), -1).cpu().numpy()
        error = {}
        for arm, prob in [('H2', p2), ('H3', p3)]:
            if a.dataset == 'lizard':
                ids = f.data['ids'][owned]
                expected = [a.saved[arm].get((args.split, r['sample_id'], int(n))) for n in ids]
                complete = all((v is not None for v in expected))
                ex = np.stack(expected) if complete and len(expected) else np.empty((0, 6))
                pred = prob[owned]
            else:
                v = a.saved[arm].get((args.split, r['sample_id']))
                complete = v is not None
                ex = v
                pred = prob
            error[arm] = dict(max_absolute_error=float(np.max(abs(pred - ex))) if complete and np.size(pred) else None, class_equal=bool(np.array_equal(np.argmax(pred, axis=-1), np.argmax(ex, axis=-1))) if complete else False, available=complete)
        record = dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], split=args.split, label_id=r['label_id'], reproduction=error, n_owned=int(owned.sum()), status='complete')
        if args.split in ('val', 'test') and any((not e['available'] or e['max_absolute_error'] is None or e['max_absolute_error'] > 1e-05 or (not e['class_equal']) for e in error.values())):
            write(path, dict(record, status='numerical_failure'))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(probability3=p3, probability2=p2, ids=f.data['ids'], owned=owned, labels=np.asarray(item['label']), valid=f.data['valid'])
        if args.stage == 'predict':
            with torch.no_grad():
                record['A2_3d_invariance'] = float((f(f.rich, arm='H2') - f(torch.zeros_like(f.rich), arm='H2')).abs().max())
        else:
            refs, covers = matcher.references(f.data, r['cluster_id'])
            perm, pc = matcher.permutations(f.data)
            im, ic = sensitivity.references(f.data, r['cluster_id'])
            methods = [('training_matched', refs, covers, 'consistent'), ('within_graph', perm, pc, 'consistent'), ('image_matched', im, ic, 'consistent')]
            if a.dataset == 'lizard':
                methods.extend([('node_only', refs, covers, 'node_only'), ('edge_only', refs, covers, 'edge_only')])
            record['coverage'] = {}
            for name, inputs, coverage, way in methods:
                probs = []
                for x in inputs:
                    with torch.no_grad():
                        probs.append(torch.softmax(f(torch.as_tensor(x, device=a.device), pathway=way), -1).cpu().numpy())
                payload[name] = np.stack(probs)
                record['coverage'][name] = (coverage.sum(1) / max(1, f.data['valid'].sum())).tolist()
        np.savez_compressed(path.with_suffix('.npz'), **payload)
        write(path, record)
        if i % 50 == 0:
            print(a.dataset, args.stage, i, flush=True)
    assert sha == {k: digest(v) for k, v in a.paths.items()}
if __name__ == '__main__':
    main()
