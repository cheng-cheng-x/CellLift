from celllift.runtime import resource_path as _public_resource
from .common import *
from .al_model import Adapter
from .model import integrated, select_cases, completeness_gate
import argparse, torch

class TargetForward:

    def __init__(self, f, index=None, class_indices=None):
        self.f = f
        self.rich = f.rich
        self.index = index
        self.class_indices = class_indices

    def __call__(self, x):
        if x.ndim == 3:
            return torch.cat([self(z) for z in x], 0)
        logits = self.f(x)
        if self.index is not None:
            return logits[self.index:self.index + 1]
        if self.class_indices is not None:
            return logits[self.class_indices].mean(0, keepdim=True)
        return logits[None]

def candidates(a, folder, split, stage):
    rows = {r['sample_id']: r for r in a.rows if r['split'] == split}
    out = []
    for p in (folder / 'predict' / split).glob('*.json'):
        r = read(p)
        if r['status'] != 'complete':
            continue
        with np.load(p.with_suffix('.npz')) as z:
            if a.dataset == 'arvaniti':
                label = rows[r['sample_id']]['label_id']
                if label < 0:
                    continue
                out.append(dict(sample_id=r['sample_id'], graph_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=label, correct2=int(z['probability2'].argmax()) == label, correct3=int(z['probability3'].argmax()) == label, target_kind='window'))
            else:
                if stage == 'ig':
                    owned = z['owned'].astype(bool)
                    for lab in np.unique(z['labels'][owned]):
                        ids = owned & (z['labels'] == lab)
                        c2 = np.log(np.clip(z['probability2'][ids], 1e-30, 1)).mean(0).argmax()
                        c3 = np.log(np.clip(z['probability3'][ids], 1e-30, 1)).mean(0).argmax()
                        out.append(dict(sample_id=r['sample_id'] + ':class' + str(lab), graph_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=int(lab), correct2=int(c2) == lab, correct3=int(c3) == lab, target_kind='class_mean_logits_of_owned_nuclei'))
                    continue
                for j in np.flatnonzero(z['owned']):
                    lab = int(z['labels'][j])
                    sid = r['sample_id'] + ':' + str(int(z['ids'][j]))
                    out.append(dict(sample_id=sid, graph_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=lab, index=int(j), nucleus_id=int(z['ids'][j]), correct2=int(z['probability2'][j].argmax()) == lab, correct3=int(z['probability3'][j].argmax()) == lab, target_kind='owned_nucleus' if stage == 'mask' else 'class_mean_logits_of_owned_nuclei'))
    if stage == 'ig' and a.dataset == 'lizard':
        dedup = {}
        for r in sorted(out, key=lambda r: stable(r['sample_id'])):
            dedup.setdefault((r['graph_id'], r['label_id']), r)
        out = list(dedup.values())
    if stage == 'ig' and split == 'test':
        return (out, rows)
    selected = set(select_cases(out))
    return ([r for r in out if r['sample_id'] in selected], rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['ig', 'mask'])
    p.add_argument('--dataset', required=True, choices=['arvaniti', 'lizard'])
    p.add_argument('--split', default='val')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(4)
    a = Adapter(args.dataset)
    folder = OUT / a.dataset / a.task / 'model'
    cases, rows = candidates(a, folder, args.split, args.stage)
    fingerprints = {k: digest(v) for k, v in a.paths.items()}
    if args.stage == 'ig' and args.split == 'test' and (not read(folder / 'ig_gate.json')['promote']):
        return
    write(folder / f'{args.stage}_cases_{args.split}.json', cases)
    from .matching import Matcher
    from .sparse import sparse
    matcher = Matcher(a)
    for number, c in enumerate(cases):
        if stable(c['sample_id']) % args.shards != args.shard:
            continue
        path = folder / args.stage / args.split / f"{stable(c['sample_id']):012x}.json"
        if path.exists():
            continue
        r = rows[c['graph_id']]
        f = a.forward(r)
        input_hash = hashlib.sha256(np.ascontiguousarray(f.data['rich']).tobytes()).hexdigest()
        target = TargetForward(f, c.get('index')) if args.stage == 'mask' else TargetForward(f, class_indices=np.flatnonzero(f.data['item']['owned'].astype(bool) & (f.data['item']['label'] == c['label_id']))) if a.dataset == 'lizard' else TargetForward(f)
        refs, coverage = matcher.references(f.data, c['cluster_id'], 4)
        record = dict(c, matched_nodes=int(coverage[0].sum()), valid_nodes=int(f.data['valid'].sum()), status='complete')
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.stage == 'ig':
            attrs = []
            diagnostics = []
            for ref in refs:
                attr, diag = integrated(target, torch.as_tensor(ref, device=a.device), c['label_id'])
                attrs.append(attr)
                diagnostics.append(diag)
            np.savez_compressed(path.with_suffix('.npz'), attribution=np.stack(attrs), matched=coverage)
            record.update(diagnostics=diagnostics, all_baselines_complete=all((d['complete'] for d in diagnostics)))
        else:
            masks, curves = sparse(target, torch.as_tensor(refs[0], device=a.device), torch.as_tensor(coverage[0], device=a.device))
            np.savez_compressed(path.with_suffix('.npz'), masks=masks, matched=coverage[0])
            record['curves'] = curves
        assert input_hash == hashlib.sha256(np.ascontiguousarray(f.rich.detach().cpu().numpy()).tobytes()).hexdigest(), 'Original analysis input mutated'
        record['input_sha256'] = input_hash
        write(path, record)
        if number % 10 == 0:
            print(args.dataset, args.stage, args.split, number, len(cases), flush=True)
    if args.stage == 'ig':
        expected = {c['sample_id'] for c in cases}
        rs = [read(p) for p in (folder / 'ig' / args.split).glob('*.json')]
        rs = [r for r in rs if r['sample_id'] in expected]
        gate = completeness_gate(rs, a.classes)
        gate.update(expected_units=len(expected), status='complete' if len(rs) == len(expected) else 'in_progress')
        gate['promote'] = gate['promote'] and len(rs) == len(expected)
        write(folder / ('ig_test_gate.json' if args.split == 'test' else 'ig_gate.json'), gate)
        if len(rs) == len(expected):
            write(folder / 'worker' / f'ig_{args.split}.json', dict(status='finished', completion='all expected case IDs have terminal records across shards', units=len(rs), shards=args.shards))
    assert fingerprints == {k: digest(v) for k, v in a.paths.items()}, 'Checkpoint mutated'
    assert all((not p.requires_grad and p.grad is None for m in a.models.values() for p in m.parameters()))
if __name__ == '__main__':
    main()
