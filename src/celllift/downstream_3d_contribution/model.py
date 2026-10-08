from celllift.runtime import resource_path as _public_resource
from .common import *
import argparse
import time
import torch
import pandas as pd
from functools import lru_cache

@lru_cache(maxsize=4)
def adapter(dataset, task=None):
    if dataset == 'bracs':
        from .bracs_model import Adapter
        return Adapter()
    if dataset == 'tcga_brca':
        from .brca_model import Adapter
        return Adapter(task)
    if dataset == 'tcga_crc_msi':
        from .crc_model import Adapter
        return Adapter()
    raise NotImplementedError(f'Frozen adapter not yet implemented: {dataset}/{task}')

def target(logits, label):
    if isinstance(label, tuple):
        return logits[:, label[0]] - logits[:, label[1]]
    other = [i for i in range(logits.shape[-1]) if i != label]
    return logits[:, label] - torch.logsumexp(logits[:, other], dim=-1)

def predicted_class(prob, adapter, arm='H3'):
    if len(prob) == 2:
        return int(prob[1] >= getattr(adapter, 'thresholds', {}).get(arm, 0.5))
    return int(np.argmax(prob))

def completeness_gate(records, classes):
    rates = {}

    def passed(r):
        return r.get('status') == 'complete' and r.get('all_baselines_complete', False)
    for lab in range(classes):
        z = [r for r in records if r['label_id'] == lab]
        rates[str(lab)] = dict(n=len(z), pass_rate=float(np.mean([passed(r) for r in z])) if z else None)
    rate = float(np.mean([passed(r) for r in records])) if records else 0
    return dict(promote=rate >= 0.95 and all((v['pass_rate'] >= 0.9 for v in rates.values() if v['n'] >= 10)), overall=rate, classes=rates, class_minimum_n=10, units=len(records))

def integrated(f, ref, label):
    delta = f.rich - ref
    with torch.no_grad():
        difference = float((target(f(f.rich), label) - target(f(ref), label)).item())
    total = None
    evaluations = 0
    for steps in (64, 128, 256):
        points = list(range(steps + 1)) if total is None else list(range(1, steps, 2))
        total = torch.zeros_like(delta) if total is None else total * 0.5
        batch = max(1, int(getattr(f, 'integration_batch_size', 1)))
        for first in range(0, len(points), batch):
            js = points[first:first + batch]
            evaluations += len(js)
            if batch == 1:
                j = js[0]
                x = (ref + j / steps * delta).detach().requires_grad_(True)
                grad = torch.autograd.grad(target(f(x), label).sum(), x)[0]
                total += grad * ((0.5 if j in (0, steps) else 1) / steps)
            else:
                alpha = delta.new_tensor(js)[:, None, None] / steps
                x = (ref[None] + alpha * delta[None]).detach().requires_grad_(True)
                grad = torch.autograd.grad(target(f(x), label).sum(), x)[0]
                weight = delta.new_tensor([0.5 if j in (0, steps) else 1 for j in js]) / steps
                total += (grad * weight[:, None, None]).sum(0)
        attr = total * delta
        error = abs(float(attr.sum()) - difference)
        tol = 0.0001 + 0.01 * abs(difference)
        if error <= tol:
            break
    return (attr.detach().cpu().numpy(), dict(steps=steps, error=error, tolerance=tol, complete=error <= tol, output_difference=difference, gradient_points_evaluated=evaluations, quadrature='nested trapezoid; unchanged nodes and weights'))

def select_cases(records):
    groups = {}
    for r in records:
        a, b = (r['correct2'], r['correct3'])
        group = 'both_correct' if a and b else 'corrected' if b else 'harmed' if a else 'both_wrong'
        groups.setdefault((r['label_id'], group), []).append(r)
    pools = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: stable(r['sample_id']))
        seen = set()
        unique = []
        others = []
        for r in rs:
            if r['cluster_id'] not in seen:
                unique.append(r)
                seen.add(r['cluster_id'])
            else:
                others.append(r)
        pools.append((unique + others)[:8])
    selected = []
    for j in range(8):
        for rs in pools:
            if j < len(rs) and len(selected) < 128:
                selected.append(rs[j]['sample_id'])
    return selected

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['predict', 'perturb', 'ig', 'mask'])
    p.add_argument('--dataset', required=True)
    p.add_argument('--task')
    p.add_argument('--split', default='test')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(4)
    a = adapter(args.dataset, args.task)
    folder = OUT / a.dataset / a.task / 'model'
    folder.mkdir(parents=True, exist_ok=True)
    fingerprints = {k: digest(v) for k, v in a.paths.items()}
    write(folder / 'input_paths.json', dict(checkpoints={k: str(v) for k, v in a.paths.items()}, sha256=fingerprints, protocol='official_seed42' if a.dataset == 'bracs' else 'corrected_routes_v2_seed42', direct_3d=['12 node descriptors' if a.dataset == 'bracs' else 'first 9 node descriptors encoded as synchronized nucleus/cell tails'], not_direct_3d=['shape matrix', '3D edges', '3D centers'], comparison='geometry branch substitution', parameters_frozen=True, image_matching='frozen node DINO; BRCA image MIL still receives original tile-global DINO'))
    if a.dataset == 'tcga_crc_msi':
        write(folder / 'input_paths.json', dict(checkpoints={k: str(v) for k, v in a.paths.items()}, sha256=fingerprints, protocol='official_seed42_B3_vs_B2', direct_3d=['12 node descriptors', '8 incoming edge relations'], indirect=['shape transforms and centered axial positions through regenerated relations'], comparison='prototype-conditioned geometry replacement', parameters_frozen=True, image_matching='node DINO'))
    rows = [r for r in a.rows if r['split'] == args.split]
    if args.stage == 'predict':
        matcher = None
    else:
        from .matching import Matcher
        matcher = Matcher(a)
        sensitivity = Matcher(a, image=True) if args.stage == 'perturb' else None
        records = [read(path) for path in (folder / 'predict' / args.split).glob('*.json')]
        records = [r for r in records if r.get('status') == 'complete']
        for r in records:
            r['correct2'] = predicted_class(r['probability2'], a, 'H2') == r['label_id']
            r['correct3'] = predicted_class(r['probability3'], a) == r['label_id']
        if args.stage in ('mask', 'ig'):
            selected = select_cases(records)
            write(folder / f'explanation_cases_{args.split}.json', selected)
            rows = [r for r in rows if r['sample_id'] in selected]
        if args.stage == 'ig' and args.split == 'test':
            gate = read(folder / 'ig_gate.json')
            if not gate['promote']:
                print('IG development gate failed; perturbation and masks remain independent', flush=True)
                return
            rows = [r for r in a.rows if r['split'] == 'test']
    for number, r in enumerate(rows):
        if stable(r['sample_id']) % args.shards != args.shard:
            continue
        name = f"{stable(r['sample_id']):012x}"
        path = folder / args.stage / args.split / (name + '.json')
        previous = read(path) if path.exists() else None
        supplement = previous and a.dataset == 'tcga_crc_msi' and (args.stage == 'perturb') and (previous.get('status') == 'complete') and (not any((v['method'] == 'edge_only' for v in previous.get('perturbations', []))))
        if previous and (not supplement):
            continue
        started = time.perf_counter()
        record = {k: r[k] for k in ('sample_id', 'split', 'cluster_id', 'label_id')}
        if a.dataset == 'bracs' and (not r['graph_ids']):
            write(path, dict(record, status='missing_resource', reason='ROI has no original cached graph'))
            continue
        f = a.forward(r)
        input_hash = hashlib.sha256(np.ascontiguousarray(f.data['rich']).tobytes()).hexdigest()
        record['input_sha256'] = input_hash
        with torch.no_grad():
            logits = f(f.rich)
            prob = torch.softmax(logits, -1)[0]
            p2 = torch.softmax(f(f.rich, arm='H2'), -1)[0]
        c3 = predicted_class(prob.cpu().numpy(), a)
        c2 = predicted_class(p2.cpu().numpy(), a, 'H2')
        record.update(probability3=prob.cpu().numpy(), probability2=p2.cpu().numpy(), class3=c3, class2=c2, correct3=c3 == r['label_id'], correct2=c2 == r['label_id'], thresholds=getattr(a, 'thresholds', {}))
        errors = {}
        for arm, pred in [('H3', prob), ('H2', p2)]:
            expected = a.saved[arm].get((args.split, str(r['sample_id'])))
            if expected is not None:
                errors[arm] = dict(max_absolute_error=float(np.max(np.abs(pred.cpu().numpy() - expected))), class_equal=int(pred.argmax()) == int(expected.argmax()))
        record['reproduction'] = errors
        if any((v['max_absolute_error'] > 1e-05 or not v['class_equal'] for v in errors.values())):
            write(path, dict(record, status='numerical_failure', reason='saved probability reproduction'))
            continue
        if args.split == 'test' and len(errors) != 2:
            write(path, dict(record, status='missing_resource', reason='saved comparison prediction unavailable'))
            continue
        if args.stage == 'predict':
            with torch.no_grad():
                identity = float((f(f.rich.clone()) - logits).abs().max())
                invariance = float((f(torch.zeros_like(f.rich), arm='H2') - f(f.rich, arm='H2')).abs().max())
            record.update(identity_max_error=identity, unused_3d_H2_max_error=invariance)
        else:
            refs, covers = matcher.references(f.data, r['cluster_id'], 20 if args.stage == 'perturb' else 4)
            ref = torch.as_tensor(refs[0], device=f.rich.device)
            record.update(valid_nodes=int(f.data['valid'].sum()), nodes=len(f.rich), matched_nodes=int(covers[0].sum()))
            if args.stage == 'perturb':
                records = list(previous['perturbations']) if supplement else []
                methods = []
                if not supplement:
                    per, pc = matcher.permutations(f.data)
                    im, ic = sensitivity.references(f.data, r['cluster_id'])
                    methods = [('training_matched', refs, covers, 'consistent'), ('within_graph', per, pc, 'consistent'), ('image_matched', im, ic, 'consistent')]
                if a.dataset == 'tcga_crc_msi':
                    methods.extend([('node_only', refs, covers, 'node_only'), ('edge_only', refs, covers, 'edge_only')])
                from .batched_forward import probabilities
                for method, inputs, coverage, way in methods:
                    predictions = probabilities(f, inputs, **{'pathway': way} if a.dataset == 'tcga_crc_msi' else {})
                    for j, (pnew, c) in enumerate(zip(predictions, coverage)):
                        records.append(dict(method=method, repeat=j, probability=pnew.cpu().numpy(), changed_nodes=int(c.sum()), coverage=float(c.sum() / max(1, f.data['valid'].sum())), kl=float((prob * (prob.clamp_min(1e-12).log() - pnew.clamp_min(1e-12).log())).sum())))
                record['perturbations'] = records
                with torch.no_grad():
                    _, old = f(f.rich, trace=True)
                    _, new = f(ref, trace=True)
                    record['path'] = {}
                    for key in old:
                        rms = float(old[key].square().mean().sqrt())
                        change = float((new[key] - old[key]).square().mean().sqrt())
                        record['path'][key] = dict(original_rms=rms, absolute_rms_change=change, relative_change=change / rms if rms > 1e-08 else None, near_zero=rms <= 1e-08)
            elif args.stage == 'ig':
                diagnostics = []
                attrs = []
                for reference in refs:
                    attr, diag = integrated(f, torch.as_tensor(reference, device=f.rich.device), r['label_id'])
                    diagnostics.append(diag)
                    attrs.append(attr)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'), attribution=np.stack(attrs), valid=f.data['valid'], matched=covers)
                record.update(diagnostics=diagnostics, all_baselines_complete=all((d['complete'] for d in diagnostics)))
            else:
                from .sparse import sparse
                valid = torch.as_tensor(covers[0], device=f.rich.device)
                masks, curves = sparse(f, ref, valid)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'), masks=masks, matched=covers[0])
                record['curves'] = curves
        assert input_hash == hashlib.sha256(np.ascontiguousarray(f.rich.detach().cpu().numpy()).tobytes()).hexdigest(), 'Original analysis input mutated'
        record['engineering'] = dict(elapsed_seconds=time.perf_counter() - started, perturbation_batch_size=getattr(f, 'perturbation_batch_size', 1), batch_probability_error=getattr(f, 'batch_equivalence_error', None), nested_ig_quadrature=True)
        write(path, dict(record, status='complete'))
        if number % 20 == 0:
            print(args.stage, args.split, number, len(rows), flush=True)
    if args.stage == 'ig':
        expected = {r['sample_id'] for r in rows}
        records = [read(p) for p in (folder / 'ig' / args.split).glob('*.json')]
        records = [r for r in records if r['sample_id'] in expected]
        gate = completeness_gate(records, a.classes)
        gate.update(expected_units=len(expected), status='complete' if len(records) == len(expected) else 'in_progress')
        gate['promote'] = gate['promote'] and len(records) == len(expected)
        write(folder / ('ig_test_gate.json' if args.split == 'test' else 'ig_gate.json'), gate)
    assert all((not p.requires_grad and p.grad is None for m in a.models.values() for module in (m if isinstance(m, tuple) else (m,)) for p in module.parameters()))
    assert fingerprints == {k: digest(v) for k, v in a.paths.items()}, 'Checkpoint mutated'
if __name__ == '__main__':
    main()
