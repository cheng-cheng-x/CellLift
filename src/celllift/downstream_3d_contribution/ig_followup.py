from celllift.runtime import resource_path as _public_resource
from .common import *
from .model import adapter, integrated, completeness_gate, predicted_class
from .matching import Matcher
from .batched_forward import probabilities
import argparse, torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--task')
    args = parser.parse_args()
    torch.set_num_threads(4)
    a = adapter(args.dataset, args.task)
    folder = OUT / a.dataset / a.task / 'model'
    expected = {r['sample_id'] for r in a.rows if r['split'] == 'test'}
    records = [read(p) for p in (folder / 'ig/test').glob('*.json')]
    records = [r for r in records if r['sample_id'] in expected]
    gate = completeness_gate(records, a.classes)
    gate.update(expected_units=len(expected), status='complete' if len(records) == len(expected) else 'in_progress')
    gate['promote'] = gate['promote'] and len(records) == len(expected)
    write(folder / 'ig_test_gate.json', gate)
    if not gate['promote']:
        return
    matcher = Matcher(a)
    rows = {r['sample_id']: r for r in a.rows if r['split'] == 'test'}
    selected = []
    for path in sorted((folder / 'mask/test').glob('*.json')):
        r = read(path)
        ig = folder / 'ig/test' / path.name
        if r.get('status') == 'complete' and ig.exists() and read(ig).get('all_baselines_complete'):
            selected.append((path, r, ig))
    local_pairs = set()
    local_ids = set()
    for _, r, _ in selected:
        key = (r['label_id'], predicted_class(np.asarray(r['probability3']), a))
        if key not in local_pairs and len(local_ids) < 12:
            local_pairs.add(key)
            local_ids.add(r['sample_id'])
    write(folder / 'ig_followup/local_cases.json', sorted(local_ids))
    for path, r, ig in selected:
        dest = folder / 'ig_followup/test' / path.name
        if dest.exists():
            continue
        f = a.forward(rows[r['sample_id']])
        refs, cover = matcher.references(f.data, r['cluster_id'], 20)
        eligible = cover[0]
        with np.load(ig.with_suffix('.npz')) as z:
            score = np.abs(z['attribution']).sum(-1).mean(0)
        ids = np.flatnonzero(eligible)
        order = ids[np.argsort(-score[ids], kind='stable')]
        _, distance = matcher.query(matcher.raw(f.data), r['cluster_id'])
        d = distance[:, 0]
        cuts = np.quantile(d[ids], [0.25, 0.5, 0.75]) if len(ids) else np.zeros(3)
        strata = np.searchsorted(cuts, d)
        with torch.no_grad():
            original = torch.softmax(f(f.rich), -1)[0]
            choice = predicted_class(original.cpu().numpy(), a)
        ranked = []
        curves = []
        pending = []
        verification = dict(max_probability_error=0.0, checked_inputs=0)

        def flush():
            if not pending:
                return
            inputs = np.stack([item[0] for item in pending])
            predictions = probabilities(f, inputs)
            if not verification['checked_inputs'] and len(inputs) > 1:
                with torch.no_grad():
                    serial = [torch.softmax(f(torch.as_tensor(x, device=f.rich.device)), -1)[0] for x in inputs]
                error = max((float((p - q).abs().max()) for p, q in zip(predictions, serial)))
                assert error <= 1e-05, f'Local ranked replacement batch mismatch: {error}'
                verification.update(max_probability_error=error, checked_inputs=len(inputs))
            for (_, metadata), p in zip(pending, predictions):
                ranked.append(dict(**metadata, probability=p.cpu().numpy(), original_probability_drop=float(original[choice] - p[choice]), kl=float((original * (original.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())))
            pending.clear()
        for fraction in (0.1, 0.2):
            top = order[:int(np.ceil(fraction * len(ids)))]
            counts = np.bincount(strata[top], minlength=4)
            for repeat, reference in enumerate(refs):
                rng = np.random.default_rng(42 + repeat)
                random = np.concatenate([rng.choice(ids[strata[ids] == j], n, replace=False) for j, n in enumerate(counts)]) if len(ids) else ids
                for method, keep in [('IG', top), ('matched_random', random)]:
                    x = f.data['rich'].copy()
                    x[keep] = reference[keep]
                    pending.append((x, dict(method=method, fraction=fraction, repeat=repeat, nodes=len(keep))))
                    if len(pending) >= max(1, getattr(f, 'perturbation_batch_size', 1)):
                        flush()
        flush()
        for fraction in (0, 0.05, 0.1, 0.2, 0.4, 1):
            x = torch.as_tensor(refs[0], device=f.rich.device).clone()
            keep = order[:int(np.ceil(fraction * len(ids)))]
            x[keep] = f.rich[keep]
            with torch.no_grad():
                p = torch.softmax(f(x), -1)[0]
            kl = float((original * (original.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())
            drop = float(original[choice] - p[choice])
            same = predicted_class(p.cpu().numpy(), a) == choice
            curves.append(dict(method='IG', seed=42, fraction=fraction, probability=p.cpu().numpy(), kl=kl, probability_drop=drop, same_class=same, faithful=same and kl <= 0.02 and (drop <= 0.05)))
        local = []
        if r['sample_id'] in local_ids:
            runner = int(np.argsort(original.cpu().numpy())[-2])
            target = r['label_id'] if r['label_id'] != choice else runner
            payload = {}
            for label, target_id in [('original_class', choice), ('confusion_logit_difference', (choice, target))]:
                attrs = []
                diags = []
                for reference in refs[:4]:
                    v, diag = integrated(f, torch.as_tensor(reference, device=f.rich.device), target_id)
                    attrs.append(v)
                    diags.append(diag)
                payload[label] = np.stack(attrs)
                local.append(dict(target=label, target_class=target_id, diagnostics=diags))
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest.with_suffix('.npz'), **payload)
        write(dest, dict(status='complete', sample_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=r['label_id'], eligible_nodes=len(ids), ranked_replacements=ranked, curves=curves, local_targets=local, random_control='same count in each quartile of nearest matched-reference distance', original_probability=original.cpu().numpy(), batch_equivalence=verification))
    assert all((not p.requires_grad and p.grad is None for m in a.models.values() for module in (m if isinstance(m, tuple) else (m,)) for p in module.parameters()))
if __name__ == '__main__':
    main()
