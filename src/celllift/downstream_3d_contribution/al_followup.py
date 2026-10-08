from celllift.runtime import resource_path as _public_resource
from .common import *
from .al_model import Adapter
from .al_explain import TargetForward
from .model import integrated
from .matching import Matcher
import argparse, torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['arvaniti', 'lizard'], required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    a = Adapter(args.dataset)
    folder = OUT / a.dataset / a.task / 'model'
    if not read(folder / 'ig_gate.json')['promote']:
        return
    matcher = Matcher(a)
    rows = {r['sample_id']: r for r in a.rows if r['split'] == 'test'}
    cases = [read(p) for p in sorted((folder / 'mask/test').glob('*.json'))]
    cases = [r for r in cases if r.get('status') == 'complete']
    local_ids = set()
    seen = set()
    for r in cases:
        key = (r['label_id'], r['correct2'], r['correct3'])
        if key not in seen and len(local_ids) < 12:
            seen.add(key)
            local_ids.add(r['sample_id'])
    for r in cases:
        path = folder / 'ig_followup/test' / f"{stable(r['sample_id']):012x}.json"
        if path.exists():
            continue
        f = a.forward(rows[r['graph_id']])
        target = TargetForward(f, r.get('index'))
        refs, cover = matcher.references(f.data, r['cluster_id'], 4)
        attrs = []
        diagnostics = []
        for ref in refs:
            value, diag = integrated(target, torch.as_tensor(ref, device=a.device), r['label_id'])
            attrs.append(value)
            diagnostics.append(diag)
        complete = all((d['complete'] for d in diagnostics))
        curves = []
        local = []
        payload = dict(true_class=np.stack(attrs))
        if complete:
            score = np.abs(np.stack(attrs)).sum(-1).mean(0)
            ids = np.flatnonzero(cover[0])
            order = ids[np.argsort(-score[ids], kind='stable')]
            with torch.no_grad():
                original = torch.softmax(target(target.rich), -1)[0]
                choice = int(original.argmax())
            for fraction in (0, 0.05, 0.1, 0.2, 0.4, 1):
                x = torch.as_tensor(refs[0], device=a.device).clone()
                keep = order[:int(np.ceil(fraction * len(ids)))]
                x[keep] = target.rich[keep]
                with torch.no_grad():
                    prob = torch.softmax(target(x), -1)[0]
                kl = float((original * (original.clamp_min(1e-12).log() - prob.clamp_min(1e-12).log())).sum())
                drop = float(original[choice] - prob[choice])
                same = int(prob.argmax()) == choice
                curves.append(dict(method='IG', seed=42, fraction=fraction, probability=prob.cpu().numpy(), kl=kl, probability_drop=drop, same_class=same, faithful=same and kl <= 0.02 and (drop <= 0.05)))
            if r['sample_id'] in local_ids:
                other = r['label_id'] if choice != r['label_id'] else int(np.argsort(original.cpu().numpy())[-2])
                for name, label in [('original_class', choice), ('confusion_logit_difference', (choice, other))]:
                    av = []
                    dv = []
                    for ref in refs:
                        value, diag = integrated(target, torch.as_tensor(ref, device=a.device), label)
                        av.append(value)
                        dv.append(diag)
                    payload[name] = np.stack(av)
                    local.append(dict(target=name, classes=label, diagnostics=dv))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix('.npz'), **payload)
        write(path, dict(status='complete' if complete else 'numerical_failure', sample_id=r['sample_id'], graph_id=r['graph_id'], cluster_id=r['cluster_id'], label_id=r['label_id'], target_kind='owned_nucleus' if a.dataset == 'lizard' else 'window', all_baselines_complete=complete, diagnostics=diagnostics, curves=curves, local_targets=local))
    assert all((not p.requires_grad and p.grad is None for m in a.models.values() for p in m.parameters()))
if __name__ == '__main__':
    main()
