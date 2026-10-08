from celllift.runtime import resource_path as _public_resource
from .common import *
from .prediction_summary import cm_metrics
import pandas as pd

def evaluate(y, original, changed, clusters, threshold=0.5):
    groups, gi = np.unique(clusters, return_inverse=True)
    ng = len(groups)
    k = original.shape[-1]
    repeats = len(changed)
    choose = lambda p: (p[..., 1] >= threshold).astype(int) if k == 2 else p.argmax(-1)
    base = choose(original)
    pred = choose(changed)
    cm0 = np.bincount(gi * k * k + y * k + base, minlength=ng * k * k).reshape(ng, k, k)
    cms = np.stack([np.bincount(gi * k * k + y * k + c, minlength=ng * k * k).reshape(ng, k, k) for c in pred])
    value0 = cm_metrics(cm0.sum(0))['macro_f1']
    value = cm_metrics(cms.sum(1))['macro_f1'].mean()
    estimates = {'macro_f1': float(value - value0)}
    boots = {'macro_f1': []}
    class_counts = np.stack([np.bincount(y[gi == g], minlength=k) for g in range(ng)])
    if k == 2:
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if not len(pos) or not len(neg):
            return dict(status='missing_class', clusters=ng)
        score = np.concatenate([original[None, :, 1], changed[:, :, 1]], 0)
        positive = score[:, pos][:, :, None]
        negative = score[:, neg][:, None, :]
        pairwise = (positive > negative) + 0.5 * (positive == negative)
        auc_kernel = pairwise[1:].mean(0) - pairwise[0]
        brier_delta = ((changed[:, :, 1] - y) ** 2).mean(0) - (original[:, 1] - y) ** 2
        estimates.update(auroc=float(auc_kernel.mean()), brier=float(brier_delta.mean()))
        boots.update(auroc=[], brier=[])
    rng = np.random.default_rng(42)
    rejected = 0
    for start in range(0, 10000, 128):
        w = rng.multinomial(ng, np.ones(ng) / ng, size=min(128, 10000 - start))
        ok = (w @ class_counts > 0).all(1)
        rejected += int((~ok).sum())
        w = w[ok]
        before = cm_metrics(np.einsum('bg,gij->bij', w, cm0))['macro_f1']
        after = cm_metrics(np.einsum('bg,rgij->brij', w, cms))['macro_f1'].mean(1)
        boots['macro_f1'].extend(after - before)
        if k == 2:
            u = w[:, gi]
            wp = u[:, pos]
            wn = u[:, neg]
            boots['auroc'].extend(np.sum(wp @ auc_kernel * wn, 1) / (wp.sum(1) * wn.sum(1)))
            boots['brier'].extend(u @ brier_delta / u.sum(1))
    return dict(version=4, status='complete' if ng > 1 else 'insufficient_independent_groups', units=len(y), clusters=ng, repeats=repeats, rejected_missing_class=rejected, metrics={m: dict(difference=v, ci=np.nanquantile(boots[m], [0.025, 0.975]) if ng > 1 and len(boots[m]) else [None, None], p=float((1 + (abs(np.asarray(boots[m]) - v) >= abs(v)).sum()) / (1 + len(boots[m]))) if ng > 1 and len(boots[m]) else None, estimable=len(boots[m])) for m, v in estimates.items()}, definition='mean over repetition-specific metric differences, not metric of averaged probabilities', bootstrap=10000, seed=42)

def main():
    for ds, tasks in TASKS.items():
        if ds in ('sicapv2', 'arvaniti'):
            continue
        for task in tasks:
            folder = OUT / ds / task / 'model'
            units = []
            for path in (folder / 'perturb/test').glob('*.json'):
                r = read(path)
                if r.get('status') != 'complete':
                    continue
                if ds == 'lizard':
                    with np.load(path.with_suffix('.npz')) as z:
                        owned = z['owned'].astype(bool)
                        methods = {m: z[m][:, owned] for m in ('training_matched', 'within_graph', 'image_matched', 'node_only', 'edge_only') if m in z.files}
                        units.append(dict(y=z['labels'][owned], p=z['probability3'][owned], group=np.repeat(r['cluster_id'], owned.sum()), methods=methods))
                else:
                    methods = {}
                    for v in r['perturbations']:
                        methods.setdefault(v['method'], []).append((v['repeat'], v['probability']))
                    methods = {m: np.asarray([p for j, p in sorted(v)])[:, None, :] for m, v in methods.items() if len(v) == 20}
                    units.append(dict(y=np.asarray([r['label_id']]), p=np.asarray(r['probability3'])[None], group=np.asarray([r['cluster_id']]), methods=methods, threshold=r.get('thresholds', {}).get('H3', 0.5)))
            if not units:
                continue
            previous = folder / 'summary/perturbation_task_metrics.json'
            old = read(previous) if previous.exists() else {}
            result = {}
            for method in sorted(set((m for u in units for m in u['methods']))):
                rows = [u for u in units if method in u['methods']]
                if method in old and old[method].get('version') == 4 and (old[method].get('units') == sum((len(u['y']) for u in rows))):
                    result[method] = old[method]
                    continue
                result[method] = evaluate(np.concatenate([u['y'] for u in rows]), np.concatenate([u['p'] for u in rows]), np.concatenate([u['methods'][method] for u in rows], 1), np.concatenate([u['group'] for u in rows]), rows[0].get('threshold', 0.5))
            independent = folder / 'summary/independent_edge_task_metrics.json'
            if independent.exists() and read(independent).get('version') == 4:
                result['independent_edge_matched'] = read(independent)
            hypotheses = [v for r in result.values() for v in r.get('metrics', {}).values()]
            for v, q in zip(hypotheses, bh([v.get('p', np.nan) for v in hypotheses])):
                v['q_task_perturbation_family'] = q
            write(folder / 'summary/perturbation_task_metrics.json', result)
            print(ds, task, len(units), flush=True)
if __name__ == '__main__':
    main()
