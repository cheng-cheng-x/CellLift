from celllift.runtime import resource_path as _public_resource
from .common import *
import pandas as pd
from scipy.stats import spearmanr

def intervals(frame, keys, metrics, weight='weight'):
    out = []
    for key, g in frame.groupby(keys, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        for metric in metrics:
            z = g[np.isfinite(g[metric])].copy()
            if z.empty:
                continue
            z['numerator'] = z[metric] * z[weight]
            s = z.groupby('cluster_id')[['numerator', weight]].sum()
            n = len(s)
            estimate = s.numerator.sum() / s[weight].sum()
            rng = np.random.default_rng(42)
            boot = []
            if n < 2:
                out.append(dict(zip(keys, key), metric=metric, estimate=estimate, ci_low=np.nan, ci_high=np.nan, p=np.nan, clusters=n, units=len(z), cluster_equal_mean=estimate, lopo_sign_agreement=np.nan, status='insufficient_independent_groups'))
                continue
            for start in range(0, 10000, 256):
                w = rng.multinomial(n, np.ones(n) / n, size=min(256, 10000 - start))
                boot.extend((w @ s.numerator / (w @ s[weight])).tolist())
            ci = np.quantile(boot, [0.025, 0.975])
            p = (1 + (abs(np.asarray(boot) - estimate) >= abs(estimate)).sum()) / 10001
            if metric not in ('original_probability_change', 'true_probability_change', 'error_improvement', 'drop'):
                p = np.nan
            delete = (s.numerator.sum() - s.numerator) / (s[weight].sum() - s[weight])
            out.append(dict(zip(keys, key), metric=metric, estimate=estimate, ci_low=ci[0], ci_high=ci[1], p=p, clusters=n, units=len(z), cluster_equal_mean=float((s.numerator / s[weight]).mean()), lopo_sign_agreement=float(np.mean(np.sign(delete) == np.sign(estimate))) if n > 1 else None))
    result = pd.DataFrame(out)
    if len(result):
        result['q'] = bh(result.p)
    return result

def main():
    for ds, tasks in TASKS.items():
        if ds == 'sicapv2':
            continue
        for task in ['local'] if ds == 'arvaniti' else tasks:
            folder = OUT / ds / task / 'model'
            dest = folder / 'summary'
            dest.mkdir(exist_ok=True, parents=True)
            pert = []
            curves = []
            stability = []
            attrs = []
            for path in (folder / 'perturb/test').glob('*.json'):
                r = read(path)
                if r.get('status') != 'complete':
                    continue
                if ds in ('arvaniti', 'lizard'):
                    with np.load(path.with_suffix('.npz')) as z:
                        original = z['probability3']
                        label = z['labels']
                        owned = z['owned'].astype(bool) if ds == 'lizard' else np.ones(1, bool)
                        if original.ndim == 1:
                            original = original[None]
                            label = np.asarray(label).reshape(-1)
                        original = original[owned]
                        label = np.asarray(label)[owned]
                        choices = original.argmax(1)
                        weight = len(original)
                        for method in ('training_matched', 'within_graph', 'image_matched', 'node_only', 'edge_only'):
                            if method not in z.files:
                                continue
                            arr = z[method]
                            if arr.ndim == 2:
                                arr = arr[:, None, :]
                            for j, p in enumerate(arr[:, owned]):
                                good = label >= 0
                                pert.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], method=method, repeat=j, weight=weight, coverage=r['coverage'][method][j], kl=float(np.mean(np.sum(original * (np.log(np.clip(original, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1))), 1))), original_probability_change=float(np.mean(p[np.arange(weight), choices] - original[np.arange(weight), choices])), true_probability_change=float(np.mean(p[np.flatnonzero(good), label[good]] - original[np.flatnonzero(good), label[good]])) if good.any() else np.nan, error_improvement=float(np.mean((choices[good] != label[good]).astype(float) - (p[good].argmax(1) != label[good]))) if good.any() else np.nan))
                else:
                    original = np.asarray(r['probability3'])
                    label = r['label_id']
                    threshold = r.get('thresholds', {}).get('H3', 0.5)
                    choice = int(original[1] >= threshold) if len(original) == 2 else original.argmax()
                    for row in r['perturbations']:
                        p = np.asarray(row['probability'])
                        pred = int(p[1] >= threshold) if len(p) == 2 else p.argmax()
                        pert.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], method=row['method'], repeat=row['repeat'], weight=1, coverage=row['coverage'], kl=row['kl'], original_probability_change=p[choice] - original[choice], true_probability_change=p[label] - original[label], error_improvement=float(choice != label) - float(pred != label)))
            if pert:
                f = pd.DataFrame(pert)
                f.to_csv(dest / 'perturbation_repeats.csv', index=False)
                cols = ['coverage', 'kl', 'original_probability_change', 'true_probability_change', 'error_improvement']
                collapsed = f.groupby(['sample_id', 'cluster_id', 'method'])[['weight'] + cols].mean().reset_index()
                collapsed.to_csv(dest / 'perturbation_units.csv', index=False)
                intervals(collapsed, ['method'], cols).to_csv(dest / 'perturbation_intervals.csv', index=False)
            for path in list((folder / 'mask/test').glob('*.json')) + list((folder / 'ig_followup/test').glob('*.json')):
                r = read(path)
                if r.get('status') != 'complete':
                    continue
                for v in r['curves']:
                    curves.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], method=v['method'], seed=v['seed'], fraction=v['fraction'], kl=v['kl'], probability_drop=v['probability_drop'], same_class=float(v['same_class']), faithful=float(v['faithful']), weight=1))
                if 'ig_followup' in path.parts:
                    continue
                with np.load(path.with_suffix('.npz')) as z:
                    m = z['masks']
                    valid = z['matched'].astype(bool)
                    m = m[:, valid]
                    ids = np.arange(m.shape[1])
                    keep = int(np.ceil(0.1 * len(ids)))
                    for a, b in ((0, 1), (0, 2), (1, 2)):
                        left = set(np.argsort(m[a])[-keep:]) if keep else set()
                        right = set(np.argsort(m[b])[-keep:]) if keep else set()
                        stability.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], seed_a=[17, 42, 73][a], seed_b=[17, 42, 73][b], eligible_nodes=len(ids), spearman=float(spearmanr(m[a], m[b]).statistic) if len(ids) > 1 else np.nan, jaccard10=len(left & right) / len(left | right) if left | right else np.nan))
            if curves:
                f = pd.DataFrame(curves)
                f.to_csv(dest / 'mask_seed_curves.csv', index=False)
                cols = ['kl', 'probability_drop', 'same_class', 'faithful']
                c = f.groupby(['sample_id', 'cluster_id', 'method', 'fraction'])[['weight'] + cols].mean().reset_index()
                intervals(c[c.method != 'IG'], ['method', 'fraction'], cols).to_csv(dest / 'mask_intervals.csv', index=False)
                pd.DataFrame(stability).to_csv(dest / 'mask_stability.csv', index=False)
                eligible = set(c.loc[c.method == 'IG', 'sample_id'])
                if eligible:
                    intervals(c[c.sample_id.isin(eligible)], ['method', 'fraction'], cols).to_csv(dest / 'mask_ig_matched_intervals.csv', index=False)
            for path in (folder / 'ig/test').glob('*.json'):
                r = read(path)
                if not r.get('all_baselines_complete', False):
                    continue
                with np.load(path.with_suffix('.npz')) as z:
                    v = z['attribution']
                    valid = z['valid'].astype(bool) if 'valid' in z.files else z['matched'].any(0)
                    den = max(1, valid.sum())
                    for j in range(v.shape[-1]):
                        attrs.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=r['label_id'], channel=j, signed=float(v[:, :, j].sum(1).mean()), absolute_per_valid_node=float(abs(v[:, :, j]).sum(1).mean() / den), baseline_sd=float(v[:, :, j].sum(1).std()), weight=1))
            if attrs:
                f = pd.DataFrame(attrs)
                f.to_csv(dest / 'ig_unit_channels.csv', index=False)
                f.groupby(['cluster_id', 'label_id', 'channel'])[['signed', 'absolute_per_valid_node', 'baseline_sd']].mean().reset_index().to_csv(dest / 'ig_cluster_channels.csv', index=False)
                intervals(f, ['label_id', 'channel'], ['signed', 'absolute_per_valid_node', 'baseline_sd']).to_csv(dest / 'ig_channel_intervals.csv', index=False)
            ranked = []
            for path in (folder / 'ig_followup/test').glob('*.json'):
                r = read(path)
                for v in r.get('ranked_replacements', []):
                    ranked.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], method=v['method'], fraction=v['fraction'], repeat=v['repeat'], drop=v['original_probability_drop'], kl=v['kl']))
            if ranked:
                f = pd.DataFrame(ranked)
                f.to_csv(dest / 'ig_ranked_replacements.csv', index=False)
                c = f.groupby(['sample_id', 'cluster_id', 'method', 'fraction'])[['drop', 'kl']].mean().reset_index()
                pivot = c.pivot(index=['sample_id', 'cluster_id', 'fraction'], columns='method', values=['drop', 'kl'])
                delta = pd.DataFrame({key: pivot[key, 'IG'] - pivot[key, 'matched_random'] for key in ('drop', 'kl')}).reset_index()
                delta['weight'] = 1
                intervals(delta, ['fraction'], ['drop', 'kl']).to_csv(dest / 'ig_ranked_vs_random_intervals.csv', index=False)
            write(dest / 'explanation_coverage.json', dict(perturbation_units=len({r['sample_id'] for r in pert}), mask_units=len({r['sample_id'] for r in curves}), ig_complete_units=len({r['sample_id'] for r in attrs}), repeats_are_not_independent=True, zero_percent='matched reference without restoring original inputs; not removal of 3D', ig_ranking='Only permissible after development/test completeness coverage gates; channel sums are not independent biological effects'))
if __name__ == '__main__':
    main()
