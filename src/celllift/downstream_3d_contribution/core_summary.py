from celllift.runtime import resource_path as _public_resource
from .common import *
import sys
sys.path.insert(0, str(PUBLIC))
import pandas as pd
from scipy.special import softmax
from .prediction_summary import cm_metrics

def perturbation_metrics(records, perturb, folder):
    table = pd.DataFrame(perturb)
    index = {r['core_id']: r for r in records}
    rows = []
    behaviors = []
    if table.empty:
        write(folder / 'perturbation_metrics_status.json', dict(status='pending', complete_cores=0))
        return
    for method, data in table.groupby('method'):
        ids = sorted((c for c, g in data.groupby('core_id') if c in index and set(g['repeat']) == set(range(20))))
        if not ids:
            continue
        predictions = data.pivot(index='core_id', columns='repeat', values='qwk_label').loc[ids, range(20)].to_numpy(int)
        n = len(ids)
        weights = np.random.default_rng(42).multinomial(n, np.ones(n) / n, size=10000)
        reader_est = []
        reader_boot = []
        for reader in ('pathologist1', 'pathologist2'):
            y = np.asarray([index[c]['gt'][reader]['qwk_label'] for c in ids], int)
            original = np.asarray([index[c]['AS']['qwk_label'] for c in ids], int)
            base = np.eye(36)[y * 6 + original].reshape(n, 6, 6)
            changed = np.eye(36)[y[:, None] * 6 + predictions].reshape(n, 20, 6, 6)
            base_q = cm_metrics(base.sum(0))['qwk']
            q = cm_metrics(changed.sum(0))['qwk']
            estimate = float(np.mean(q - base_q))
            draws = []
            for start in range(0, len(weights), 100):
                w = weights[start:start + 100]
                b = cm_metrics(np.einsum('bg,gij->bij', w, base))['qwk']
                p = cm_metrics(np.einsum('bg,grij->brij', w, changed))['qwk']
                valid = np.isfinite(p).all(1) & np.isfinite(b) & (w @ np.eye(6)[y] > 0).all(1)
                v = np.full(len(w), np.nan)
                v[valid] = np.mean(p[valid] - b[valid, None], axis=1)
                draws.append(v)
            boot = np.concatenate(draws)
            reader_est.append(estimate)
            reader_boot.append(boot)
            valid = np.isfinite(boot)
            rows.append(dict(method=method, reader=reader, metric='qwk_change', estimate=estimate, ci_low=float(np.nanquantile(boot, 0.025)), ci_high=float(np.nanquantile(boot, 0.975)), complete_cores=n, bootstrap_estimable=int(valid.sum()), bootstrap_not_estimable=int((~valid).sum()), repeats=20))
            for i, c in enumerate(ids):
                for j in range(20):
                    behaviors.append(dict(core_id=c, reader=reader, method=method, repeat=j, label=int(y[i]), original=int(original[i]), perturbed=int(predictions[i, j]), group=('both_correct' if original[i] == y[i] else 'corrected') if predictions[i, j] == y[i] else 'harmed' if original[i] == y[i] else 'both_wrong', squared_error_improvement=float((original[i] - y[i]) ** 2 - (predictions[i, j] - y[i]) ** 2)))
        boot = (reader_boot[0] + reader_boot[1]) / 2
        valid = np.isfinite(boot)
        rows.append(dict(method=method, reader='mean_readers', metric='qwk_change', estimate=float(np.mean(reader_est)), ci_low=float(np.nanquantile(boot, 0.025)), ci_high=float(np.nanquantile(boot, 0.975)), complete_cores=n, bootstrap_estimable=int(valid.sum()), bootstrap_not_estimable=int((~valid).sum()), repeats=20))
    pd.DataFrame(rows).to_csv(folder / 'perturbation_qwk_intervals.csv', index=False)
    pd.DataFrame(behaviors).to_csv(folder / 'perturbation_reader_changes.csv', index=False)
    write(folder / 'perturbation_metrics_status.json', dict(status='complete' if len(set(table.core_id)) == len(records) else 'partial_coverage', expected_cores=len(records), complete_cores=len(set(table.core_id)), unit='core; patient identity unavailable in fixed metadata', reader_labels_resampled_together=True, aggregation='mean of 20 repeat-specific QWK changes; never QWK of averaged probabilities'))

def paired_metrics(records, folder):
    n = len(records)
    k = 6
    rng = np.random.default_rng(42)
    w = rng.multinomial(n, np.ones(n) / n, size=10000)
    est = {}
    boot = {}
    for reader in ('pathologist1', 'pathologist2'):
        y = np.asarray([r['gt'][reader]['qwk_label'] for r in records])
        summaries = []
        bootstrap = []
        for arm in ('A2', 'AS'):
            pred = np.asarray([r[arm]['qwk_label'] for r in records])
            m = np.eye(k * k)[y * k + pred].reshape(n, k, k)
            summaries.append(cm_metrics(m.sum(0)))
            bootstrap.append(cm_metrics(np.einsum('bg,gij->bij', w, m)))
        est[reader] = dict(A2=summaries[0], AS=summaries[1], qwk_difference=float(summaries[1]['qwk'] - summaries[0]['qwk']))
        boot[reader] = bootstrap[1]['qwk'] - bootstrap[0]['qwk']
        missing = (w @ np.eye(k)[y] == 0).any(1)
        boot[reader][missing] = np.nan
        est[reader]['qwk_difference_ci'] = np.nanquantile(boot[reader], [0.025, 0.975])
        est[reader]['bootstrap_missing_class'] = int(missing.sum())
        est[reader]['bootstrap_estimable'] = int(np.isfinite(boot[reader]).sum())
    b = (boot['pathologist1'] + boot['pathologist2']) / 2
    est['mean_readers'] = dict(qwk_difference=(est['pathologist1']['qwk_difference'] + est['pathologist2']['qwk_difference']) / 2, qwk_difference_ci=np.nanquantile(b, [0.025, 0.975]))
    write(folder / 'paired_core_metrics.json', dict(cores=n, readers=est, bootstrap=10000, seed=42, reader_labels_resampled_together=True))

def main():
    from celllift.core_and_nucleus_prediction.evaluate import _window_meta_map, _alphas_for_core, _core_gt_map
    from celllift.model_inputs.arvaniti_lizard.official import pool_official_from_alphas
    windows = _window_meta_map()
    gt = _core_gt_map()
    folder = OUT / 'arvaniti/local/model'
    dest = folder / 'core_summary'
    dest.mkdir(parents=True, exist_ok=True)
    root = RESULT / 'arvaniti_lizard_full_v16c_seed42_fix1/arvaniti/interact'
    saved = {}
    expected = {}
    for arm in ('A2', 'AS'):
        with np.load(root / arm / 'seed42/test_infer_logits.npz', allow_pickle=True) as z:
            saved[arm] = {str(g): p for g, p in zip(z['graph_id'], softmax(z['logits'].astype(float), axis=-1))}
            for g, c in zip(z['graph_id'], z['core_id']):
                expected.setdefault(str(c), set()).add(str(g))
    bycore = {}
    paths = {}
    for path in (folder / 'predict/test').glob('*.json'):
        r = read(path)
        if r['status'] != 'complete':
            continue
        with np.load(path.with_suffix('.npz')) as z:
            bycore.setdefault(r['cluster_id'], {})[r['sample_id']] = {'A2': z['probability2'], 'AS': z['probability3']}
        paths[r['sample_id']] = path
    records = []
    issues = []
    context = {}
    for core, want in expected.items():
        got = bycore.get(core, {})
        if set(got) != want:
            issues.append(dict(core_id=core, status='incomplete_windows', expected=len(want), available=len(got)))
            continue
        ids = sorted(want)
        alpha = _alphas_for_core(core, [windows[g] for g in ids])
        r = dict(core_id=core, gt=gt[core], windows=len(ids))
        context[core] = (ids, alpha, got)
        for arm in ('A2', 'AS'):
            r[arm] = pool_official_from_alphas(np.stack([got[g][arm] for g in ids]), alpha)
            ref = pool_official_from_alphas(np.stack([saved[arm][g] for g in ids]), alpha)
            r[arm]['saved_core_class_equal'] = r[arm]['qwk_label'] == ref['qwk_label']
        if not all((r[arm]['saved_core_class_equal'] for arm in ('A2', 'AS'))):
            issues.append(dict(core_id=core, status='numerical_failure', reason='discrete official core grade mismatch'))
            continue
        records.append(r)
    write(dest / 'original_core_outputs.json', records)
    write(dest / 'coverage.json', dict(expected_cores=len(expected), complete_cores=len(records), issues=issues))
    if records:
        paired_metrics(records, dest)
    perturb = []
    for core, (ids, alpha, original) in context.items():
        pp = [folder / 'perturb/test' / paths[g].name for g in ids]
        if not all((p.exists() and read(p).get('status') == 'complete' for p in pp)):
            continue
        values = {}
        for p in pp:
            with np.load(p.with_suffix('.npz')) as z:
                for name in ('training_matched', 'within_graph', 'image_matched'):
                    values.setdefault(name, []).append(z[name])
        for name, parts in values.items():
            arr = np.stack(parts, 1)
            for j in range(20):
                perturb.append(dict(core_id=core, method=name, repeat=j, **pool_official_from_alphas(arr[j], alpha)))
    pd.DataFrame(perturb).to_csv(dest / 'perturbed_core_outputs.csv', index=False)
    perturbation_metrics(records, perturb, dest)
    restored = []
    for p in (folder / 'mask/test').glob('*.json'):
        r = read(p)
        core = r['cluster_id']
        if core not in context:
            continue
        ids, alpha, original = context[core]
        base = np.stack([original[g]['AS'] for g in ids])
        gid = r['graph_id']
        i = ids.index(gid)
        for curve in r['curves']:
            x = base.copy()
            x[i] = curve['probability']
            restored.append(dict(core_id=core, graph_id=gid, method=curve['method'], seed=curve['seed'], fraction=curve['fraction'], **pool_official_from_alphas(x, alpha)))
    pd.DataFrame(restored).to_csv(dest / 'single_window_restoration_core_outputs.csv', index=False)
if __name__ == '__main__':
    main()
