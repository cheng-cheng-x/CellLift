from celllift.runtime import resource_path as _public_resource
from .common import *
import pandas as pd
import argparse, itertools
from scipy.interpolate import CubicSpline

def design(frame, ref, k, score_columns=None):
    labels = frame.label_id.to_numpy(int)
    x = [np.ones(len(frame))] + [(labels == j).astype(float) for j in range(1, k)]
    spec = {}
    for col in COVARIATES:
        v = ref[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            spec[col] = {'status': 'missing_development'}
            x.append(np.full(len(frame), np.nan))
            continue
        center = v.mean()
        scale = v.std()
        if scale < 1e-10:
            spec[col] = {'constant': True}
            continue
        knots = np.unique(np.quantile((v - center) / scale, [0, 1 / 3, 2 / 3, 1]))
        spec[col] = dict(center=center, scale=scale, knots=knots)
        if len(knots) < 2:
            continue
        spline = CubicSpline(knots, np.eye(len(knots)), bc_type='natural', axis=0)
        z = (frame[col].to_numpy(float) - center) / scale
        clip = np.clip(z, knots[0], knots[-1])
        basis = spline(clip) + (z - clip)[:, None] * spline(clip, 1)
        x.extend((basis[:, j] for j in range(1, len(knots))))
    for col in score_columns or []:
        x.append(frame[col].to_numpy(float))
    return (np.column_stack(x), spec)

def fit(x, y, groups, k, reps=10000):
    unique, gi = np.unique(groups, return_inverse=True)
    ng = len(unique)
    p = x.shape[1]
    d = y.shape[1]
    if ng < 2 or np.linalg.matrix_rank(x) < p:
        raise ValueError('insufficient clusters or rank-deficient design')
    xx = np.zeros((ng, p, p))
    xy = np.zeros((ng, p, d))
    counts = np.zeros((ng, k))
    for j in range(ng):
        a = x[gi == j]
        b = y[gi == j]
        xx[j] = a.T @ a
        xy[j] = a.T @ b
        lab = np.argmax(np.column_stack((1 - a[:, 1:k].sum(1), a[:, 1:k])), axis=1)
        counts[j] = np.bincount(lab, minlength=k)
    total = xx.sum(0)
    rhs = xy.sum(0)
    beta = np.linalg.solve(total, rhs)[:k]
    rng = np.random.default_rng(42)
    boot = []
    invalid = 0
    for start in range(0, reps, 128):
        n = min(128, reps - start)
        w = rng.multinomial(ng, np.ones(ng) / ng, size=n)
        gram = np.einsum('bg,gij->bij', w, xx)
        rhsb = np.einsum('bg,gjk->bjk', w, xy)
        sv = np.linalg.eigvalsh(gram)
        ok = (sv[:, 0] > sv[:, -1] * 1e-12) & (w @ counts > 0).all(1)
        invalid += int((~ok).sum())
        if ok.any():
            boot.append(np.linalg.solve(gram[ok], rhsb[ok])[:, :k])
    if not boot or sum((len(a) for a in boot)) < 100:
        raise ValueError('fewer than100 estimable bootstrap replicates')
    lopo = []
    omitted = []
    for j in range(ng):
        a = total - xx[j]
        if np.linalg.matrix_rank(a) < p or (counts.sum(0) - counts[j] == 0).any():
            continue
        lopo.append(np.linalg.solve(a, rhs - xy[j])[:k])
        omitted.append(str(unique[j]))
    w = 1 / np.bincount(gi)[gi]
    equal = np.linalg.lstsq(x * np.sqrt(w[:, None]), y * np.sqrt(w[:, None]), rcond=None)[0][:k]
    return (beta, np.concatenate(boot), np.array(lopo), omitted, equal, dict(clusters=ng, units=len(x), design_rank=p, rejected_replicates=invalid, requested_replicates=reps))

def family(c):
    if c in PRIMARY:
        return 'primary40'
    if c in DISTANCE:
        return 'distance12'
    if c.startswith('between_graph_'):
        return 'between_graph'
    return '2d_reference'

def analyze(frame, ref, folder, labels, score_columns=None):
    folder.mkdir(parents=True, exist_ok=True)
    k = len(labels)
    feats = [c for c in PRIMARY + DISTANCE + columns(TWO) + ['node_count'] if c in frame]
    feats += [c for c in frame if c.startswith('between_graph_')]
    x, spec = design(frame, ref, k, score_columns)
    write(folder / 'design.json', dict(labels=labels, splines=spec, reference='development only', additional_probability_columns=score_columns or []))
    effects = []
    omnibus = []
    status = []
    deleted = []
    within = []
    for mode, mat in [('raw', x[:, :k]), ('adjusted', x)]:
        wanted = [c for c in feats if mode == 'raw' or family(c) != '2d_reference']
        grouped = {}
        for c in wanted:
            mask = np.isfinite(frame[c].to_numpy(float)) & np.isfinite(mat).all(1)
            grouped.setdefault(mask.tobytes(), (mask, []))[1].append(c)
        for mask, cols in grouped.values():
            means = ref[cols].mean().to_numpy()
            sd = ref[cols].std(ddof=0).to_numpy()
            sd = np.where(sd > 1e-10, sd, 1)
            y = (frame.loc[mask, cols].to_numpy(float) - means) / sd
            try:
                b, bs, lp, om, eq, meta = fit(mat[mask], y, frame.loc[mask, 'cluster_id'].astype(str).to_numpy(), k)
            except (ValueError, np.linalg.LinAlgError) as e:
                status.append(dict(mode=mode, features=cols, status='not_estimable', reason=str(e)))
                continue
            status.append(dict(mode=mode, features=cols, status='complete', **meta))
            for j, c in enumerate(cols):
                cov = np.atleast_2d(np.cov(bs[:, 1:, j].T))
                inv = np.linalg.pinv(cov)
                stat = float(b[1:, j] @ inv @ b[1:, j])
                centered = bs[:, 1:, j] - b[None, 1:, j]
                null = np.einsum('bi,ij,bj->b', centered, inv, centered)
                p = (1 + (null >= stat).sum()) / (1 + len(bs))
                omnibus.append(dict(feature=c, family=family(c), mode=mode, p=p, statistic=stat, **meta))
                for lo, hi in itertools.combinations(range(k), 2):
                    contrast = np.zeros(k)
                    contrast[hi] = 1
                    if lo:
                        contrast[lo] -= 1
                    est = float(contrast @ b[:, j])
                    samples = bs[:, :, j] @ contrast
                    ci = np.quantile(samples, [0.025, 0.975])
                    ps = (1 + (abs(samples - est) >= abs(est)).sum()) / (1 + len(samples))
                    dels = lp[:, :, j] @ contrast if len(lp) else np.array([])
                    effects.append(dict(feature=c, family=family(c), mode=mode, low=labels[lo], high=labels[hi], effect=est, ci_low=ci[0], ci_high=ci[1], p=ps, patient_equal_effect=float(contrast @ eq[:, j]), lopo_sign_agreement=float(np.mean(np.sign(dels) == np.sign(est))) if len(dels) else np.nan, **meta))
                    deleted.extend((dict(feature=c, mode=mode, low=labels[lo], high=labels[hi], omitted_cluster=g, effect=v) for g, v in zip(om, dels)))
            print(folder, mode, len(cols), meta, flush=True)
    for name, records in [('effects', effects), ('omnibus', omnibus)]:
        t = pd.DataFrame(records)
        if not t.empty:
            t['q'] = np.nan
            for _, ix in t.groupby(['family', 'mode']).groups.items():
                t.loc[ix, 'q'] = bh(t.loc[ix, 'p'])
        t.to_csv(folder / (name + '.csv'), index=False)
    pd.DataFrame(deleted).to_csv(folder / 'leave_one_cluster_out.csv', index=False)
    write(folder / 'fit_status.json', status)
    dist = []
    for lab, g in frame.groupby('label_id'):
        for c in feats:
            v = g[c].dropna()
            dist.append(dict(label=labels[int(lab)], feature=c, n=len(v), clusters=g.loc[g[c].notna(), 'cluster_id'].nunique(), median=v.median(), q25=v.quantile(0.25), q75=v.quantile(0.75)))
    pd.DataFrame(dist).to_csv(folder / 'distributions.csv', index=False)
    for cid, g in frame.groupby('cluster_id'):
        for a, z in itertools.combinations(sorted(g.label_id.unique()), 2):
            delta = (g[g.label_id == z][PRIMARY].mean() - g[g.label_id == a][PRIMARY].mean()) / ref[PRIMARY].std(ddof=0)
            within.extend((dict(cluster_id=cid, low=labels[int(a)], high=labels[int(z)], feature=c, effect=v) for c, v in delta.items()))
    pd.DataFrame(within).to_csv(folder / 'within_cluster_contrasts.csv', index=False)

def themes(ref, folder):
    if not (folder / 'train/effects.csv').exists() or (folder / 'train/effects.csv').stat().st_size < 3:
        write(folder / 'themes.json', dict(themes=[], source='development only', status='no estimable development effects'))
        return
    e = pd.read_csv(folder / 'train/effects.csv')
    e = e[(e.family == 'primary40') & (e['mode'] == 'adjusted')].copy()
    e['priority'] = abs(e.effect) * e.lopo_sign_agreement
    e = e.sort_values(['priority', 'feature', 'low', 'high'], ascending=[False, True, True, True])
    chosen = []
    for _, r in e.iterrows():
        if not np.isfinite(r.priority):
            continue
        if any((r.feature.split('__')[0] == q['feature'].split('__')[0] or abs(ref[r.feature].corr(ref[q['feature']], method='spearman')) > 0.85 for q in chosen)):
            continue
        chosen.append(r.to_dict())
        if len(chosen) == 3:
            break
    write(folder / 'themes.json', dict(themes=chosen, source='development only', status='complete' if chosen else 'no estimable adjusted themes'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=TASKS)
    p.add_argument('--task')
    a = p.parse_args()
    f = pd.read_parquet(OUT / a.dataset / 'shared/units.parquet')
    for task in [a.task] if a.task else TASKS[a.dataset]:
        t = f[f.task == task].copy()
        ref = t[t.split.isin(['train', 'fit'])]
        dest = OUT / a.dataset / task / 'phenotype'
        if ref.empty:
            write(dest / 'status.json', dict(status='not_estimable', reason='no development units'))
            continue
        for split in ('train', 'val', 'test'):
            z = t[t.split.isin(['train', 'fit']) if split == 'train' else t.split == split].reset_index(drop=True)
            if len(z):
                analyze(z, ref, dest / split, LABELS[task])
        themes(ref, dest)
        write(dest / 'status.json', dict(status='complete', units=len(t), clusters=t.cluster_id.nunique()))
if __name__ == '__main__':
    main()
