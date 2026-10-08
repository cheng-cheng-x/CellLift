from celllift.runtime import resource_path as _public_resource
from .common import *
from .phenotype import fit
import pandas as pd, itertools

def main():
    f = pd.read_parquet(OUT / 'lizard/shared/type_pairs.parquet')
    pairs = list(itertools.combinations_with_replacement(range(6), 2))
    label = {v: i for i, v in enumerate(pairs)}
    f['label_id'] = [label[a, b] for a, b in zip(f.type_a, f.type_b)]
    cols = columns(EDGE + DIST)
    ref = f[f.split.isin(['train', 'fit'])]
    test = f[f.split == 'test']
    folder = OUT / 'lizard/nucleus6/phenotype/type_pairs'
    folder.mkdir(parents=True, exist_ok=True)
    distribution = []
    for (a, b), g in test.groupby(['type_a', 'type_b']):
        for c in cols:
            distribution.append(dict(type_a=LABELS['nucleus6'][a], type_b=LABELS['nucleus6'][b], feature=c, median=g[c].median(), q25=g[c].quantile(0.25), q75=g[c].quantile(0.75), source_groups=g.cluster_id.nunique(), edges=int(g.edges.sum())))
    pd.DataFrame(distribution).to_csv(folder / 'distributions.csv', index=False)
    effects = []
    omnibus = []
    status = []
    for c in cols:
        z = test[test[c].notna()]
        labels = z.label_id.to_numpy()
        x = np.column_stack([np.ones(len(z))] + [(labels == j).astype(float) for j in range(1, 21)])
        sd = ref[c].std(ddof=0)
        sd = sd if sd > 1e-10 else 1
        y = ((z[c] - ref[c].mean()) / sd).to_numpy()[:, None]
        try:
            b, bs, lp, om, eq, meta = fit(x, y, z.cluster_id.to_numpy(str), 21)
        except (ValueError, np.linalg.LinAlgError) as exc:
            status.append(dict(feature=c, status='not_estimable', reason=str(exc)))
            continue
        cov = np.cov(bs[:, 1:, 0].T)
        inv = np.linalg.pinv(cov)
        stat = float(b[1:, 0] @ inv @ b[1:, 0])
        null = bs[:, 1:, 0] - b[None, 1:, 0]
        p = (1 + (np.einsum('bi,ij,bj->b', null, inv, null) >= stat).sum()) / (len(bs) + 1)
        omnibus.append(dict(feature=c, p=p, statistic=stat, **meta))
        status.append(dict(feature=c, status='complete', **meta))
        for a, z in itertools.combinations(range(21), 2):
            v = np.zeros(21)
            v[z] = 1
            if a:
                v[a] -= 1
            est = float(v @ b[:, 0])
            samples = bs[:, :, 0] @ v
            ci = np.quantile(samples, [0.025, 0.975])
            p = (1 + (abs(samples - est) >= abs(est)).sum()) / (1 + len(samples))
            effects.append(dict(feature=c, low=str(pairs[a]), high=str(pairs[z]), effect=est, ci_low=ci[0], ci_high=ci[1], p=p))
    for name, rows in [('omnibus', omnibus), ('effects', effects)]:
        t = pd.DataFrame(rows)
        if len(t):
            t['q'] = bh(t.p)
        t.to_csv(folder / (name + '.csv'), index=False)
    write(folder / 'status.json', status)
if __name__ == '__main__':
    main()
