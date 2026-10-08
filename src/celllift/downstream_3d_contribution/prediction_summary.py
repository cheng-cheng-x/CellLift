from celllift.runtime import resource_path as _public_resource
from .common import *
import pandas as pd
from sklearn.metrics import roc_auc_score

def cm_metrics(cm):
    n = cm.sum(axis=(-2, -1))
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    true = cm.sum(-1)
    pred = cm.sum(-2)
    recall = np.divide(diag, true, out=np.zeros_like(diag, dtype=float), where=true > 0)
    f1 = np.divide(2 * diag, true + pred, out=np.zeros_like(diag, dtype=float), where=true + pred > 0)
    k = cm.shape[-1]
    w = (np.arange(k)[:, None] - np.arange(k)) ** 2
    observed = (cm * w).sum(axis=(-2, -1))
    expected = (true[..., :, None] * pred[..., None, :] * w).sum(axis=(-2, -1)) / np.maximum(n, 1)
    return dict(macro_f1=f1.mean(-1), recall=recall, qwk=1 - np.divide(observed, expected, out=np.full_like(observed, np.nan, dtype=float), where=expected > 0))

def analyze(f, folder, ordinal=False):
    folder.mkdir(parents=True, exist_ok=True)
    k = len(f.iloc[0].p3)
    groups, gi = np.unique(f.cluster_id.astype(str), return_inverse=True)
    ng = len(groups)
    y = f.label_id.to_numpy(int)
    p2 = np.stack(f.p2)
    p3 = np.stack(f.p3)
    c2 = p2.argmax(1)
    c3 = p3.argmax(1)
    if 'class2' in f:
        c2 = f.class2.to_numpy(int)
        c3 = f.class3.to_numpy(int)
    cm = []
    metrics = []
    for p, c in [(p2, c2), (p3, c3)]:
        mats = np.bincount(gi * k * k + y * k + c, minlength=ng * k * k).reshape(ng, k, k)
        cm.append(mats)
        met = cm_metrics(mats.sum(0))
        met['confusion'] = mats.sum(0)
        if not ordinal:
            met.pop('qwk')
        if k == 2:
            met.update(auroc=float(roc_auc_score(y, p[:, 1])) if len(np.unique(y)) == 2 else None, brier=float(np.mean((p[:, 1] - y) ** 2)))
        metrics.append(met)
    rng = np.random.default_rng(42)
    bs = {key: [] for key in ['macro_f1'] + (['qwk'] if ordinal else [])}
    invalid = 0
    for start in range(0, 10000, 256):
        w = rng.multinomial(ng, np.ones(ng) / ng, size=min(256, 10000 - start))
        counts = w @ np.stack([np.bincount(y[gi == j], minlength=k) for j in range(ng)])
        ok = (counts > 0).all(1)
        invalid += int((~ok).sum())
        left = cm_metrics(np.einsum('bg,gij->bij', w[ok], cm[0]))
        right = cm_metrics(np.einsum('bg,gij->bij', w[ok], cm[1]))
        for key in bs:
            bs[key].extend((right[key] - left[key]).tolist())
    intervals = {key: dict(difference=float(metrics[1][key] - metrics[0][key]), ci=np.nanquantile(values, [0.025, 0.975]), estimable=len(values)) for key, values in bs.items()}
    if k == 2:
        rng = np.random.default_rng(42)
        auc = []
        brier = []
        for j in range(10000):
            w = rng.multinomial(ng, np.ones(ng) / ng)[gi]
            if len(np.unique(y[w > 0])) < 2:
                continue
            auc.append(roc_auc_score(y, p3[:, 1], sample_weight=w) - roc_auc_score(y, p2[:, 1], sample_weight=w))
            brier.append(np.average((p3[:, 1] - y) ** 2 - (p2[:, 1] - y) ** 2, weights=w))
        for key, values in [('auroc', auc), ('brier', brier)]:
            intervals[key] = dict(difference=metrics[1][key] - metrics[0][key], ci=np.quantile(values, [0.025, 0.975]), estimable=len(values))
    correct2 = c2 == y
    correct3 = c3 == y
    f = f.copy()
    f['class2'] = c2
    f['class3'] = c3
    f['group'] = np.where(correct2 & correct3, 'both_correct', np.where(correct3, 'corrected', np.where(correct2, 'harmed', 'both_wrong')))
    f['true_probability_change'] = p3[np.arange(len(y)), y] - p2[np.arange(len(y)), y]
    if ordinal:
        f['squared_grade_error_improvement'] = (c2 - y) ** 2 - (c3 - y) ** 2
    f.drop(columns=['p2', 'p3']).to_csv(folder / 'paired_changes.csv', index=False)
    f.groupby(['cluster_id', 'group']).size().unstack(fill_value=0).to_csv(folder / 'cluster_changes.csv')
    write(folder / 'metrics.json', dict(version=2, units=len(f), clusters=ng, models=metrics, paired_difference=intervals, bootstrap=10000, seed=42, rejected_missing_class=invalid, ordinal=ordinal, groups=f.group.value_counts().to_dict(), uncertainty='conditional on existing frozen weights'))

def main():
    for ds, tasks in TASKS.items():
        if ds in ('arvaniti', 'sicapv2'):
            continue
        for task in tasks:
            folder = OUT / ds / task / 'model'
            records = []
            for path in (folder / 'predict/test').glob('*.json'):
                r = read(path)
                if r.get('status') != 'complete':
                    continue
                if ds == 'lizard':
                    with np.load(path.with_suffix('.npz')) as z:
                        for j in np.flatnonzero(z['owned']):
                            records.append(dict(sample_id=r['sample_id'] + ':' + str(z['ids'][j]), cluster_id=r['cluster_id'], label_id=int(z['labels'][j]), p2=z['probability2'][j], p3=z['probability3'][j]))
                else:
                    row = dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], label_id=r['label_id'], p2=r['probability2'], p3=r['probability3'])
                    if ds == 'tcga_brca' and len(row['p3']) == 2:
                        for arm, col, pcol in [('H2', 'class2', 'p2'), ('H3', 'class3', 'p3')]:
                            config = read(RESULT / f'tcga_brca_downstream_v1/brca_downstream_routes_v2_seed42/{task}/{arm}/seed42/metrics.json')
                            threshold = config.get('threshold')
                            threshold = 0.5 if threshold is None else threshold
                            row[col] = int(row[pcol][1] >= threshold)
                    records.append(row)
            previous = folder / 'summary/metrics.json'
            if records and (not (previous.exists() and read(previous).get('version') == 2 and (read(previous)['units'] == len(records)))):
                analyze(pd.DataFrame(records), folder / 'summary', ordinal=False)
            print('prediction summary', ds, task, len(records), flush=True)
if __name__ == '__main__':
    main()
