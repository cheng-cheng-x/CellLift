from celllift.runtime import resource_path as _public_resource
from .common import *
from .phenotype import fit
import pandas as pd, itertools

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(TASKS))
    args = parser.parse_args()
    pairs = [('nucleus_direction_spacing', 'paired_2d_xy_direction_spacing'), ('nucleus_distance_3d_um', 'nucleus_distance_xy_um'), ('cell_distance_3d_um', 'cell_distance_xy_um'), ('nucleus_distance_3d_um', 'paired_2d_xy_distance')]
    for ds, tasks in TASKS.items():
        if args.dataset and ds != args.dataset:
            continue
        path = OUT / ds / 'shared/units.parquet'
        if not path.exists():
            continue
        allunits = pd.read_parquet(path)
        for task in tasks:
            data = allunits[allunits.task == task]
            ref = data[data.split.isin(['train', 'fit'])]
            test = data[data.split == 'test']
            out = []
            status = []
            k = len(LABELS[task])
            for a, b in pairs:
                for stat in ('median', 'iqr'):
                    cols = [a + '__' + stat, b + '__' + stat]
                    if not all((c in data for c in cols)):
                        status.append(dict(features=cols, status='missing_paired_descriptor'))
                        continue
                    z = test[test[cols].notna().all(1)]
                    labels = z.label_id.to_numpy(int)
                    x = np.column_stack([np.ones(len(z))] + [(labels == j).astype(float) for j in range(1, k)])
                    sd = ref[cols].std(ddof=0).replace(0, 1)
                    y = ((z[cols] - ref[cols].mean()) / sd).to_numpy()
                    try:
                        beta, boot, _, _, _, meta = fit(x, y, z.cluster_id.to_numpy(str), k)
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        status.append(dict(features=cols, status='not_estimable', reason=str(exc)))
                        continue
                    for lo, hi in itertools.combinations(range(k), 2):
                        v = np.zeros(k)
                        v[hi] = 1
                        if lo:
                            v[lo] -= 1
                        e = v @ beta
                        samples = np.einsum('k,bkd->bd', v, boot)
                        difference = e[0] - e[1]
                        dsamples = samples[:, 0] - samples[:, 1]
                        absdiff = abs(e[0]) - abs(e[1])
                        absboot = abs(samples[:, 0]) - abs(samples[:, 1])
                        ci = np.quantile(dsamples, [0.025, 0.975])
                        aci = np.quantile(absboot, [0.025, 0.975])
                        p = (1 + (abs(dsamples - difference) >= abs(difference)).sum()) / (len(dsamples) + 1)
                        out.append(dict(feature3d=cols[0], reference=cols[1], low=LABELS[task][lo], high=LABELS[task][hi], effect3d=e[0], effect_reference=e[1], signed_effect_difference=difference, ci_low=ci[0], ci_high=ci[1], absolute_effect_difference=absdiff, absolute_ci_low=aci[0], absolute_ci_high=aci[1], p=p, **meta))
            folder = OUT / ds / task / 'phenotype/paired_relations'
            folder.mkdir(parents=True, exist_ok=True)
            t = pd.DataFrame(out)
            if len(t):
                t['q_exploratory_family'] = bh(t.p)
            t.to_csv(folder / 'effects.csv', index=False)
            write(folder / 'status.json', dict(issues=status, interpretation='jointly standardized effects, same valid edge pairs; larger effects do not prove more accurate reconstruction'))
if __name__ == '__main__':
    main()
