from celllift.runtime import resource_path as _public_resource
from .common import *
from .perturb_metrics import evaluate
from .model_summary import intervals
import pandas as pd

def main():
    for ds, task in [('lizard', 'nucleus6'), ('tcga_crc_msi', 'msi')]:
        folder = OUT / ds / task / 'model'
        ys = []
        base = []
        changed = []
        groups = []
        diagnostics = []
        status = {}
        for p in (folder / 'edge_diagnostic/test').glob('*.json'):
            r = read(p)
            status[r['status']] = status.get(r['status'], 0) + 1
            if r['status'] != 'complete':
                continue
            with np.load(p.with_suffix('.npz')) as z:
                own = z['owned'].astype(bool)
                before = z['original']
                after = z['perturbed']
                y = z['labels']
                if before.ndim == 1:
                    before = before[None]
                    after = after[:, None, :]
                before = before[own]
                after = after[:, own]
                y = y.reshape(-1)[own]
                n = len(y)
                ys.append(y)
                base.append(before)
                changed.append(after)
                groups.extend([r['cluster_id']] * n)
                kl = np.sum(before[None] * (np.log(np.clip(before[None], 1e-12, 1)) - np.log(np.clip(after, 1e-12, 1))), -1).mean()
                delta = (after[:, np.arange(n), y] - before[np.arange(n), y]).mean()
                diagnostics.append(dict(sample_id=r['sample_id'], cluster_id=r['cluster_id'], method='independent_edge_matched', coverage=r['coverage'], true_probability_change=delta, kl=kl, weight=n))
        dest = folder / 'summary'
        if ys:
            metrics = evaluate(np.concatenate(ys), np.concatenate(base), np.concatenate(changed, 1), np.asarray(groups))
            write(dest / 'independent_edge_task_metrics.json', metrics)
            frame = pd.DataFrame(diagnostics)
            frame.to_csv(dest / 'independent_edge_units.csv', index=False)
            intervals(frame, ['method'], ['coverage', 'true_probability_change', 'kl']).to_csv(dest / 'independent_edge_intervals.csv', index=False)
        write(dest / 'independent_edge_coverage.json', dict(records=status, interpretation='Directed edge-input rows matched on both endpoint 2D descriptors and 2D edge quantities; an input pathway diagnostic, not a coherent inferred scene'))
if __name__ == '__main__':
    main()
