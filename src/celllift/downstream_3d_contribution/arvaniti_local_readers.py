from celllift.runtime import resource_path as _public_resource
from .common import *
from .prediction_summary import analyze
from .model_summary import intervals
import pandas as pd

def main():
    labels = pd.read_parquet(DATA / 'model_input_v1/arvaniti/04_labels_splits_v2/supervised_windows.parquet').set_index('graph_id')
    folder = OUT / 'arvaniti/local/model'
    tables = {1: [], 2: []}
    pert = {1: [], 2: []}
    for path in (folder / 'predict/test').glob('*.json'):
        r = read(path)
        gid = r['sample_id']
        if r.get('status') != 'complete' or gid not in labels.index:
            continue
        with np.load(path.with_suffix('.npz')) as z:
            p2 = z['probability2']
            p3 = z['probability3']
        for reader in (1, 2):
            label = int(labels.loc[gid, f'label_id_p{reader}'])
            if label >= 0:
                tables[reader].append(dict(sample_id=gid, cluster_id=r['cluster_id'], label_id=label, p2=p2, p3=p3))
        pp = folder / 'perturb/test' / path.name
        if not pp.exists() or read(pp).get('status') != 'complete':
            continue
        with np.load(pp.with_suffix('.npz')) as z:
            for method in ('training_matched', 'within_graph', 'image_matched'):
                prob = z[method]
                for reader in (1, 2):
                    label = int(labels.loc[gid, f'label_id_p{reader}'])
                    if label < 0:
                        continue
                    pert[reader].append(dict(sample_id=gid, cluster_id=r['cluster_id'], reader=reader, method=method, true_probability_change=float(np.mean(prob[:, label] - p3[label])), error_improvement=float(np.mean(float(p3.argmax() != label) - (prob.argmax(1) != label))), weight=1))
    for reader in (1, 2):
        dest = folder / f'local_reader{reader}'
        dest.mkdir(parents=True, exist_ok=True)
        if tables[reader]:
            analyze(pd.DataFrame(tables[reader]), dest, ordinal=True)
        if pert[reader]:
            frame = pd.DataFrame(pert[reader])
            frame.to_csv(dest / 'perturbation_units.csv', index=False)
            intervals(frame, ['method'], ['true_probability_change', 'error_improvement']).to_csv(dest / 'perturbation_intervals.csv', index=False)
        write(dest / 'scope.json', dict(status='available_for_current_records', reader=reader, local_units=len(tables[reader]), perturbed_local_units=len({r['sample_id'] for r in pert[reader]}), formal_endpoint=False, interpretation='Window-label summaries only. Formal task performance remains complete-core official pooling and both core readers. Reader labels are retained separately; the same sorted core IDs and seed42 retain the paired cluster resampling scheme.'))
if __name__ == '__main__':
    main()
