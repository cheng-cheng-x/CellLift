from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from celllift.reconstruction.src.common import RESULT, atomic_json, record

def main():
    reports = {}
    paired = {}
    rng = np.random.default_rng(42)
    for split in ('val',):
        folders = sorted((RESULT / '05_evaluation' / split).glob('*/complete.json'))
        assert len(folders) == 1102
        raw = pd.concat([pd.read_csv(p.parent / 'raw.csv.gz') for p in folders], ignore_index=True)
        rows = sum([json.loads((p.parent / 'scene.json').read_text()) for p in folders], [])
        raw['view'] = raw.kind + raw.plane.map(lambda x: '_middle' if x == 1 else '_adjacent')
        tracks = raw.groupby(['mode', 'track_id', 'view']).agg(dice=('dice', 'mean'), empty=('empty_prediction', 'mean'), area_error=('relative_area_error', 'mean'), centroid_error=('centroid_error_um', 'mean')).reset_index()
        scenes = pd.DataFrame([{**r, **{'collision_' + s: v for s, v in r['collision'].items()}} for r in rows])
        st = scenes.groupby(['mode', 'track_id']).mean(numeric_only=True).reset_index()
        reports[split] = {}
        for mode in ('main', 'soft', 'reference'):
            x = tracks[tracks['mode'] == mode]
            s = st[st['mode'] == mode]
            rr = [r for r in rows if r['mode'] == mode]
            reports[split][mode] = dict(dice=x.groupby('view').dice.mean().to_dict(), collision={k: float(s['collision_' + k].mean()) for k in rr[0]['collision']}, nucleus_z_std=float(s.nucleus_z_std.mean()), coverage=sum((r['valid'] for r in rr)) / sum((r['nodes'] for r in rr)), physical_violations=sum((r['nucleus_stats']['violations'] + r['cell_stats']['violations'] for r in rr)), containment_violations=sum((r['containment_violations'] for r in rr)), graphs=len(rr), tracks=len(s))
        paired[split] = {}
        comparators = {mode: tracks[tracks['mode'] == mode] for mode in ('soft', 'reference')}
        scene_comparators = {mode: st[st['mode'] == mode] for mode in ('soft', 'reference')}
        for baseline, frame in comparators.items():
            result = {}
            for view in ('nucleus_adjacent', 'cell_adjacent', 'cell_middle'):
                a = tracks[(tracks['mode'] == 'main') & (tracks['view'] == view)].set_index('track_id').dice
                b = frame[frame['view'] == view].set_index('track_id').dice
                d = (a - b).dropna().to_numpy()
                boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
                result[view] = dict(delta=float(d.mean()), ci=np.quantile(boot, [0.025, 0.975]).tolist())
            a = st[st['mode'] == 'main'].set_index('track_id')['collision_0.98']
            b = scene_comparators[baseline].set_index('track_id')['collision_0.98']
            d = (a - b).dropna().to_numpy()
            boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
            result['collision_098'] = dict(delta=float(d.mean()), ci=np.quantile(boot, [0.025, 0.975]).tolist())
            paired[split][baseline] = result
        tracks.to_csv(RESULT / '05_evaluation' / (split + '_track_metrics.csv'), index=False)
        st.to_csv(RESULT / '05_evaluation' / (split + '_track_scenes.csv'), index=False)
        atomic_json(RESULT / '05_evaluation' / (split + '_scene_rows.json'), rows)
    atomic_json(RESULT / '05_evaluation/report.json', dict(reports=reports, protocol='Full dataset 8813 TRAIN; 1102 VAL for stopping and selection; reported VAL scores are development scores; reference diagnostic only; TEST unopened'))
    atomic_json(RESULT / '05_evaluation/paired.json', paired)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    gh = json.loads((RESULT / '03_training/geometry_validation/history.json').read_text())
    sh = json.loads((RESULT / '03_training/scorer_validation/history.json').read_text())
    for key in ('scene_free_energy', 'actual_energy'):
        axs[0].plot([x['epoch'] for x in gh], [x['monitor'][key] for x in gh], label=key)
    for key in ('collision', 'core_collision'):
        axs[1].plot([x['epoch'] for x in gh], [100 * x['monitor'][key] for x in gh], label=key)
    axs[2].plot([x['epoch'] for x in sh], [x['monitor']['kl'] for x in sh], label='scorer KL')
    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT / '06_figures/training.png', dpi=160)
    plt.close(fig)
    atomic_json(RESULT / 'runtime/summarize_done.json', record(status='COMPLETE'))
    print(json.dumps(reports), flush=True)
if __name__ == '__main__':
    main()
