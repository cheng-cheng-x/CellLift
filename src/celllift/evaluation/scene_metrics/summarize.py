from celllift.runtime import resource_path as _public_resource
import json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import RESULT, SCALES, CONFIG, write, record

def main():
    reports = {}
    paired = {}
    rng = np.random.default_rng(42)
    for split, expected in CONFIG['splits'].items():
        folders = sorted((RESULT / '05_evaluation' / split).glob('*/complete.json'))
        assert len(folders) == expected
        observations = pd.concat([pd.read_csv(f.parent / 'observations.csv.gz') for f in folders], ignore_index=True)
        track = observations.drop(columns='graph_id').groupby(['method', 'mode', 'view', 'track_id']).sum(numeric_only=True).reset_index()
        track['dice'] = track.dice_sum / track.n
        track['coverage'] = track.nonempty / track.n
        track['conditional_dice'] = track.dice_sum / track.nonempty.replace(0, np.nan)
        for scale in SCALES:
            key = str(scale)
            track['J_' + key] = track['j_' + key] / track.n
            track['eligible_fraction_' + key] = track['eligible_' + key] / track.n
        scene = []
        dual = []
        original_records = []
        for f in folders:
            for s in json.loads((f.parent / 'scene.json').read_text()):
                original_records.append(s)
                row = dict(method=s['method'], mode=s['mode'], track_id=s['track_id'], graph_id=s['graph_id'], nodes=s['nodes'], valid=s['valid'], z_std=s['nucleus_z_std'])
                row.update({'collision_' + k: v for k, v in s['collision'].items()})
                if 'original' in s:
                    for kind in ['original', 'flattened']:
                        row[kind + '_volume_overlap'] = s[kind]['volume_overlap']
                        for rep in range(4):
                            row[kind + '_r' + str(rep)] = s[kind]['volume_overlap_replicates'][rep]
                            row[kind + '_half_r' + str(rep)] = s[kind]['volume_overlap_half_replicates'][rep]
                    row['volume_change'] = row['flattened_volume_overlap'] - row['original_volume_overlap']
                    row.update({'flattened_collision_' + k: v for k, v in s['flattened_collision'].items()})
                scene.append(row)
                for d in s['dual_observations']:
                    dual.append(dict(method=s['method'], mode=s['mode'], track_id=s['track_id'], **d))
        scene = pd.DataFrame(scene)
        st = scene.drop(columns='graph_id').groupby(['method', 'mode', 'track_id']).mean(numeric_only=True).reset_index()
        dt = pd.DataFrame(dual).groupby(['method', 'mode', 'kind', 'track_id']).sum(numeric_only=True).reset_index()
        report = {}
        for (method, mode), t in track.groupby(['method', 'mode']):
            ss = st[(st.method == method) & (st['mode'] == mode)]
            orig = scene[(scene.method == method) & (scene['mode'] == mode)]
            out = dict(graphs=len(orig), tracks=len(ss), legal_coverage=float(orig.valid.sum() / orig.nodes.sum()), observations={})
            for view, v in t.groupby('view'):
                out['observations'][view] = dict(dice=float(v.dice.mean()), coverage=float(v.coverage.mean()), conditional_dice=float(v.conditional_dice.fillna(0).mean()), J={str(scale): float(v['J_' + str(scale)].mean()) for scale in SCALES}, eligible={str(scale): float(v['eligible_fraction_' + str(scale)].mean()) for scale in SCALES})
            out['collision'] = {str(scale): float(ss['collision_' + str(scale)].mean()) for scale in SCALES}
            out['nucleus_z_std'] = float(ss.z_std.mean())
            if orig.original_volume_overlap.notna().any():
                out['volume'] = {}
                for kind in ['original', 'flattened']:
                    rep = ss[[kind + '_r' + str(i) for i in range(4)]].mean().to_numpy()
                    half = ss[[kind + '_half_r' + str(i) for i in range(4)]].mean().to_numpy()
                    roi_gap = orig[kind + '_volume_overlap'] - orig[[kind + '_half_r' + str(i) for i in range(4)]].mean(1)
                    out['volume'][kind] = dict(mean=float(rep.mean()), replicates=rep.tolist(), numerical_standard_error=float(rep.std(ddof=1) / 2), half_sample_mean=float(half.mean()), half_to_full_difference=float(rep.mean() - half.mean()), roi_abs_half_to_full_q95=float(roi_gap.abs().quantile(0.95)))
                out['volume']['intervention_delta'] = float(ss.volume_change.mean())
                out['flattened_collision'] = {str(scale): float(ss['flattened_collision_' + str(scale)].mean()) for scale in SCALES}
            out['dual_positive'] = {}
            for kind in ['nucleus', 'cell']:
                d = dt[(dt.method == method) & (dt['mode'] == mode) & (dt.kind == kind)]
                keep = d.n > 0
                d = d[keep]
                out['dual_positive'][kind] = dict(objects=int(d.n.sum()), both_nonempty=float((d.both_nonempty / d.n).mean()), worst_side_dice=float((d.minimum_dice_sum / d.n).mean()))
            report.setdefault(method, {})[mode] = out
        reports[split] = report
        paired[split] = {}
        for method in ['cylinder', 'ellipsoid', 'p4', 'v16c_soft']:
            bm = 'v16c' if method.startswith('v16c_') else method
            mode = method.split('_')[1] if bm == 'v16c' else 'main'
            entry = {}
            for view in ['nucleus_adjacent', 'cell_adjacent', 'cell_middle']:
                a = track[(track.method == 'v16c') & (track['mode'] == 'main') & (track['view'] == view)].set_index('track_id')
                b = track[(track.method == bm) & (track['mode'] == mode) & (track['view'] == view)].set_index('track_id')
                for metric in ['dice', 'J_0.98']:
                    assert set(a.index) == set(b.index)
                    d = (a[metric] - b[metric]).to_numpy()
                    boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
                    entry[view + '_' + metric] = dict(delta=float(d.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist())
            a = st[(st.method == 'v16c') & (st['mode'] == 'main')].set_index('track_id')
            b = st[(st.method == bm) & (st['mode'] == mode)].set_index('track_id')
            for metric in ['collision_0.98', 'original_volume_overlap']:
                d = (a[metric] - b[metric]).to_numpy()
                boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
                entry[metric] = dict(delta=float(d.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist())
            paired[split][method] = entry
        track.to_csv(RESULT / '05_evaluation' / (split + '_track_metrics.csv'), index=False)
        st.to_csv(RESULT / '05_evaluation' / (split + '_track_scenes.csv'), index=False)
        dt.to_csv(RESULT / '05_evaluation' / (split + '_dual_tracks.csv'), index=False)
    write(RESULT / '05_evaluation/report.json', reports)
    write(RESULT / '05_evaluation/paired.json', paired)
    figure = RESULT / '06_figures'
    figure.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    eps = [100 * (1 - s) for s in SCALES]
    for method in ['cylinder', 'ellipsoid', 'p4', 'v16c']:
        value = reports['test'][method]['main']
        for ax, view in zip(axes[:2], ['nucleus_adjacent', 'cell_adjacent']):
            ax.plot(eps, [value['observations'][view]['J'][str(s)] for s in SCALES], '-o', label=method)
            ax.set_xlabel('Permitted shrink (%)')
            ax.set_ylabel('Separation-qualified Dice: ' + view)
        axes[2].scatter(value['collision']['0.98'] * 100, value['observations']['nucleus_adjacent']['dice'], label=method)
        axes[2].annotate(method, (value['collision']['0.98'] * 100, value['observations']['nucleus_adjacent']['dice']))
    axes[2].set_xlabel('Collision after 2% shrink (%)')
    axes[2].set_ylabel('Original adjacent nucleus Dice')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure / 'joint_quality.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = ['cylinder', 'ellipsoid', 'p4', 'v16c']
    x = np.arange(4)
    ax.bar(x - 0.17, [100 * reports['test'][m]['main']['volume']['original']['mean'] for m in names], 0.34, label='Original')
    ax.bar(x + 0.17, [100 * reports['test'][m]['main']['volume']['flattened']['mean'] for m in names], 0.34, label='Nucleus centers at z=2.5')
    ax.set_xticks(x, names)
    ax.set_ylabel('Repeated cell occupancy volume (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure / 'volume_intervention.png', dpi=180)
    plt.close(fig)
    write(RESULT / 'runtime/complete.json', record(status='COMPLETE'))
    print(json.dumps(reports['test']), flush=True)
if __name__ == '__main__':
    main()
