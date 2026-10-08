from celllift.runtime import resource_path as _public_resource
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from celllift.evaluation.geometric_baselines.src.common import RESULT, CONFIG, write, record

def main():
    summary = {}
    paired = {}
    rng = np.random.default_rng(42)
    modelroot = RESULT.with_name('v16c_staged_shape_scene_reconstruction') / '05_evaluation'
    for split in ['train', 'val']:
        modelt = pd.read_csv(modelroot / (split + '_track_metrics.csv'))
        models = pd.read_csv(modelroot / (split + '_track_scenes.csv'))
        modelt = modelt[modelt['mode'] == 'main']
        models = models[models['mode'] == 'main']
        summary[split] = {}
        paired[split] = {}
        for method in CONFIG['methods']:
            folders = sorted((RESULT / '05_evaluation' / method / split).glob('*/complete.json'))
            assert len(folders) == (1024 if split == 'train' else 256)
            parts = []
            scenes = []
            for f in folders:
                raw = pd.read_csv(f.parent / 'raw.csv.gz', usecols=['mode', 'track_id', 'kind', 'plane', 'dice', 'empty_prediction', 'relative_area_error', 'centroid_error_um'])
                raw['view'] = raw.kind + raw.plane.map(lambda x: '_middle' if x == 1 else '_adjacent')
                parts.append(raw.groupby(['mode', 'track_id', 'view']).agg(dice_sum=('dice', 'sum'), n=('dice', 'size'), empty_sum=('empty_prediction', 'sum'), area_sum=('relative_area_error', 'sum'), centroid_sum=('centroid_error_um', 'sum'), centroid_n=('centroid_error_um', 'count')).reset_index())
                for s in json.loads((f.parent / 'scene.json').read_text()):
                    scenes.append(dict(mode=s['mode'], track_id=s['track_id'], nodes=s['nodes'], valid=s['valid'], collision=s['collision']['1.0'], c98=s['collision']['0.98'], depth=s['penetration_per_valid'], nucleus_volume=s['nucleus_stats']['volume_mean'], cell_volume=s['cell_stats']['volume_mean'], violations=s['nucleus_stats']['violations'] + s['cell_stats']['violations'], containment=s['containment_violations']))
            track = pd.concat(parts).groupby(['mode', 'track_id', 'view']).sum(numeric_only=True).reset_index()
            track['dice'] = track.dice_sum / track.n
            track['empty'] = track.empty_sum / track.n
            track['area_error'] = track.area_sum / track.n
            scene = pd.DataFrame(scenes)
            st = scene.groupby(['mode', 'track_id']).mean(numeric_only=True).reset_index()
            track.to_csv(RESULT / '05_evaluation' / (method + '_' + split + '_tracks.csv'), index=False)
            st.to_csv(RESULT / '05_evaluation' / (method + '_' + split + '_scenes.csv'), index=False)
            summary[split][method] = {}
            for mode in ['short', 'main', 'long']:
                t = track[track['mode'] == mode]
                s = st[st['mode'] == mode]
                original = scene[scene['mode'] == mode]
                summary[split][method][mode] = dict(dice=t.groupby('view').dice.mean().to_dict(), empty=t.groupby('view')['empty'].mean().to_dict(), collision=float(s.collision.mean()), c98=float(s.c98.mean()), coverage=float(original.valid.sum() / original.nodes.sum()), violations=int(original.violations.sum()), containment=int(original.containment.sum()), nucleus_volume=float(s.nucleus_volume.mean()), cell_volume=float(s.cell_volume.mean()), graphs=len(original), tracks=len(s))
            out = {}
            for metric in ['nucleus_adjacent', 'cell_adjacent', 'cell_middle', 'c98']:
                if metric == 'c98':
                    a = models.set_index('track_id')['collision_0.98']
                    b = st[st['mode'] == 'main'].set_index('track_id').c98
                else:
                    a = modelt[modelt['view'] == metric].set_index('track_id').dice
                    b = track[(track['mode'] == 'main') & (track['view'] == metric)].set_index('track_id').dice
                assert set(a.index) == set(b.index)
                d = (a - b).to_numpy()
                boot = d[rng.integers(len(d), size=(2000, len(d)))].mean(1)
                out[metric] = dict(model_minus_baseline=float(d.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(), tracks=len(d))
            paired[split][method] = out
    write(RESULT / '05_evaluation/report.json', summary)
    write(RESULT / '05_evaluation/paired.json', paired)
    reference = json.loads((modelroot / 'report.json').read_text())['reports']['val']['main']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in CONFIG['methods']:
        x = summary['val'][method]
        for ax, view in zip(axes, ['nucleus_adjacent', 'cell_adjacent']):
            ax.plot([100 * x[m]['c98'] for m in ['short', 'main', 'long']], [x[m]['dice'][view] for m in ['short', 'main', 'long']], '-o', label=method)
            for m in ['short', 'main', 'long']:
                ax.annotate(m, (100 * x[m]['c98'], x[m]['dice'][view]), fontsize=7)
    for ax, view in zip(axes, ['nucleus_adjacent', 'cell_adjacent']):
        ax.scatter([100 * reference['collision']['0.98']], [reference['dice'][view]], marker='*', s=120, label='V16C input-only')
        ax.set_xlabel('Collision after 2% shrink (%)')
        ax.set_ylabel(view + ' Dice')
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT / '06_figures/height_tradeoff.png', dpi=180)
    plt.close(fig)
    write(RESULT / 'runtime/summary_done.json', record(status='COMPLETE'))
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
