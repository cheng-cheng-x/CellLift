from celllift.runtime import resource_path as _public_resource
import sys, json, time
import torch, pandas as pd
from celllift.evaluation.geometric_baselines.src.common import RESULT, PROJECT, PATHS, write, record
from celllift.evaluation.geometric_baselines.src.data import rows
from celllift.evaluation.geometric_baselines.src.targets import prepare_targets
from celllift.evaluation.geometric_baselines.src.geometry import coefficients, stats, contact_pairs, contact
from celllift.evaluation.geometric_baselines.src.raster import raw_rows

@torch.no_grad()
def main(method):
    torch.set_num_threads(4)
    start = time.time()
    assert (RESULT / 'runtime/predictions_complete.json').exists()
    prepared = PROJECT / _public_resource('artifact_0052')
    lookup = pd.read_parquet(prepared / 'layer_lookup.parquet')
    by_uid = lookup.set_index('roi_layer_id')
    index = pd.read_parquet(prepared / 'nucleus_training_index.parquet')
    done = 0
    for split in ['test']:
        for row in rows(split):
            uid = row['graph_id']
            dest = RESULT / '05_evaluation' / method / split / uid
            if (dest / 'complete.json').exists():
                done += 1
                continue
            value = torch.load(RESULT / '04_predictions' / method / split / (uid + '.pt'), map_location='cpu', weights_only=False)
            ids = value['ids']
            center = value['center'].cuda()
            root = value['root'].cuda()
            cf = coefficients(root, method)
            meta = value['metadata']
            obs = torch.load(PATHS['cache'] / row['supervision_file'], map_location='cpu', weights_only=False)
            layer = by_uid.loc[uid]
            track = lookup[lookup.track_id == meta['track_id']].set_index('section_id')
            idx = index[index.anchor_layer_idx == meta['layer_idx']].set_index('anchor_nucleus_id')
            targets = {k: prepare_targets(obs[k], ids.numpy(), idx, layer, track, k, 'cuda') for k in ['nucleus', 'cell']}
            raw = []
            scenes = []
            for mode, body in value['bodies'].items():
                geo = {k: v.cuda() for k, v in body.items()}
                valid = geo['valid']
                n = len(valid)
                Q = cf['Q'] * geo['lam'][:, None, None].square()
                i, j = contact_pairs(center, Q, valid)
                pen = []
                for st in range(0, len(i), 16384):
                    a = i[st:st + 16384]
                    b = j[st:st + 16384]
                    pen.append((1 - contact(Q[a], Q[b], center[b] - center[a])).clamp_min(0))
                pen = torch.cat(pen) if pen else center.new_empty(0)
                coll = {}
                for scale in [1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9]:
                    hit = torch.zeros_like(valid)
                    bad = pen > (1 - scale if scale < 1 else 1e-05)
                    hit[i[bad]] = True
                    hit[j[bad]] = True
                    coll[str(scale)] = float(hit.sum() / valid.sum().clamp_min(1))
                scene = dict(method=method, mode=mode, graph_id=uid, track_id=meta['track_id'], nodes=n, valid=int(valid.sum()), collision=coll, penetration_per_valid=float(pen.sum() / valid.sum().clamp_min(1)), nucleus_z_std=0.0, cell_z_std=0.0, containment_violations=int(((geo['lam'] < 1 - 1e-08) | (geo['hc'] < geo['h'] - 1e-08))[valid].sum()))
                for kind, cap, ratio in [('nucleus', 800, 3), ('cell', 4000, 6)]:
                    st = stats(cf, geo, method, kind)
                    scene[kind + '_stats'] = dict(volume_mean=float(st['volume'][valid].mean()), volume_max=float(st['volume'][valid].max()), rms_ratio_max=float(st['ratio'][valid].max()), height_mean=float((geo['h'] if kind == 'nucleus' else geo['hc'])[valid].mean()), violations=int((((st['volume'] > cap * (1 + 1e-08)) | (st['ratio'] > ratio * (1 + 1e-08))) & valid).sum()))
                    h = geo['h'] if kind == 'nucleus' else geo['hc']
                    lam = torch.ones_like(h) if kind == 'nucleus' else geo['lam']
                    for item in raw_rows(center, root, h, lam, method, targets[kind], valid):
                        item.update(method=method, mode=mode, graph_id=uid, track_id=meta['track_id'], kind=kind)
                        raw.append(item)
                scenes.append(scene)
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(raw).to_csv(dest / 'raw.csv.gz', index=False)
            write(dest / 'scene.json', scenes)
            write(dest / 'complete.json', record(uid=uid))
            done += 1
            if done % 32 == 0:
                write(RESULT / 'runtime' / ('evaluate_' + method + '.json'), record(done=done, total=1280, seconds=time.time() - start))
                print('EVAL', method, done, flush=True)
    write(RESULT / 'runtime' / ('evaluate_' + method + '_done.json'), record(done=done, seconds=time.time() - start))
if __name__ == '__main__':
    main(sys.argv[1])
