from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import time
import sys
from celllift.runtime import ResourcePath as Path
from celllift.runtime import torch
import numpy as np
import pandas as pd
from celllift.reconstruction.src.common import RESULT, PROJECT, paths, atomic_json, atomic_save, record
from celllift.reconstruction.src.data import InputDataset, Supervision, EllipseObservationBatch, move
from celllift.reconstruction.src.geometry import Scene, physical_stats
from celllift.reconstruction.src.candidate_tables import observation_tables, select
from celllift.reconstruction.src.model import solve_scene
from celllift.reconstruction.src.fast_raster import raw_rows
from celllift.reconstruction.src.targets import prepare_targets

@torch.no_grad()
def main(shard, shards):
    torch.set_num_threads(4)
    assert (RESULT / 'runtime/predictions_complete.json').exists()
    info = json.loads((RESULT / '02_inputs/manifest.json').read_text())
    prepared = PROJECT / _public_resource('artifact_0061')
    from celllift.reconstruction.src.full_inputs import RawSource, confidence_index
    from celllift.reconstruction.src.common import atomic_save
    raw_source = RawSource()
    lookup = pd.read_parquet(prepared / 'layer_lookup.parquet')
    by_uid = lookup.set_index('roi_layer_id')
    index = pd.read_parquet(prepared / 'nucleus_training_index.parquet')
    done = 0
    started = time.time()
    for split in info['config']['evaluation_splits']:
        qtable, qgroups = confidence_index(split)
        data = InputDataset(paths(), split, False)
        for gi in range(shard, len(data), shards):
            uid = data.rows[gi]['graph_id']
            dest = RESULT / '05_evaluation' / split / uid
            if (dest / 'complete.json').exists():
                done += 1
                continue
            g, meta = data[gi]
            value = torch.load(RESULT / '04_predictions' / split / (uid + '.pt'), map_location='cuda', weights_only=False)
            assert torch.equal(value['ids'].cpu(), g.nucleus_id)
            targetpath = paths()['cache'] / data.rows[gi]['supervision_file']
            if not targetpath.exists():
                idxq = qtable.iloc[qgroups[int(meta['layer_idx'])]].set_index('anchor_nucleus_id')
                atomic_save(targetpath, raw_source.observations(uid, g, idxq))
            rawsup = torch.load(targetpath, map_location='cpu', weights_only=False)
            layer = by_uid.loc[uid]
            track = lookup[lookup.track_id == meta['track_id']].set_index('section_id')
            idx = index[index.anchor_layer_idx == meta['layer_idx']].set_index('anchor_nucleus_id')
            if split == 'train':
                weights = torch.load(paths()['result'] / '02_inputs' / data.rows[gi]['supervision_file'], weights_only=True, map_location='cpu')
            else:
                aligned = idx.reindex(g.nucleus_id.numpy())
                weights = {'nucleus_id': g.nucleus_id}
                for kind in ('nucleus', 'cell'):
                    ob = rawsup[kind]
                    w = torch.ones_like(ob['weight'])
                    for plane, side in ((0, 'lower'), (2, 'upper')):
                        keep = ob['plane_index'] == plane
                        quality = aligned[side + '_link_class'].to_numpy()[ob['anchor_node_index'][keep].numpy()]
                        assert np.isin(quality, ['gold', 'silver_nucleus']).all()
                        w[keep] = torch.tensor(np.where(quality == 'gold', 1.0, 0.25), dtype=w.dtype)
                    weights[kind + '_weight'] = w
            assert torch.equal(weights['nucleus_id'], g.nucleus_id)
            for kind in ('nucleus', 'cell'):
                rawsup[kind]['weight'] = weights[kind + '_weight']
            sup = move(Supervision(EllipseObservationBatch(**rawsup['nucleus']), EllipseObservationBatch(**rawsup['cell']), rawsup['confirmed_empty_mask']), 'cuda')
            g = move(g, 'cuda')
            geo = value['geometry']
            t = value['tables']
            labels = dict(value['labels'])
            if split != 'test':
                obs = observation_tables(geo, g, sup, 9, 32768)
                cost = obs['projection'] + obs['positive_gap']
                unary = cost + t.prior
                reference, refsearch = solve_scene(t, unary, g.graph_ptr, info['config'])
                labels = {**value['labels'], 'reference': reference}
            raw = []
            scenes = []
            n = len(g.nucleus_id)
            node = torch.arange(n, device='cuda')
            valid = t.valid
            targets = {kind: prepare_targets(rawsup[kind], g.nucleus_id.cpu().numpy(), idx, layer, track, kind, 'cuda') for kind in ('nucleus', 'cell')}
            for mode, lab in labels.items():
                ids = node * 9 + lab
                scene = Scene(geo.nucleus.select(ids), geo.cell.select(ids), valid, geo.raw[ids])
                rr = dict(mode=mode, split=split, graph_id=uid, track_id=meta['track_id'], nodes=n, valid=int(valid.sum()))
                e = torch.arange(len(t.i), device='cuda')
                pen = t.contact[e, lab[t.i], lab[t.j]]
                rr['collision'] = {}
                for scale in (1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9):
                    hit = torch.zeros_like(valid)
                    flag = pen > (1 - scale if scale < 1 else 1e-05)
                    hit[t.i[flag]] = True
                    hit[t.j[flag]] = True
                    rr['collision'][str(scale)] = float(hit.sum() / valid.sum().clamp_min(1))
                rr['penetration_per_valid'] = float(pen.sum() / valid.sum().clamp_min(1))
                for kind, cap, ratio in [('nucleus', 800, 3), ('cell', 4000, 6)]:
                    body = getattr(scene, kind)
                    st = physical_stats(body)
                    rr[kind + '_z_std'] = float(body.center[valid, 2].std(unbiased=False))
                    rr[kind + '_stats'] = dict(volume_mean=float(st['volume'][valid].mean()), volume_max=float(st['volume'][valid].max()), rms_ratio_max=float(st['aspect'][valid].max()), violations=int((((st['volume'] > cap * (1 + 1e-05)) | (st['aspect'] > ratio * (1 + 1e-05))) & valid).sum()))
                    for row in raw_rows(body, targets[kind], valid):
                        row.update(mode=mode, graph_id=uid, track_id=meta['track_id'], kind=kind)
                        raw.append(row)
                tn = scene.nucleus.transform.double()
                tc = scene.cell.transform.double()
                dmat = torch.linalg.solve(tn, tc)
                shift = torch.linalg.solve(tn, (scene.cell.center.double() - scene.nucleus.center.double())[:, :, None])[:, :, 0]
                margin = dmat.diagonal(dim1=-2, dim2=-1).amin(-1) - 1
                scaled = shift / margin.clamp_min(1e-12)[:, None]
                bad = (margin < -1e-05) | (scaled[:, :2].square().sum(-1) + scaled[:, 2].pow(4) > 1 + 0.0001)
                rr['containment_violations'] = int((bad & valid).sum())
                scenes.append(rr)
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(raw).to_csv(dest / 'raw.csv.gz', index=False)
            atomic_json(dest / 'scene.json', scenes)
            if split != 'test':
                atomic_save(dest / 'reference.pt', dict(labels=reference.cpu(), search=refsearch))
            atomic_json(dest / 'complete.json', record(uid=uid))
            done += 1
            if done % 8 == 0:
                atomic_json(RESULT / 'runtime' / ('evaluate_%d_status.json' % shard), record(done=done, total=(len(data) + shards - 1 - shard) // shards, seconds=time.time() - started))
                print('EVALUATE', shard, done, flush=True)
    atomic_json(RESULT / 'runtime' / ('evaluate_%d_done.json' % shard), record(done=done, seconds=time.time() - started))
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
