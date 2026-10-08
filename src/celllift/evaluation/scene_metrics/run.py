from celllift.runtime import resource_path as _public_resource
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
from .common import ROOT, RESULT, BASE, MODEL, MODEL_CODE, SCALES, CONFIG, write, record, source_protocol
sys.path.append(str(MODEL_CODE))
from .geometry import pairs, contact_pen, max_penalty, volume_overlap

def summarize_observations(raw, valid, maximum, method, mode):
    x = raw[raw['mode'] == mode].copy()
    owners = torch.as_tensor(x.node.to_numpy(), device=valid.device)
    assert np.array_equal(x.geometry_valid.to_numpy(), valid[owners].cpu().numpy())
    x['view'] = x.kind + np.where(x.plane == 1, '_middle', '_adjacent')
    d = torch.as_tensor(x.dice.to_numpy(), device=valid.device)
    for scale in SCALES:
        tol = 1 - scale if scale < 1 else 1e-05
        good = valid & (maximum <= tol)
        x['j_' + str(scale)] = (d * good[owners]).cpu().numpy()
        x['eligible_' + str(scale)] = good[owners].cpu().numpy()
    x['nonempty'] = ~x.empty_prediction
    ag = dict(n=('dice', 'size'), dice_sum=('dice', 'sum'), nonempty=('nonempty', 'sum'))
    for scale in SCALES:
        ag['j_' + str(scale)] = ('j_' + str(scale), 'sum')
        ag['eligible_' + str(scale)] = ('eligible_' + str(scale), 'sum')
    out = x.groupby(['view']).agg(**ag).reset_index()
    out['method'] = method
    out['mode'] = mode
    dual = []
    for kind in ['nucleus', 'cell']:
        sub = x[(x.kind == kind) & (x.plane != 1)]
        d = sub.pivot(index='node', columns='plane', values='dice')
        if 0 in d and 2 in d:
            both = d.dropna()
            ne = sub.pivot(index='node', columns='plane', values='nonempty').loc[both.index]
            dual.append(dict(kind=kind, n=len(both), both_nonempty=int((ne[0] & ne[2]).sum()), minimum_dice_sum=float(both.min(1).sum())))
    return (out, dual)

def scene_stats(c, T, nuc_z, valid, edge, pen, method, mode, integrate, coaxial=False):
    maximum = max_penalty(len(c), edge, pen)
    rates = {str(s): float(((maximum > (1 - s if s < 1 else 1e-05)) & valid).sum() / valid.sum().clamp_min(1)) for s in SCALES}
    row = dict(method=method, mode=mode, nodes=len(c), valid=int(valid.sum()), collision=rates, penetration_per_valid=float(pen.sum() / valid.sum().clamp_min(1)), nucleus_z_std=float(nuc_z[valid].std(unbiased=False)))
    if integrate:
        broad = pairs(c, T, valid, 'p4' if method == 'v16c' else method)
        row['original'] = volume_overlap(c, T, valid, broad, 'p4' if method == 'v16c' else method)
        if coaxial:
            row['flattened'] = dict(row['original'])
            row['flattened_collision'] = rates
            row['flattened_is_exact_noop'] = True
        else:
            flat = c.clone()
            flat[:, 2] += 2.5 - nuc_z
            pe = pairs(flat, T, valid, 'p4')
            pn = contact_pen(flat, T, pe)
            mx = max_penalty(len(c), pe, pn)
            row['flattened'] = volume_overlap(flat, T, valid, pe, 'p4')
            row['flattened_collision'] = {str(s): float(((mx > (1 - s if s < 1 else 1e-05)) & valid).sum() / valid.sum().clamp_min(1)) for s in SCALES}
            row['flattened_is_exact_noop'] = False
    return (row, maximum)

@torch.no_grad()
def evaluate_roi(split, uid):
    start = time.time()
    dest = RESULT / '05_evaluation' / split / uid
    frames = []
    scenes = []
    node_cache = {}
    track = None
    for method in ['cylinder', 'ellipsoid', 'p4']:
        p = torch.load(BASE / '04_predictions' / method / split / (uid + '.pt'), map_location='cpu', weights_only=False)
        raw = pd.read_csv(BASE / '05_evaluation' / method / split / uid / 'raw.csv.gz')
        old = {x['mode']: x for x in json.loads((BASE / '05_evaluation' / method / split / uid / 'scene.json').read_text())}
        root = p['root'].double().cuda()
        xy = p['center'].double().cuda()
        n = len(xy)
        track = p['metadata']['track_id']
        center = torch.cat((xy, xy.new_full((n, 1), 2.5)), 1)
        previous = None
        for mode in ['short', 'main', 'long']:
            b = {k: v.cuda() for k, v in p['bodies'][mode].items()}
            valid = b['valid']
            T = root.new_zeros((n, 3, 3))
            T[:, :2, :2] = root * b['lam'][:, None, None]
            T[:, 2, 2] = b['hc']
            T = torch.where(valid[:, None, None], T, torch.eye(3, device='cuda', dtype=T.dtype))
            if previous is not None and torch.equal(previous[0], b['lam']):
                edge, pen = previous[1:]
            else:
                edge = pairs(center, T, valid, method)
                pen = contact_pen(center, T, edge, coaxial=True)
                previous = (b['lam'].clone(), edge, pen)
            sr, maximum = scene_stats(center, T, center[:, 2], valid, edge, pen, method, mode, mode == 'main', True)
            for scale in SCALES:
                assert abs(sr['collision'][str(scale)] - old[mode]['collision'][str(scale)]) < 2e-06, (uid, method, mode, scale)
            obs, dual = summarize_observations(raw, valid, maximum, method, mode)
            sr['dual_observations'] = dual
            scenes.append(sr)
            frames.append(obs)
            node_cache[method + '_' + mode] = dict(valid=valid.cpu(), maximum_penetration=maximum.cpu())
    p = torch.load(MODEL / '04_predictions' / split / (uid + '.pt'), map_location='cpu', weights_only=False)
    raw = pd.read_csv(MODEL / '05_evaluation' / split / uid / 'raw.csv.gz')
    old = {x['mode']: x for x in json.loads((MODEL / '05_evaluation' / split / uid / 'scene.json').read_text())}
    assert p['metadata']['track_id'] == track
    labels = p['labels']
    n = len(p['ids'])
    node = torch.arange(n)
    valid = p['tables'].valid.cuda()
    edge = torch.stack((p['tables'].i, p['tables'].j)).cuda()
    ee = torch.arange(len(edge[0]))
    for mode in ['main', 'soft']:
        lab = labels[mode]
        ids = node * 9 + lab
        geo = p['geometry']
        c = geo.cell.center[ids].double().cuda()
        T = geo.cell.transform[ids].double().cuda()
        z = geo.nucleus.center[ids, 2].double().cuda()
        pen = p['tables'].contact[ee, lab[p['tables'].i], lab[p['tables'].j]].cuda()
        sr, maximum = scene_stats(c, T, z, valid, edge, pen, 'v16c', mode, True)
        for scale in SCALES:
            assert abs(sr['collision'][str(scale)] - old[mode]['collision'][str(scale)]) < 2e-06, (uid, 'v16c', mode, scale)
        obs, dual = summarize_observations(raw, valid, maximum, 'v16c', mode)
        sr['dual_observations'] = dual
        scenes.append(sr)
        frames.append(obs)
        node_cache['v16c_' + mode] = dict(valid=valid.cpu(), maximum_penetration=maximum.cpu())
    for s in scenes:
        s.update(graph_id=uid, track_id=track, split=split)
    dest.mkdir(parents=True, exist_ok=True)
    observations = pd.concat(frames, ignore_index=True)
    observations['graph_id'] = uid
    observations['track_id'] = track
    observations.to_csv(dest / 'observations.csv.gz', index=False)
    torch.save(node_cache, dest / 'node_metrics.pt')
    write(dest / 'scene.json', scenes)
    write(dest / 'complete.json', record(uid=uid, seconds=time.time() - start))
    return time.time() - start

def main(shard, world):
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    assert (RESULT / '01_validation/complete.json').exists()
    started = time.time()
    all_rows = []
    for split in ['test']:
        files = sorted((BASE / '05_evaluation/p4' / split).glob('*/complete.json'))
        assert len(files) == CONFIG['splits'][split]
        all_rows.extend(((split, p.parent.name) for p in files))
    todo = all_rows[shard::world]
    done = 0
    write(RESULT / 'runtime' / ('worker_%d.json' % shard), record(status='RUNNING', total=len(todo), config=CONFIG))
    for split, uid in todo:
        if not (RESULT / '05_evaluation' / split / uid / 'complete.json').exists():
            seconds = evaluate_roi(split, uid)
        else:
            seconds = 0
        done += 1
        if done % 8 == 0:
            status = record(done=done, total=len(todo), seconds=time.time() - started, last_roi_seconds=seconds, split=split)
            write(RESULT / 'runtime' / ('progress_%d.json' % shard), status)
            print(json.dumps(status), flush=True)
    write(RESULT / 'runtime' / ('worker_%d_done.json' % shard), record(done=done, seconds=time.time() - started))
if __name__ == '__main__':
    try:
        main(int(sys.argv[1]), int(sys.argv[2]))
    except Exception as exc:
        import traceback
        write(RESULT / 'runtime' / ('failure_' + str(time.time_ns()) + '.json'), record(error=str(exc), traceback=traceback.format_exc()))
        raise
