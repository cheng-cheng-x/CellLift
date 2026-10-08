from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import random
import time
from celllift.runtime import torch
from .common import RESULT, atomic_json, record
from .data import move
from .geometry import physical_stats
from .candidate_tables import observation_tables
from .model import solve_scene
from .learning import metrics

@torch.no_grad()
def snapshot(model, dataset, info, config, stage, epoch):
    dest = RESULT / '07_diagnosis' / stage / ('%s.json' % epoch)
    if dest.exists():
        return
    groups = {}
    for item in info['items']:
        if item['split'] == 'train':
            groups.setdefault(item['track_id'], []).append(item)
    tracks = sorted(groups)
    random.Random(42).shuffle(tracks)
    items = [min(groups[t], key=lambda x: x['uid']) for t in tracks[:8]]
    start = time.time()
    was_training = model.training
    model.eval()
    rows = []
    for item in items:
        g, s = move(dataset[item['index']], 'cuda')
        p = model.propose(g)
        t = p.tables
        n, k = p.raw.shape[:2]
        valid = t.valid
        unavoidable = torch.zeros_like(valid)
        bad = t.contact.amin((1, 2)) > 0.02
        unavoidable[t.i[bad]] = True
        unavoidable[t.j[bad]] = True
        row = dict(uid=item['uid'], nodes=n, valid=int(valid.sum()), unavoidable_c98=float(unavoidable.sum() / valid.sum().clamp_min(1)))
        for kind in ('nucleus', 'cell'):
            body = getattr(p.geometry, kind)
            st = physical_stats(body)
            z = body.center[:, 2].reshape(n, k)
            row[kind] = dict(volume=float(st['volume'][p.geometry.valid].mean()), halfheight=float(st['axial_halfheight'][p.geometry.valid].mean()), rms_ratio=float(st['aspect'][p.geometry.valid].mean()), candidate_z_range=float((z.max(1).values - z.min(1).values)[valid].mean()))
        row['shape_raw_candidate_std'] = float(p.raw[:, :, 1:][valid].std(1, unbiased=False).mean())
        obs = observation_tables(p.geometry, g, s, k, config['observation_chunk'])
        if k == 9:
            labels, _ = solve_scene(t, obs['projection'] + obs['positive_gap'] + t.prior, g.graph_ptr, config)
        else:
            labels = torch.zeros(n, device=p.raw.device, dtype=torch.long)
        row['reference_lex_diagnostic'] = metrics(p, g, obs, labels)
        rows.append(row)
    model.train(was_training)
    atomic_json(dest, record(stage=stage, epoch=epoch, rows=rows, seconds=time.time() - start, scope='same eight TRAIN tracks; reference-assisted diagnostic only, excluded from VAL selection decisions'))
