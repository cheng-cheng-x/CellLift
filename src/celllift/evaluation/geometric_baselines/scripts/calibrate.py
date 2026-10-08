from celllift.runtime import resource_path as _public_resource
import json, time, sys, itertools, math
from concurrent.futures import ThreadPoolExecutor
import torch
from celllift.evaluation.geometric_baselines.src.common import RESULT, CONFIG, layout, write, save, record
from celllift.evaluation.geometric_baselines.src.data import rows, load
from celllift.evaluation.geometric_baselines.src.geometry import coefficients, decode, stats, contact, contact_pairs, slab_factor

def student(x):
    return 2.5 * torch.log1p((x / 0.46).square() / 4)

@torch.no_grad()
def main(method):
    layout()
    torch.set_num_threads(4)
    start = time.time()
    torch.cuda.set_device(0)
    dest = RESULT / '03_calibration' / method
    if (dest / 'complete.json').exists():
        return
    rr = rows('train')
    roots = []
    centers = []
    norm = []
    which = []
    plane = []
    kindid = []
    target = []
    support = []
    weight = []
    left = []
    right = []
    offset = 0
    dirs = torch.stack((torch.cos(torch.arange(36, dtype=torch.float64) * 2 * math.pi / 36), torch.sin(torch.arange(36, dtype=torch.float64) * 2 * math.pi / 36)), 1).cuda()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for ix, item in enumerate(pool.map(lambda x: load(x, True), rr)):
            root = item['root'].cuda()
            center = item['center'].cuda()
            n = len(root)
            cf = coefficients(root, method)
            pairs = contact_pairs(center, cf['Q'] * 2.3 ** 2, cf['valid'])
            left.append(pairs[0] + offset)
            right.append(pairs[1] + offset)
            roots.append(root)
            centers.append(center)
            norm.append(root.new_full((n,), 1 / n / len(rr)))
            for kid, kind in enumerate(['nucleus', 'cell']):
                ob = {k: v.cuda() for k, v in item['supervision'][kind].items()}
                keep = ob['weight'] > 0
                if kid == 0:
                    keep &= ob['plane_index'] != 1
                ids = torch.where(keep)[0]
                owner = ob['anchor_node_index'][ids]
                L = root[owner]
                tc = (L @ ob['target_center_normalized'][ids].double()[:, :, None])[:, :, 0]
                tr = L @ ob['target_root_normalized'][ids].double()
                target.append((tc @ dirs.T + torch.einsum('nji,kj->nki', tr, dirs).norm(dim=-1)).float())
                support.append(torch.einsum('nji,kj->nki', L, dirs).norm(dim=-1).float())
                which.append(owner + offset)
                plane.append(ob['plane_index'][ids])
                kindid.append(torch.full_like(owner, kid))
                weight.append((ob['weight'][ids] / n / len(rr)).float())
            offset += n
            if ix % 128 == 0:
                print('LOAD', method, ix, flush=True)
    root = torch.cat(roots)
    center = torch.cat(centers)
    normal = torch.cat(norm)
    cf = coefficients(root, method)
    owner = torch.cat(which)
    planes = torch.cat(plane)
    kind = torch.cat(kindid)
    tar = torch.cat(target)
    base = torch.cat(support)
    ww = torch.cat(weight) * cf['valid'][owner]
    i = torch.cat(left)
    j = torch.cat(right)
    cache = {}
    trials = []

    def objective(params):
        k, l, g = params
        geo = decode(cf, params, method)
        h = torch.where(kind == 0, geo['h'][owner], geo['hc'][owner])
        lam = torch.where(kind == 0, torch.ones_like(h), geo['lam'][owner])
        factor, exists, gap = slab_factor(h, planes, method, True)
        observation = normal.new_zeros(())
        for st in range(0, len(owner), 16384):
            sl = slice(st, st + 16384)
            pred = base[sl] * (lam[sl] * factor[sl]).float()[:, None]
            observation += (student(pred - tar[sl]).mean(1) * ww[sl]).sum().double()
        observation += (student(gap.float()) * ww).sum().double()
        prior = sum(((stats(cf, geo, method, name)['prior'] * normal * cf['valid']).sum() for name in ['nucleus', 'cell']))
        key = (k, l)
        if key not in cache:
            Q = cf['Q'] * geo['lam'][:, None, None].square()
            energy = normal.new_zeros(())
            hits = torch.zeros_like(cf['valid'])
            depth = energy.clone()
            for st in range(0, len(i), 16384):
                a = i[st:st + 16384]
                b = j[st:st + 16384]
                alpha = contact(Q[a], Q[b], center[b] - center[a])
                pen = (1 - alpha).clamp_min(0)
                energy += (10 * pen * normal[a]).sum()
                flag = pen > 0.02
                hits[a[flag]] = True
                hits[b[flag]] = True
                depth += ((pen - 0.02).clamp_min(0) * normal[a]).sum()
            cache[key] = (float(energy), float((hits * normal).sum()), float(depth))
        ce, hit, depth = cache[key]
        value = float(observation + prior) + ce
        row = dict(parameters=list(params), energy=value, observation=float(observation), prior=float(prior), contact=ce, core_incidence_per_original_node=hit, core_depth_per_original_node=depth)
        trials.append(row)
        return row
    grid = list(itertools.product(CONFIG['height_grid'], CONFIG['xy_grid'], CONFIG['z_grid']))
    for step, p in enumerate(grid):
        objective(p)
        if step % 10 == 0:
            write(RESULT / 'runtime' / ('calibrate_' + method + '.json'), record(done=step + 1, total=177, best=min((x['energy'] for x in trials)), seconds=time.time() - start))
            print('GRID', method, step, flush=True)
    coarse = min(trials, key=lambda x: x['energy'])
    p = coarse['parameters']
    f = CONFIG['refinement_factor']
    fine = sorted(set(((p[0] * a, max(1, p[1] * b), max(1, p[2] * c)) for a, b, c in itertools.product([1 / f, 1.0, f], repeat=3))))
    for pp in fine:
        objective(pp)
    best = min(trials, key=lambda x: x['energy'])
    write(dest / 'trials.json', trials)
    write(dest / 'complete.json', record(method=method, best=best, coarse_best=coarse, train_graphs=len(rr), nodes=len(root), valid=int(cf['valid'].sum()), trial_count=len(trials), seconds=time.time() - start, policy='TRAIN-only O+P+10 linear contact; no VAL model selection'))
    print('CALIBRATED', method, best, flush=True)
if __name__ == '__main__':
    main(sys.argv[1])
