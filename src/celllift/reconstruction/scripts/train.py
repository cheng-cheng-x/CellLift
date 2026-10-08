from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import json
import time
import random
import traceback
import shutil
import datetime
from celllift.runtime import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from celllift.reconstruction.src.common import ROOT, RESULT, paths, atomic_json, atomic_save, record, digest
from celllift.reconstruction.src.data import InputDataset, move, collate, identity_collate
from celllift.reconstruction.src.model import ConditionalSceneNetwork
from celllift.reconstruction.src.learning import objective
from celllift.reconstruction.src.candidate_tables import observation_tables
from celllift.reconstruction.src.scorer import Scorer, learning_terms
from celllift.reconstruction.src.stage_audit import snapshot

class Plateau:

    def __init__(self):
        self.best = float('inf')
        self.best_epoch = 0
        self.base = float('inf')
        self.bad = 0
        self.tier = 0

    def update(self, value, epoch):
        improved = value < self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
        if epoch <= 2:
            self.base = min(self.base, value)
        elif value <= self.base * 0.995:
            self.base = value
            self.bad = 0
        else:
            self.bad += 1
        stop = epoch > 2 and self.bad >= 2 and (self.tier == 2)
        if epoch > 2 and self.bad >= 2 and (self.tier < 2):
            self.tier += 1
            self.bad = 0
        return (improved, stop)

def seed():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def rng():
    return dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())

def set_rng(r):
    random.setstate(r['python'])
    torch.set_rng_state(r['torch'].cpu())
    torch.cuda.set_rng_state(r['cuda'].cpu())

def main():
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2))
    barrier = lambda: dist.barrier() if world > 1 else None
    info = json.loads((RESULT / '02_inputs/manifest.json').read_text())
    cfg = info['config']
    dataset = InputDataset(paths(), 'train', True)
    validation = InputDataset(paths(), 'val', True)
    fit = list(range(len(dataset)))
    monitor = list(range(len(validation)))
    score_dataset = ConcatDataset([dataset, validation])
    score_rows = dataset.rows + validation.rows
    score_monitor = list(range(len(dataset), len(score_dataset)))
    assert info['protocol'] == 'train_val_test' and len(dataset) == 8813 and (len(validation) == 1102)

    def sync(x):
        if world > 1:
            dist.all_reduce(x)
        return x
    ntrain = len(dataset)

    def read_g(phase):
        assert phase == 'validation'
        folder = RESULT / '03_training' / ('geometry_' + phase)
        name = 'best.pt'
        state = torch.load(folder / name, map_location='cuda', weights_only=False)
        m = ConditionalSceneNetwork(info['full_stats'], cfg).cuda()
        m.load_state_dict(state['model'])
        return m

    def geometry(phase, stage='geometry'):
        folder = RESULT / '03_training' / (stage + '_' + phase)
        if (folder / 'complete.json').exists():
            return
        assert phase == 'validation'
        indices = fit
        schedule = None
        seed()
        m = ConditionalSceneNetwork(info['full_stats'], cfg, single=stage == 'shape').cuda()
        if stage == 'geometry':
            source = RESULT / '03_training' / ('shape_' + phase) / 'best.pt'
            inherited = torch.load(source, map_location='cuda', weights_only=False)
            transfer = m.transfer_shape(inherited['model'])
            if rank == 0:
                atomic_json(folder / 'transfer.json', dict(transfer, source=str(source), source_epoch=inherited['next_epoch'] - 1, external_weights=False))
            del inherited
        opt = torch.optim.AdamW(m.parameters(), lr=0.0003, weight_decay=0.0001)
        params = list(m.parameters())
        plateau = Plateau()
        history = []
        epoch0 = 1
        step0 = 0
        last = folder / ('rank%d_last.pt' % rank)
        if last.exists():
            state = torch.load(last, map_location='cuda', weights_only=False)
            if state['source_sha256'] != digest():
                raise RuntimeError('Source changed since checkpoint: explicit migration required')
            m.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            plateau.__dict__.update(state['plateau'])
            history = state['history']
            epoch0 = state['next_epoch']
            step0 = state['next_step']
            set_rng(state['rng'])
        elif rank == 0:
            atomic_save(folder / 'initial.pt', dict(model=m.state_dict(), normalization=info['full_stats'], source_sha256=digest()))
        if phase == 'validation' and epoch0 == 1 and (step0 == 0):
            if rank == 0:
                snapshot(m, dataset, info, cfg, stage, 'initial')
            barrier()
        localindices = indices[rank::world]
        batches = []
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=identity_collate, num_workers=4, pin_memory=True, persistent_workers=True, timeout=180)
        started = time.time()
        maxepoch = cfg[stage + '_max_epochs'] if schedule is None else len(schedule)
        stopped = bool(history and plateau.tier == 2 and (plateau.bad >= 2) and (phase == 'validation'))

        def save(next_epoch, next_step):
            atomic_save(last, dict(model=m.state_dict(), optimizer=opt.state_dict(), plateau=vars(plateau), history=history, next_epoch=next_epoch, next_step=next_step, rng=rng(), config=cfg, source_sha256=digest()))
        for epoch in range(epoch0, epoch0 if stopped else maxepoch + 1):
            used = cfg['lr'][plateau.tier] if schedule is None else schedule[epoch - 1]['lr']
            for pg in opt.param_groups:
                pg['lr'] = used
            order = list(indices)
            random.Random(42 + epoch).shuffle(order)
            globalb = [order[j:j + cfg['batch_size']] for j in range(0, len(order), cfg['batch_size'])]
            allb = [batch[rank::world] for batch in globalb]
            batches[:] = allb[step0:]
            m.train()
            epstart = time.time()
            for step, items in enumerate(loader, start=step0):
                opt.zero_grad(set_to_none=True)
                if items:
                    g, s = move(collate(items), 'cuda')
                    p = m.propose(g)
                    loss, values = objective(m, p, g, s, cfg, batch_graphs=len(globalb[step]))
                else:
                    loss = params[0].sum() * 0
                    values = dict(graphs=0)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite scene objective')
                loss.backward()
                flat = torch.cat([(x.grad if x.grad is not None else torch.zeros_like(x)).reshape(-1) for x in params])
                sync(flat)
                if not torch.isfinite(flat).all():
                    raise FloatingPointError('nonfinite geometry gradient')
                off = 0
                for x in params:
                    x.grad = flat[off:off + x.numel()].view_as(x)
                    off += x.numel()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
                opt.step()
                if step % 4 == 0 or step + 1 == len(allb):
                    row = record(phase=stage + '_' + phase, rank=rank, epoch=epoch, step=step + 1, steps=len(allb), lr=used, metrics=values, seconds=time.time() - epstart, gradient_norm=float(gn), peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
                    atomic_json(RESULT / 'runtime' / ('rank%d_status.json' % rank), row)
                    print(json.dumps(row), flush=True)
                del loss, flat
                if items:
                    del p, g, s
                if (step + 1) % 32 == 0:
                    save(epoch, step + 1)
            step0 = 0
            measured = {}
            improved = False
            stop = False
            if phase == 'validation':
                m.eval()
                total = {}
                with torch.no_grad():
                    for ix in monitor[rank::world]:
                        g, s = move(validation[ix], 'cuda')
                        p = m.propose(g)
                        _, met = objective(m, p, g, s, cfg, live=False)
                        for key, value in met.items():
                            total[key] = total.get(key, 0.0) + value
                keys = sorted(total)
                x = sync(torch.tensor([total[k] for k in keys], device='cuda', dtype=torch.float64))
                count = float(x[keys.index('graphs')])
                measured = {k: float(x[i] / count) for i, k in enumerate(keys)}
                improved, stop = plateau.update(measured['scene_free_energy'], epoch)
            history.append(dict(epoch=epoch, lr=used, seconds=time.time() - epstart, monitor=measured))
            save(epoch + 1, 0)
            barrier()
            if rank == 0:
                shutil.copyfile(last, folder / ('epoch_%03d.pt' % epoch))
                atomic_json(folder / 'history.json', history)
                if improved:
                    shutil.copyfile(last, folder / 'best.pt')
            if phase == 'validation' and epoch in (2, 7):
                if rank == 0:
                    snapshot(m, dataset, info, cfg, stage, str(epoch))
            barrier()
            if stop:
                stopped = True
                break
        if rank == 0:
            if phase == 'validation':
                out = dict(best_epoch=plateau.best_epoch, epochs=len(history), schedule=[dict(lr=x['lr']) for x in history[:plateau.best_epoch]], stop_reason='plateau' if stopped else 'maximum_epochs', selection_split='val')
            else:
                raise ValueError('Full replay is disabled under VAL selection')
            atomic_json(folder / 'complete.json', dict(out, seconds=time.time() - started))
        barrier()
        del m, opt, loader
        torch.cuda.empty_cache()

    def cache(phase):
        out = RESULT / '02_inputs' / ('score_' + phase)
        m = read_g(phase).eval()
        indices = list(range(len(score_dataset)))
        start = time.time()
        done = 0
        with torch.no_grad():
            for ix in indices[rank::world]:
                uid = score_rows[ix]['graph_id']
                dest = out / (uid + '.pt')
                if dest.exists():
                    continue
                g, s = move(score_dataset[ix], 'cuda')
                p = m.propose(g)
                features = m.features(g, p)
                obs = observation_tables(p.geometry, g, s, 9, cfg['observation_chunk'])
                cost = obs['projection'] + obs['positive_gap']
                observed = torch.zeros(len(g.nucleus_id), device='cuda', dtype=torch.bool)
                for kind in ('nucleus', 'cell'):
                    ob = getattr(s, kind)
                    keep = ob.weight > 0
                    if kind == 'nucleus':
                        keep &= ob.plane_index != 1
                    observed[ob.anchor_node_index[keep]] = True
                atomic_save(dest, move(dict(**features, observed_cost=cost, observed=observed, ids=g.nucleus_id, uid=uid), 'cpu'))
                done += 1
                if done % 8 == 0:
                    atomic_json(RESULT / 'runtime' / ('cache_%s_rank%d.json' % (phase, rank)), record(done=done, seconds=time.time() - start))
        barrier()
        del m
        torch.cuda.empty_cache()

    def scoring(phase):
        folder = RESULT / '03_training' / ('scorer_' + phase)
        if (folder / 'complete.json').exists():
            return
        if rank != 0:
            barrier()
            return
        start = time.time()
        cachepath = RESULT / '02_inputs' / ('score_' + phase)
        from collections import OrderedDict

        class CachedRows:

            def __init__(self):
                self.cache = OrderedDict()

            def __getitem__(self, ix):
                if ix not in self.cache:
                    self.cache[ix] = torch.load(cachepath / (score_rows[ix]['graph_id'] + '.pt'), weights_only=True, map_location='cpu')
                    if len(self.cache) > cfg['score_cpu_cache_graphs']:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(ix)
                return self.cache[ix]
        rows = CachedRows()
        assert phase == 'validation'
        indices = fit
        norm = {}
        for kind in ('node', 'candidate'):
            dim = 548 if kind == 'node' else 246
            sm = torch.zeros(dim, device='cuda', dtype=torch.float64)
            sq = sm.clone()
            count = 0
            for ix in indices:
                x = rows[ix][kind][rows[ix]['valid']].reshape(-1, dim).to(device='cuda', dtype=torch.float64)
                sm += x.sum(0)
                sq += x.square().sum(0)
                count += len(x)
            mean = sm / count
            norm[kind + '_mean'] = mean.float()
            norm[kind + '_std'] = (sq / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.0001).float()
        seed()
        model = Scorer(548, 246, norm).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        plateau = Plateau()
        history = []
        epoch0 = 1
        last = folder / 'last.pt'
        schedule = None
        if last.exists():
            state = torch.load(last, map_location='cuda', weights_only=False)
            if state['source_sha256'] != digest():
                raise RuntimeError('Scorer checkpoint source mismatch')
            model.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            plateau.__dict__.update(state['plateau'])
            history = state['history']
            epoch0 = state['epoch'] + 1
            set_rng(state['rng'])

        def packed(ids):
            chunks = [rows[i] for i in ids]
            sizes = [len(c['node']) for c in chunks]
            offset = [0]
            for n in sizes:
                offset.append(offset[-1] + n)
            x = {key: torch.cat([c[key] for c in chunks]) for key in ('node', 'candidate', 'valid', 'observed', 'observed_cost')}
            x['edge_index'] = torch.cat([c['edge_index'] + o for c, o in zip(chunks, offset)], 1)
            x = move(x, 'cuda')
            x['norm'] = torch.cat([torch.full((n,), 1 / n / len(ids), device='cuda') for n in sizes])
            return x
        maxepoch = cfg['scorer_max_epochs']
        stopped = bool(history and plateau.tier == 2 and (plateau.bad >= 2))
        for epoch in range(epoch0, epoch0 if stopped else maxepoch + 1):
            used = cfg['lr'][plateau.tier] if schedule is None else schedule[epoch - 1]
            for pg in opt.param_groups:
                pg['lr'] = used
            order = list(indices)
            random.Random(42 + epoch).shuffle(order)
            model.train()
            epstart = time.time()
            for j in range(0, len(order), 8):
                x = packed(order[j:j + 8])
                score = model(x)
                term, _, _, _, _ = learning_terms(score, x['observed_cost'], x['observed'], x['valid'], 'compatibility')
                loss = (term * x['norm']).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                opt.step()
            measured = {}
            improved = False
            stop = False
            if phase == 'validation':
                model.eval()
                sums = torch.zeros(3, device='cuda')
                with torch.no_grad():
                    for j in range(0, len(score_monitor), 8):
                        ids = score_monitor[j:j + 8]
                        x = packed(ids)
                        term, kl, reg, _, _ = learning_terms(model(x), x['observed_cost'], x['observed'], x['valid'], 'compatibility')
                        sums += torch.stack([(z * x['norm']).sum() * len(ids) / len(score_monitor) for z in (term, kl, reg)])
                measured = dict(zip(['cross_entropy', 'kl', 'regret'], sums.cpu().tolist()))
                improved, stop = plateau.update(measured['cross_entropy'], epoch)
            history.append(dict(epoch=epoch, lr=used, seconds=time.time() - epstart, monitor=measured))
            atomic_save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), normalization=norm, epoch=epoch, history=history, plateau=vars(plateau), rng=rng(), source_sha256=digest()))
            shutil.copyfile(last, folder / ('epoch_%03d.pt' % epoch))
            if improved:
                shutil.copyfile(last, folder / 'best.pt')
            atomic_json(folder / 'history.json', history)
            atomic_json(RESULT / 'runtime/rank0_status.json', record(phase='scorer_' + phase, epoch=epoch, lr=used, monitor=measured, seconds=time.time() - start))
            print('SCORER', phase, history[-1], flush=True)
            if stop:
                stopped = True
                break
        if phase == 'validation':
            out = dict(best_epoch=plateau.best_epoch, schedule=[x['lr'] for x in history[:plateau.best_epoch]], stop_reason='plateau' if stopped else 'maximum_epochs', selection_split='val')
        else:
            raise ValueError('Full replay is disabled under VAL selection')
        selected = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
        generator = read_g('validation')
        generator.attach_scorer(norm)
        generator.scorer.load_state_dict(selected['model'])
        atomic_save(RESULT / '03_training/final.pt', dict(model=generator.state_dict(), scorer_normalization=norm, config=cfg, source_sha256=digest(), normalization=info['full_stats'], selection_split='val', scorer_epoch=selected['epoch']))
        atomic_json(folder / 'complete.json', dict(out, epochs=len(history), seconds=time.time() - start))
        del model, opt, rows
        torch.cuda.empty_cache()
        barrier()
    geometry('validation', 'shape')
    geometry('validation')
    cache('validation')
    scoring('validation')
    atomic_json(RESULT / 'runtime' / ('train_rank%d_done.json' % rank), record(status='TRAINED', rank=rank))
    if world > 1:
        torch.cuda.set_device(local)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[local])
        dist.destroy_process_group()
if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic_json(RESULT / 'runtime' / ('train_failure_%d_%d.json' % (int(os.environ.get('RANK', 0)), time.time_ns())), record(traceback=traceback.format_exc()))
        raise
