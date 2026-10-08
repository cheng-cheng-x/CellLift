from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import nn
from celllift.morphology_interaction.data import SceneCache
from celllift.morphology_interaction.features import NODE_2D, NODE_3D
from .io import job_dir, reused, save_model, seed_all, write_pass
from .metrics import macro_f1, patient_auroc, qwk
from .paths import load_config, result_root
HIDDEN = 128

class DualSet(nn.Module):

    def __init__(self, dim_a: int, dim_b: int, classes: int, kind: str):
        super().__init__()
        self.kind = kind
        self.enc_a = nn.Sequential(nn.Linear(dim_a, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU()) if kind == 'deepsets' else None
        self.enc_b = nn.Sequential(nn.Linear(dim_b, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU()) if kind == 'deepsets' else None
        in_dim = (HIDDEN if kind == 'deepsets' else dim_a) + (HIDDEN if kind == 'deepsets' else dim_b) + 2
        self.fuse = nn.Sequential(nn.Linear(in_dim, HIDDEN), nn.SiLU(), nn.Dropout(0.1))
        self.head = nn.Linear(HIDDEN, classes)

    def _pool(self, tokens, encoder):
        if tokens.numel() == 0:
            width = HIDDEN if encoder is not None else 1
            return (tokens.new_zeros(width), tokens.new_zeros(()))
        hidden = encoder(tokens) if encoder is not None else tokens
        pooled = hidden.mean(0)
        return (pooled, torch.log1p(tokens.new_tensor(float(tokens.shape[0]))))

    def forward(self, nuc, cell):
        a, ca = self._pool(nuc, self.enc_a)
        b, cb = self._pool(cell, self.enc_b)
        fused = self.fuse(torch.cat([a, b, ca.reshape(-1), cb.reshape(-1)], 0))
        return self.head(fused)

def _route_cfg(dataset: str) -> dict[str, Any]:
    cfg = load_config(dataset)
    return {'paths': cfg['paths'], 'cache_batch': cfg.get('cache_batch') or 'parallel-set_encoding'}

def _residual_path(job: Mapping[str, Any]):
    fold = int(job.get('fold') or 0)
    return result_root() / job['dataset'] / 'residual' / 'probe' / job.get('unit', 'u') / f"seed_{int(job['seed'])}" / f'fold_{fold:02d}' / 'best.pt'

def _finite(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(array, np.float32), copy=False, nan=0.0, posinf=0.0, neginf=0.0)

def _sanitize_node(graph) -> tuple[np.ndarray, np.ndarray]:
    two = _finite(graph.node2d)
    three = np.asarray(graph.node3d, np.float32).copy()
    valid = np.asarray(getattr(graph, 'valid3d', np.ones(three.shape[0], bool)), bool)
    if valid.size == three.shape[0] and (~valid).any():
        three[~valid] = 0.0
    return (two, _finite(three))

def _channels(arm: str, node: np.ndarray, residual: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    two = _finite(node[:, :NODE_2D])
    three = node[:, NODE_2D:NODE_2D + NODE_3D] if node.shape[1] >= NODE_2D + NODE_3D else node[:, NODE_2D:]
    three = _finite(three)
    if residual is None:
        residual = np.zeros_like(three)
    residual = _finite(residual)
    empty = np.zeros((two.shape[0], 1), np.float32)
    if arm == 'G2':
        return (two, empty)
    if arm == 'G3':
        return (three, np.zeros((three.shape[0], 1), np.float32))
    if arm == 'GR':
        return (residual, np.zeros((residual.shape[0], 1), np.float32))
    if arm in {'G2R', 'G2+raw', 'G2raw'}:
        return (two, three)
    return (two, residual)

def _official_labels(dataset: str) -> dict[str, int]:
    from celllift.runtime import json
    from .paths import split_root
    labels: dict[str, int] = {}
    root = split_root() / dataset
    for name in ('train_rows.json', 'val_rows.json', 'test_rows.json'):
        path = root / name
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding='utf-8')):
            if row.get('label_id') in (None, ''):
                continue
            lid = int(row['label_id'])
            labels[str(row['sample_id'])] = lid
            if row.get('graph_id'):
                labels[str(row['graph_id'])] = lid
    return labels

def _bags(dataset: str, cache: SceneCache, ids: list[str]) -> list[dict[str, Any]]:
    official = _official_labels(dataset)
    if dataset == 'sicapv2':
        bags = []
        for graph_id in ids:
            if graph_id not in cache.graphs:
                continue
            meta = (cache.entries.get(graph_id) or {}).get('metadata') or {}
            label = official.get(graph_id)
            if label is None:
                label = int(meta.get('label_id') or 0)
            bags.append({'sample_id': graph_id, 'graph_ids': [graph_id], 'label_id': int(label)})
        return bags
    key = 'roi_id' if dataset == 'bracs' else 'patient_id'
    grouped = defaultdict(list)
    labels = {}
    wanted = set(map(str, ids))
    for graph_id, entry in cache.entries.items():
        meta = entry.get('metadata') or {}
        bag_id = str(meta.get(key) or graph_id)
        if bag_id not in wanted and graph_id not in wanted:
            continue
        grouped[bag_id].append(graph_id)
        if bag_id in official:
            labels[bag_id] = official[bag_id]
        elif meta.get('label_id') is not None:
            labels[bag_id] = int(meta['label_id'])
    return [{'sample_id': bag_id, 'graph_ids': graph_ids, 'label_id': labels.get(bag_id, official.get(bag_id, 0))} for bag_id, graph_ids in grouped.items()]

def _stack_bag(cache: SceneCache, bag: dict[str, Any], arm: str, residual_map: dict[str, np.ndarray] | None):
    nucs, cells = ([], [])
    for graph_id in bag['graph_ids']:
        graph = cache.get(graph_id)
        two, three = _sanitize_node(graph)
        include = np.asarray(graph.include, bool)
        if include.size == two.shape[0] and include.any():
            two, three = (two[include], three[include])
        residual = None if residual_map is None else residual_map.get(graph_id)
        if residual is not None:
            residual = _finite(residual)
            if include.size == residual.shape[0] and include.any():
                residual = residual[include]
        node = np.concatenate([two, three], 1)
        a, b = _channels(arm, node, residual)
        if a.size:
            nucs.append(a)
            cells.append(b)
    if not nucs:
        nucs = [np.zeros((1, 2), np.float32)]
        cells = [np.zeros((1, 2), np.float32)]
    return (np.concatenate(nucs, 0), np.concatenate(cells, 0))

def _metric(dataset: str, labels, logits):
    logits = np.asarray(logits)
    if not np.isfinite(logits).all():
        return float('nan')
    if dataset == 'sicapv2':
        return qwk(np.asarray(labels), logits.argmax(1))
    if dataset == 'bracs':
        return macro_f1(np.asarray(labels), logits.argmax(1), 7)
    scores = 1 / (1 + np.exp(-logits.reshape(-1)))
    return patient_auroc(np.asarray(labels), scores)

def train_residual(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing and (destination / 'best.pt').is_file() and (float(existing.get('best_loss', 1000000000.0)) < 100000000.0):
        return existing
    cache = SceneCache(_route_cfg(job['dataset']), job['dataset'])
    cache.preload()
    xs, ys = ([], [])
    for graph_id in job['official_fit']:
        if graph_id not in cache.graphs:
            continue
        graph = cache.get(graph_id)
        include = np.asarray(graph.include, bool)
        two, three = _sanitize_node(graph)
        if include.size == two.shape[0] and include.any():
            two, three = (two[include], three[include])
        if two.size:
            xs.append(two)
            ys.append(three)
    if job['dataset'] != 'sicapv2' and (not xs):
        bags = _bags(job['dataset'], cache, list(map(str, job['official_fit'])))
        for bag in bags:
            for graph_id in bag['graph_ids']:
                graph = cache.get(graph_id)
                include = np.asarray(graph.include, bool)
                two, three = _sanitize_node(graph)
                if include.size == two.shape[0] and include.any():
                    two, three = (two[include], three[include])
                if two.size:
                    xs.append(two)
                    ys.append(three)
    x = _finite(np.concatenate(xs, 0) if xs else np.zeros((1, NODE_2D), np.float32))
    y = _finite(np.concatenate(ys, 0) if ys else np.zeros((1, NODE_3D), np.float32))
    if x.shape[0] > 1000000:
        pick = np.random.default_rng(int(job['seed'])).choice(x.shape[0], 1000000, replace=False)
        x, y = (x[pick], y[pick])
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.SiLU(), nn.Linear(64, y.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, stale = (1000000000.0, 0)
    last_state = None
    for epoch in range(int(job.get('max_epochs', 80))):
        model.train()
        perm = np.random.permutation(x.shape[0])
        total = 0.0
        steps = 0
        for start in range(0, x.shape[0], 4096):
            idx = perm[start:start + 4096]
            pred = model(torch.from_numpy(x[idx]).to(device))
            target = torch.from_numpy(y[idx]).to(device)
            loss = nn.functional.smooth_l1_loss(pred, target)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            steps += 1
        value = total / max(1, steps)
        last_state = {'model': model.state_dict(), 'in_dim': x.shape[1], 'out_dim': y.shape[1], 'loss': value}
        if np.isfinite(value) and value < best:
            best, stale = (value, 0)
            save_model(destination, last_state)
        else:
            stale += 1
            if stale >= int(job.get('patience', 12)):
                break
    if not (destination / 'best.pt').is_file() and last_state is not None:
        save_model(destination, last_state)
        best = float(last_state['loss'])
    if not (destination / 'best.pt').is_file():
        raise RuntimeError('residual probe produced no checkpoint')
    return write_pass(destination, {'dataset': job['dataset'], 'arm': 'probe', 'best_loss': best, 'n': int(x.shape[0])})

def _load_residual_map(job, cache, device):
    path = _residual_path(job)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    model = nn.Sequential(nn.Linear(payload['in_dim'], 64), nn.SiLU(), nn.Linear(64, payload['out_dim']))
    model.load_state_dict(payload['model'])
    model.to(device).eval()
    mapping = {}
    with torch.no_grad():
        for graph_id, graph in cache.graphs.items():
            two, three = _sanitize_node(graph)
            pred = model(torch.from_numpy(two).to(device)).cpu().numpy()
            mapping[graph_id] = _finite(three - pred)
    return mapping

def train_geometry(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    dataset = job['dataset']
    cache = SceneCache(_route_cfg(dataset), dataset)
    cache.preload()
    arm = job['arm']
    residual_map = _load_residual_map(job, cache, device) if arm in {'GR', 'G2S', 'G2+residual'} else None
    fit = _bags(dataset, cache, list(map(str, job['official_fit'])))
    val = _bags(dataset, cache, list(map(str, job['official_val'])))
    if not fit:
        raise RuntimeError(f"empty official FIT bags for {job.get('job_id')}")
    sample = _stack_bag(cache, fit[0], arm, residual_map)
    classes = 4 if dataset == 'sicapv2' else 7 if dataset == 'bracs' else 1
    kind = job.get('encoder') or 'deepsets'
    seed_all(int(job['seed']))
    model = DualSet(sample[0].shape[1], sample[1].shape[1], classes, kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best, stale, history = (-1000000000.0, 0, [])
    max_epochs, patience = (int(job.get('max_epochs', 60)), int(job.get('patience', 10)))
    if job.get('fixed_epochs') is not None:
        max_epochs = int(job['fixed_epochs'])
        patience = 10 ** 9

    def logits_of(bags):
        out, labels = ([], [])
        model.eval()
        with torch.no_grad():
            for bag in bags:
                a, b = _stack_bag(cache, bag, arm, residual_map)
                logit = model(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
                out.append(logit.detach().cpu().numpy())
                labels.append(bag['label_id'])
        return (np.stack(out, 0), np.asarray(labels))
    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(int(job['seed']) + epoch * 1009)
        order = rng.permutation(len(fit))
        running = 0.0
        for index in order:
            bag = fit[int(index)]
            a, b = _stack_bag(cache, bag, arm, residual_map)
            logit = model(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
            label = torch.tensor(bag['label_id'], device=device)
            loss = nn.functional.cross_entropy(logit.unsqueeze(0), label.unsqueeze(0)) if classes > 1 else nn.functional.binary_cross_entropy_with_logits(logit.reshape(()), label.float())
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
        if val:
            val_logits, val_labels = logits_of(val)
            metric = _metric(dataset, val_labels, val_logits)
        else:
            metric = float('nan')
        history.append({'epoch': epoch, 'train_loss': running / max(1, len(fit)), 'val_metric': metric})
        if not val or job.get('fixed_epochs') is not None or (np.isfinite(metric) and metric > best + 1e-12):
            if np.isfinite(metric):
                best = metric
            stale = 0
            save_model(destination, {'model': model.state_dict(), 'epoch': epoch, 'metric': metric, 'arm': arm, 'kind': kind, 'dims': [sample[0].shape[1], sample[1].shape[1]]})
        else:
            stale += 1
            if stale >= patience:
                break
    if any((not np.isfinite(float(row.get('train_loss', np.nan))) for row in history)):
        raise RuntimeError(f"geometry train_loss was non-finite for {job.get('job_id') or arm}")
    if not (destination / 'best.pt').is_file():
        raise RuntimeError(f"geometry produced no checkpoint for {job.get('job_id') or arm}")
    saved = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    predictions = []
    if val:
        val_logits, val_labels = logits_of(val)
        for bag, logit, label in zip(val, val_logits, val_labels):
            if classes == 1:
                prob = float(1 / (1 + np.exp(-float(np.asarray(logit).reshape(-1)[0]))))
                probability = [1 - prob, prob]
            else:
                z = np.asarray(logit, np.float64)
                z = z - z.max()
                p = np.exp(z)
                probability = (p / p.sum()).tolist()
            predictions.append({'sample_id': bag['sample_id'], 'label_id': int(label), 'probability': probability, 'arm': job.get('arm_tag') or arm})
    return write_pass(destination, {'dataset': dataset, 'arm': job.get('arm_tag') or arm, 'encoder': kind, 'best': best, 'epochs': len(history), 'history': history, 'selected_epoch': int(saved.get('epoch', 0)), 'no_rgb': True, 'no_dino': True, 'no_baseline': True}, predictions)
