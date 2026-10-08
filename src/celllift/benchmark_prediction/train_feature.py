from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import nn
from .io import job_dir, reused, save_model, seed_all, write_pass
from .metrics import macro_f1, patient_auroc
from .train_geometry import DualSet, _bags, _route_cfg, _stack_bag
from celllift.morphology_interaction.data import SceneCache

def train_feature(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    dataset = job['dataset']
    cache = SceneCache(_route_cfg(dataset), dataset)
    cache.preload()
    arm = 'G2' if job['arm'] in {'H2', 'C1'} else 'G3'
    fit = _bags(dataset, cache, list(map(str, job['official_fit'])))
    val = _bags(dataset, cache, list(map(str, job['official_val'])))
    sample = _stack_bag(cache, fit[0], arm, None)
    classes = 7 if dataset == 'bracs' else 1
    seed_all(int(job['seed']))
    geom = DualSet(sample[0].shape[1], sample[1].shape[1], 32, 'deepsets').to(device)
    head = nn.Sequential(nn.Linear(32 + 384, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, classes)).to(device)
    opt = torch.optim.AdamW([{'params': geom.parameters(), 'lr': 0.0001}, {'params': head.parameters(), 'lr': 2e-05}], weight_decay=0.0001)
    best, stale, history = (-1000000000.0, 0, [])

    def rgb_of(bag):
        vecs = []
        for graph_id in bag['graph_ids']:
            graph = cache.get(graph_id)
            dino = np.asarray(graph.dino, np.float32)
            include = np.asarray(graph.include, bool)
            if include.any():
                dino = dino[include]
            vecs.append(dino.mean(0) if dino.size else np.zeros(384, np.float32))
        return np.mean(np.stack(vecs, 0), 0)

    def forward(bag):
        a, b = _stack_bag(cache, bag, arm, None)
        g = geom(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
        rgb = torch.from_numpy(rgb_of(bag)).to(device)
        return head(torch.cat([g, rgb], 0))

    def eval_bags(bags):
        logits, labels = ([], [])
        geom.eval()
        head.eval()
        with torch.no_grad():
            for bag in bags:
                logits.append(forward(bag).detach().cpu().numpy())
                labels.append(bag['label_id'])
        return (np.stack(logits, 0), np.asarray(labels))
    for epoch in range(int(job.get('max_epochs', 80))):
        geom.train()
        head.train()
        running = 0.0
        for bag in fit:
            logit = forward(bag)
            label = torch.tensor(bag['label_id'], device=device)
            loss = nn.functional.cross_entropy(logit.unsqueeze(0), label.unsqueeze(0)) if classes > 1 else nn.functional.binary_cross_entropy_with_logits(logit.reshape(()), label.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
        val_logits, val_labels = eval_bags(val)
        if classes > 1:
            metric = macro_f1(val_labels, val_logits.argmax(1), classes)
        else:
            metric = patient_auroc(val_labels, 1 / (1 + np.exp(-val_logits.reshape(-1))))
        history.append({'epoch': epoch, 'train_loss': running / max(1, len(fit)), 'val_metric': metric})
        if np.isfinite(metric) and metric > best + 1e-12:
            best, stale = (metric, 0)
            save_model(destination, {'geom': geom.state_dict(), 'head': head.state_dict(), 'epoch': epoch, 'metric': metric})
        else:
            stale += 1
            if stale >= int(job.get('patience', 12)):
                break
    val_logits, val_labels = eval_bags(val)
    predictions = []
    for bag, logit, label in zip(val, val_logits, val_labels):
        if classes == 1:
            p = float(1 / (1 + np.exp(-float(np.asarray(logit).reshape(-1)[0]))))
            probability = [1 - p, p]
        else:
            z = np.asarray(logit, np.float64)
            z = z - z.max()
            e = np.exp(z)
            probability = (e / e.sum()).tolist()
        predictions.append({'sample_id': bag['sample_id'], 'label_id': int(label), 'probability': probability, 'arm': job['arm']})
    return write_pass(destination, {'dataset': dataset, 'arm': job['arm'], 'best': best, 'history': history, 'frozen_rgb': 'tile_dino_mean', 'geometry': 'raw3d' if arm == 'G3' else 'mask2d'}, predictions)
