from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from . import SEED
from .io import job_dir, reused, save_model, seed_all, write_pass
from .crc_score import hard_vote_fraction, patient_auroc_hard_vote
from .metrics import macro_f1, patient_auroc, qwk, weighted_f1
from .paths import load_config

class _MemmapRGB(Dataset):

    def __init__(self, rows: list[dict[str, Any]], cache_root):
        from celllift.runtime import ResourcePath as Path
        self.rows = rows
        self.cache_root = Path(cache_root)
        self.arrays = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        shard = int(row['shard'])
        split = str(row.get('official_split') or '')
        key = (split, shard)
        if key not in self.arrays:
            folder = self.cache_root / 'official_test' if split == 'test' else self.cache_root
            self.arrays[key] = np.load(folder / f'rgb224_{shard:02d}.npy', mmap_mode='r')
        image = np.asarray(self.arrays[key][int(row['offset'])], np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).contiguous()
        return {'image': tensor, 'label': int(row['label_id']), 'graph_id': str(row['graph_id']), 'patient_id': str(row.get('patient_id') or '')}

def _select_ids(index_rows: list[dict[str, Any]], wanted: set[str], keys=('graph_id', 'patient_id')) -> list[dict[str, Any]]:
    output = []
    for row in index_rows:
        if any((str(row.get(key) or '') in wanted for key in keys)):
            output.append(row)
    return output

def _rgb_rows(cache_root, wanted: set[str], keys=('graph_id', 'patient_id')) -> list[dict[str, Any]]:
    from celllift.runtime import ResourcePath as Path
    from celllift.geometry_baselines.paper_rgb import read_rows
    rows = []
    root = Path(cache_root)
    for name in ('index.parquet', 'official_test/index.parquet'):
        path = root / name
        if not path.is_file():
            continue
        loaded = read_rows(str(path))
        from_test_dir = name.startswith('official_test')
        for row in loaded:
            item = dict(row)
            if from_test_dir and str(item.get('official_split') or '').lower() != 'test':
                item['official_split'] = 'test'
            rows.append(item)
    return _select_ids(rows, wanted, keys=keys)

def _train_sicap(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    from celllift.geometry_baselines.models import PaperFSConv
    from celllift.geometry_baselines.paper_rgb import _gpu_preprocess
    cfg = load_config('sicapv2')
    cache_root = cfg['paths']['paper_rgb_cache']
    fit = _rgb_rows(cache_root, set(map(str, job['official_fit'])))
    val = _rgb_rows(cache_root, set(map(str, job['official_val'])))
    seed_all(int(job['seed']))
    model = PaperFSConv().to(device)
    counts = Counter((int(row['label_id']) for row in fit))
    weights = torch.tensor([len(fit) / (4 * max(1, counts[i])) for i in range(4)], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0)
    best, stale, history = (-1000000000.0, 0, [])
    destination = job_dir(job)
    max_epochs, patience = (int(job.get('max_epochs', 200)), int(job.get('patience', 10)))
    if job.get('fixed_epochs') is not None:
        max_epochs = int(job['fixed_epochs'])
        patience = 10 ** 9
    if job.get('paper_recipe'):
        max_epochs = int(job.get('max_epochs', 200))
        patience = 10 ** 9
    generator = torch.Generator(device=device).manual_seed(int(job['seed']) + 73)
    if not val:
        patience = 10 ** 9
    for epoch in range(max_epochs):
        model.train()
        loader = DataLoader(_MemmapRGB(fit, cache_root), batch_size=32, shuffle=True, num_workers=2, drop_last=False)
        running = 0.0
        for batch in loader:
            images = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=True, generator=generator)
            labels = batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images)['logits'], labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
        model.eval()
        metric = float('nan')
        if val:
            logits, labels = ([], [])
            with torch.no_grad():
                loader = DataLoader(_MemmapRGB(val, cache_root), batch_size=64, shuffle=False, num_workers=2)
                for batch in loader:
                    images = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator)
                    logits.append(model(images)['logits'].float().cpu().numpy())
                    labels.extend((int(v) for v in batch['label'].numpy()))
            stacked = np.concatenate(logits, 0) if logits else np.zeros((0, 4))
            metric = qwk(np.asarray(labels), stacked.argmax(1))
        history.append({'epoch': epoch, 'train_loss': running / max(1, len(fit)), 'val_qwk': metric})
        improved = np.isfinite(metric) and metric > best + 1e-12
        lock_last = bool(job.get('paper_recipe') or (not val and job.get('fixed_epochs') is not None))
        if lock_last:
            if epoch == max_epochs - 1:
                save_model(destination, {'model': model.state_dict(), 'epoch': epoch, 'metric': metric, 'arm': 'B'})
            if improved:
                best = metric
            stale = 0
        elif not val or job.get('fixed_epochs') is not None or improved:
            if np.isfinite(metric):
                best = metric
            stale = 0
            save_model(destination, {'model': model.state_dict(), 'epoch': epoch, 'metric': metric, 'arm': 'B'})
        else:
            stale += 1
            if best <= 0 and job.get('fixed_epochs') is None:
                stale = min(stale, max(0, patience - 1))
            if stale >= patience:
                break
    saved = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    model.eval()
    predictions = []
    if val:
        with torch.no_grad():
            loader = DataLoader(_MemmapRGB(val, cache_root), batch_size=64, shuffle=False, num_workers=2)
            for batch in loader:
                images = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator)
                prob = torch.softmax(model(images)['logits'], 1).cpu().numpy()
                for graph_id, label, p in zip(batch['graph_id'], batch['label'].numpy(), prob):
                    predictions.append({'sample_id': graph_id, 'label_id': int(label), 'probability': p.tolist(), 'arm': 'B'})
    return write_pass(destination, {'dataset': 'sicapv2', 'arm': 'B', 'b_name': 'B', 'metric': 'patch_qwk', 'best': best, 'epochs': len(history), 'history': history, 'n_fit': len(fit), 'n_val': len(val), 'selected_epoch': int(saved.get('epoch', 0))}, predictions)

def _train_crc(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    from celllift.geometry_baselines.models import build_paper_crc_resnet18, crc_adam_groups
    from celllift.geometry_baselines.paper_rgb import _gpu_preprocess
    cfg = load_config('tcga_crc_msi')
    cache_root = cfg['paths']['paper_rgb_cache']
    fit_tiles = set(map(str, job.get('official_fit') or []))
    val_tiles = set(map(str, job.get('official_val') or []))
    fit = _rgb_rows(cache_root, fit_tiles, keys=('graph_id', 'patient_id'))
    val = _rgb_rows(cache_root, val_tiles, keys=('graph_id', 'patient_id'))
    seed_all(int(job['seed']))
    model, _names = build_paper_crc_resnet18(imagenet_weights=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(crc_adam_groups(model))
    criterion = nn.CrossEntropyLoss()
    destination = job_dir(job)
    generator = torch.Generator(device=device).manual_seed(int(job['seed']) + 73)
    best, stale, history = (-1000000000.0, 0, [])
    max_epochs, patience = (int(job.get('max_epochs', 100)), int(job.get('patience', 10)))

    def patient_metric(rows, logits):
        return patient_auroc_hard_vote([str(row['patient_id']) for row in rows], [int(row['label_id']) for row in rows], logits)
    for epoch in range(max_epochs):
        model.train()
        loader = DataLoader(_MemmapRGB(fit, cache_root), batch_size=128, shuffle=True, num_workers=4)
        running = 0.0
        for batch in loader:
            images = _gpu_preprocess(batch['image'].to(device), 'tcga_crc_msi', train=True, generator=generator)
            labels = batch['label'].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
        model.eval()
        collected = []
        with torch.no_grad():
            loader = DataLoader(_MemmapRGB(val, cache_root), batch_size=256, shuffle=False, num_workers=2)
            for batch in loader:
                images = _gpu_preprocess(batch['image'].to(device), 'tcga_crc_msi', train=False, generator=generator)
                output = model(images)
                logits = output['logits'] if isinstance(output, dict) else output
                collected.append((batch, logits.float().cpu().numpy()))
        val_logits = np.concatenate([item[1] for item in collected], 0) if collected else np.zeros((0, 2))
        val_rows = []
        for batch, _ in collected:
            for i in range(len(batch['graph_id'])):
                val_rows.append({'patient_id': batch['patient_id'][i], 'label_id': int(batch['label'][i])})
        metric = patient_metric(val_rows, val_logits) if val_rows else float('nan')
        history.append({'epoch': epoch, 'train_loss': running / max(1, len(fit)), 'val_auroc': metric})
        if np.isfinite(metric) and metric > best + 1e-12:
            best, stale = (metric, 0)
            save_model(destination, {'model': model.state_dict(), 'epoch': epoch, 'metric': metric, 'arm': 'B'})
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    predictions = []
    with torch.no_grad():
        loader = DataLoader(_MemmapRGB(val, cache_root), batch_size=256, shuffle=False, num_workers=2)
        grouped = defaultdict(list)
        labels = {}
        for batch in loader:
            images = _gpu_preprocess(batch['image'].to(device), 'tcga_crc_msi', train=False, generator=generator)
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            logit_np = logits.float().cpu().numpy()
            for pid, label, logit in zip(batch['patient_id'], batch['label'].numpy(), logit_np):
                grouped[str(pid)].append(logit)
                labels[str(pid)] = int(label)
        for pid, items in grouped.items():
            score = hard_vote_fraction(np.stack(items, 0))
            predictions.append({'sample_id': pid, 'label_id': labels[pid], 'probability': [1 - score, score], 'arm': 'B'})
    return write_pass(destination, {'dataset': 'tcga_crc_msi', 'arm': 'B', 'b_name': 'B', 'metric': 'patient_hard_vote_fraction', 'best': best, 'epochs': len(history), 'history': history, 'selected_epoch': int(saved.get('epoch', 0))}, predictions)

class _RoiTiles(Dataset):

    def __init__(self, bags: list[dict[str, Any]]):
        self.bags = bags

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int):
        bag = self.bags[index]
        images = []
        from PIL import Image
        for path in bag['paths'][:32]:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert('RGB').resize((224, 224)), np.uint8))
        if not images:
            images.append(np.zeros((224, 224, 3), np.uint8))
        stacked = np.stack(images, 0)
        return {'image': torch.from_numpy(stacked), 'n': len(images), 'label': int(bag['label_id']), 'sample_id': bag['sample_id']}

def _collate_roi(batch):
    max_n = max((item['n'] for item in batch))
    images = torch.zeros(len(batch), max_n, 224, 224, 3, dtype=torch.uint8)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    labels, ids = ([], [])
    for i, item in enumerate(batch):
        n = item['n']
        images[i, :n] = item['image']
        mask[i, :n] = True
        labels.append(item['label'])
        ids.append(item['sample_id'])
    return {'image': images, 'mask': mask, 'label': torch.tensor(labels), 'sample_id': ids}

def _bracs_bags(ids: list[str] | set[str]) -> list[dict[str, Any]]:
    from .paths import split_root
    from celllift.runtime import json
    wanted = set(map(str, ids))
    bags = []
    for name in ('train_rows.json', 'val_rows.json', 'test_rows.json'):
        path = split_root() / 'bracs' / name
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding='utf-8')):
            if str(row['sample_id']) not in wanted:
                continue
            paths = [p for p in row.get('rgb_paths') or [] if p]
            if not paths:
                continue
            bags.append({'paths': paths[:32], 'label_id': int(row['label_id']), 'sample_id': str(row['sample_id'])})
    return bags

def _train_bracs(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    fit = _bracs_bags(job['official_fit'])
    val = _bracs_bags(job['official_val'])
    if not fit:
        raise RuntimeError('BRACS B_rgb empty FIT RGB bags')
    from torchvision.models import ResNet18_Weights
    from celllift.pretrained import resnet18
    seed_all(int(job['seed']))
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = nn.Linear(feat_dim, 7)
    model = nn.ModuleDict({'backbone': backbone, 'head': head}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    destination = job_dir(job)
    best, stale, history = (-1000000000.0, 0, [])
    max_epochs, patience = (int(job.get('max_epochs', 60)), int(job.get('patience', 10)))

    def forward(batch):
        images = batch['image'].to(device).float() / 255.0
        b, n, h, w, c = images.shape
        flat = images.permute(0, 1, 4, 2, 3).reshape(b * n, 3, h, w)
        feat = model['backbone'](flat).reshape(b, n, -1)
        mask = batch['mask'].to(device).unsqueeze(-1)
        pooled = (feat * mask).sum(1) / mask.sum(1).clamp_min(1)
        return model['head'](pooled)
    for epoch in range(max_epochs):
        model.train()
        loader = DataLoader(_RoiTiles(fit), batch_size=2, shuffle=True, collate_fn=_collate_roi, num_workers=0)
        running = 0.0
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward(batch)
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch['label'])
            count += len(batch['label'])
        model.eval()
        pred, labels = ([], [])
        with torch.no_grad():
            loader = DataLoader(_RoiTiles(val), batch_size=2, shuffle=False, collate_fn=_collate_roi, num_workers=0)
            for batch in loader:
                logits = forward(batch)
                pred.extend(logits.argmax(1).cpu().tolist())
                labels.extend((int(v) for v in batch['label'].numpy()))
        metric = macro_f1(np.asarray(labels), np.asarray(pred), 7)
        w_f1 = weighted_f1(np.asarray(labels), np.asarray(pred), 7)
        history.append({'epoch': epoch, 'train_loss': running / max(1, count), 'val_macro_f1': metric, 'val_weighted_f1': w_f1})
        if np.isfinite(metric) and metric > best + 1e-12:
            best, stale = (metric, 0)
            save_model(destination, {'model': model.state_dict(), 'epoch': epoch, 'metric': metric, 'arm': 'B_rgb'})
        else:
            stale += 1
            if stale >= patience:
                break
    saved = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    predictions = []
    with torch.no_grad():
        loader = DataLoader(_RoiTiles(val), batch_size=2, shuffle=False, collate_fn=_collate_roi, num_workers=0)
        for batch in loader:
            logits = forward(batch)
            prob = torch.softmax(logits, 1).cpu().numpy()
            for sample_id, label, p in zip(batch['sample_id'], batch['label'].numpy(), prob):
                predictions.append({'sample_id': sample_id, 'label_id': int(label), 'probability': p.tolist(), 'arm': 'B_rgb'})
    return write_pass(destination, {'dataset': 'bracs', 'arm': 'B_rgb', 'b_name': 'B_rgb', 'b_is_hact_net': False, 'metric': 'roi_macro_f1', 'best': best, 'epochs': len(history), 'history': history, 'selected_epoch': int(saved.get('epoch', 0))}, predictions)

def train_baseline(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    destination = job_dir(job)
    existing = reused(destination)
    if existing:
        return existing
    if job['dataset'] == 'sicapv2':
        return _train_sicap(job, device)
    if job['dataset'] == 'tcga_crc_msi':
        return _train_crc(job, device)
    return _train_bracs(job, device)
