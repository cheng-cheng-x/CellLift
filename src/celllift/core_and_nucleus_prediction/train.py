from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.pretrained import resnet18 as _packaged_resnet18
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Callable
import numpy as np
from celllift.runtime import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from .constants import RESULT_ROOT, SEED
from .data import ArvanitiWindowDataset, LizardNucleusDataset, LizardTileDataset, TileGroupedSampler, lizard_class_weights, pad_collate_windows
from .models import DeepSetsWindow, FeatureFusion, ImageNodeExpert, MobileNetV1Half, NucleusMLP, RelationExpert, SpatialFieldNet, WindowFusion

def _out(dataset: str, route: str, arm: str) -> Path:
    path = RESULT_ROOT / dataset / route / arm / 'seed42'
    path.mkdir(parents=True, exist_ok=True)
    return path

def _save(path: Path, model: torch.nn.Module, metrics: dict[str, Any]) -> None:
    torch.save(model.state_dict(), path / 'model.pt')
    (path / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')

def _dump_lizard_nucleus(model_fn, dataset, path: Path, name: str) -> None:
    ids, nids, groups, logits, labels = ([], [], [], [], [])
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=lambda items: {'token': torch.from_numpy(np.stack([item.get('token', item.get('dino', np.zeros(384, np.float32))) for item in items])).float(), 'dino': torch.from_numpy(np.stack([item['dino'] for item in items])).float(), 'crop': torch.from_numpy(np.stack([item['crop'] if item.get('crop') is not None else np.zeros((3, 224, 224), np.float32) for item in items])).float(), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long), 'graph_id': [item['graph_id'] for item in items], 'nucleus_id': np.asarray([item['nucleus_id'] for item in items], np.int64), 'group_id': [item.get('group_id') for item in items]})
    for batch in loader:
        out, target = model_fn(batch)
        ids.extend(batch['graph_id'])
        nids.append(batch['nucleus_id'])
        groups.extend(batch['group_id'])
        logits.append(out.detach().cpu().numpy())
        labels.append(target.detach().cpu().numpy())
    if not logits:
        return
    _save_logits(path, name, ids, np.concatenate(logits), np.concatenate(labels), nucleus_id=np.concatenate(nids), group_id=np.asarray(groups))

def _save_logits(path: Path, name: str, ids: list[str], logits: np.ndarray, labels: np.ndarray, **extra) -> None:
    payload = {'graph_id': np.asarray(ids), 'logits': np.asarray(logits, np.float32), 'label': np.asarray(labels, np.int64)}
    payload.update({key: np.asarray(value) for key, value in extra.items()})
    np.savez_compressed(path / name, **payload)

def _loop_ce(model_fn: Callable, loader, optimizer, device, weights=None):
    total = 0.0
    n = 0
    for batch in loader:
        logits, target = model_fn(batch)
        valid = target >= 0
        if not bool(valid.any()):
            continue
        if not torch.isfinite(logits).all():
            optimizer.zero_grad(set_to_none=True)
            continue
        loss = torch.nn.functional.cross_entropy(logits[valid], target[valid], weight=weights)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        total += float(loss.detach()) * int(valid.sum())
        n += int(valid.sum())
    return total / max(n, 1)

@torch.no_grad()
def _eval_acc(model_fn: Callable, loader, device):
    correct = 0
    n = 0
    ys, ps = ([], [])
    for batch in loader:
        logits, target = model_fn(batch)
        valid = target >= 0
        if not bool(valid.any()):
            continue
        pred = logits[valid].argmax(-1)
        correct += int((pred == target[valid]).sum())
        n += int(valid.sum())
        ys.append(target[valid].cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys) if ys else np.zeros(0, np.int64)
    p = np.concatenate(ps) if ps else np.zeros(0, np.int64)
    macro = float(f1_score(y, p, average='macro')) if len(y) else 0.0
    return (correct / max(n, 1), macro)

@torch.no_grad()
def _collect_window(model_fn: Callable, loader):
    ids, cores, logits, labels = ([], [], [], [])
    for batch in loader:
        out, target = model_fn(batch)
        ids.extend(batch['graph_id'])
        cores.extend(batch.get('core_id', batch['graph_id']))
        logits.append(out.cpu().numpy())
        labels.append(target.cpu().numpy())
    return (ids, cores, np.concatenate(logits), np.concatenate(labels))

def _window_core_qwk(model_fn: Callable, loader) -> float:
    from .evaluate import score_arvaniti_predictions
    from .predict import _softmax
    ids, cores, logits, labels = _collect_window(model_fn, loader)
    if not len(ids):
        return -1.0
    probs = _softmax(logits)
    rows = [{'graph_id': str(gid), 'core_id': str(cid), 'probs': prob, 'label': int(lab)} for gid, cid, prob, lab in zip(ids, cores, probs, labels)]
    scored = score_arvaniti_predictions(rows, 'val', bootstrap=False)
    value = scored.get('core_qwk_p1')
    if value is None or value != value:
        return -1.0
    return float(value)

class _CachedBatches:

    def __init__(self, loader, label: str='val_infer'):
        self.loader = loader
        self.label = label
        self._batches: list | None = None
        self.dataset = getattr(loader, 'dataset', None)

    def __iter__(self):
        if self._batches is None:
            print(f'caching {self.label} batches...', flush=True)
            self._batches = [batch for batch in self.loader]
            print(f'cached {self.label} n_batches={len(self._batches)}', flush=True)
        return iter(self._batches)

def _infer_loader(geom: str, batch_size: int, load_dino: bool, *, load_rgb: bool=False, load_geometry: bool=True) -> _CachedBatches:
    data = ArvanitiWindowDataset('val', geom=geom, supervised=False, load_dino=load_dino, load_rgb=load_rgb, cache_items=False, load_geometry=load_geometry)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_windows, num_workers=0)
    return _CachedBatches(loader, label=f'val_infer_{geom}')

def _fit(model, train_fn, val_fn, train_loader, val_loader, optimizer, device, path, *, max_epochs: int, patience: int, weights=None, metric: str='acc', select_fn: Callable | None=None):
    best = -1.0
    stale = 0
    history = []
    print(f'fit start path={path} n_train={len(train_loader.dataset)} n_val={len(val_loader.dataset)} max_epochs={max_epochs} metric={metric}', flush=True)
    for epoch in range(max_epochs):
        model.train()
        loss = _loop_ce(train_fn, train_loader, optimizer, device, weights)
        model.eval()
        acc, macro = _eval_acc(val_fn, val_loader, device)
        core_qwk = None
        if metric == 'core_qwk':
            if select_fn is None:
                raise RuntimeError('core_qwk selection needs select_fn')
            core_qwk = float(select_fn())
            score = core_qwk
        elif metric == 'macro':
            score = macro
        else:
            score = acc
        row = {'epoch': epoch, 'loss': loss, 'val_acc': acc, 'val_macro_f1': macro}
        if core_qwk is not None:
            row['val_core_qwk_p1'] = core_qwk
        history.append(row)
        torch.save(model.state_dict(), path / f'epoch_{epoch:04d}.pt')
        extra = f' val_core_qwk_p1={core_qwk:.4f}' if core_qwk is not None else ''
        print(f'epoch {epoch} loss={loss:.4f} val_acc={acc:.4f} val_macro={macro:.4f}{extra} best={best:.4f}', flush=True)
        if score > best:
            best = score
            stale = 0
            _save(path, model, {'best': best, 'best_acc': acc, 'best_macro_f1': macro, 'best_core_qwk_p1': core_qwk, 'select_metric': metric, 'history': history})
        else:
            stale += 1
            if stale >= patience:
                break
    if (path / 'model.pt').is_file():
        model.load_state_dict(torch.load(path / 'model.pt', map_location=device))
    (path / 'metrics.json').write_text(json.dumps({'best': best, 'epochs': len(history), 'select_metric': metric, 'history': history}, indent=2), encoding='utf-8')
    return {'best': best, 'epochs': len(history), 'history': history}

def train_arvaniti_geom(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = arm.lower()
    fit = ArvanitiWindowDataset('fit', geom=geom, load_dino=False)
    val = ArvanitiWindowDataset('val', geom=geom, load_dino=False)
    model = DeepSetsWindow({'g2': 38, 'g3': 9, 'gr': 9, 'g23': 47, 'g2r': 47}[geom]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=32, shuffle=True, collate_fn=pad_collate_windows, num_workers=0), DataLoader(val, batch_size=32, shuffle=False, collate_fn=pad_collate_windows, num_workers=0))

    def fn(batch):
        return (model(batch['tokens'].to(device), batch['mask'].to(device)), batch['label'].to(device))
    path = _out('arvaniti', 'geom', arm)
    infer = _infer_loader(geom, 32, False, load_rgb=False)

    def select():
        return _window_core_qwk(fn, infer)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=60, patience=10, metric='core_qwk', select_fn=select)
    return {'arm': arm, **metrics}

def train_arvaniti_baseline(device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fit = ArvanitiWindowDataset('fit', geom='g2', augment=True, load_dino=False, load_geometry=False)
    val = ArvanitiWindowDataset('val', geom='g2', load_dino=False, load_geometry=False)
    model = MobileNetV1Half().to(device)
    from .official_fcn import maybe_load_imagenet
    initialization = maybe_load_imagenet(model)
    path = _out('arvaniti', 'baseline', 'B_paper')
    (path / 'initialization.json').write_text(json.dumps(initialization, indent=2), encoding='utf-8')
    print(f'B_paper initialization: {initialization}', flush=True)
    loaders = (DataLoader(fit, batch_size=32, shuffle=True, collate_fn=pad_collate_windows, num_workers=0), DataLoader(val, batch_size=32, shuffle=False, collate_fn=pad_collate_windows, num_workers=0))
    if not fit.rows or 'rgb' not in next(iter(loaders[0])):
        path = _out('arvaniti', 'baseline', 'B_paper')
        _save(path, model, {'status': 'NO_RGB'})
        return {'arm': 'B_paper', 'status': 'NO_RGB'}

    def fn(batch):
        return (model(batch['rgb'].to(device)), batch['label'].to(device))
    for param in model.features.parameters():
        param.requires_grad = False
    opt = torch.optim.Adam(model.head.parameters(), lr=0.001)
    for epoch in range(5):
        model.train()
        model.features.eval()
        loss = _loop_ce(fn, loaders[0], opt, device)
        print(f'warmup epoch {epoch} loss={loss:.6f}', flush=True)
    for param in model.parameters():
        param.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    path = _out('arvaniti', 'baseline', 'B_paper')
    infer = _infer_loader('g2', 32, False, load_rgb=True, load_geometry=False)

    def select():
        return _window_core_qwk(fn, infer)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=500, patience=15, metric='core_qwk', select_fn=select)
    return {'arm': 'B_paper', **metrics}

def train_arvaniti_h(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = 'g2' if arm == 'H2' else 'g23'
    fit = ArvanitiWindowDataset('fit', geom=geom, load_dino=True, cache_items=True)
    val = ArvanitiWindowDataset('val', geom=geom, load_dino=True)
    image = MobileNetV1Half().to(device)
    ckpt = _out('arvaniti', 'baseline', 'B_paper') / 'model.pt'
    if not ckpt.is_file():
        raise SystemExit(f'H requires finished B_paper at {ckpt}')
    image.load_state_dict(torch.load(ckpt, map_location=device))
    for param in image.parameters():
        param.requires_grad = False
    image.eval()
    model = WindowFusion(512 + 384, 38 if arm == 'H2' else 47).to(device)
    opt = torch.optim.AdamW([{'params': model.image.parameters(), 'lr': 2e-05}, {'params': model.sets.parameters(), 'lr': 0.0001}, {'params': model.head.parameters(), 'lr': 2e-05}], weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=16, shuffle=True, collate_fn=pad_collate_windows, num_workers=0), DataLoader(val, batch_size=16, shuffle=False, collate_fn=pad_collate_windows, num_workers=0))

    def fn(batch):
        rgb = batch['rgb'].to(device) if 'rgb' in batch else torch.zeros(len(batch['label']), 3, 224, 224, device=device)
        emb = torch.cat((image.embedding(rgb), batch['image_summary'].to(device)), -1)
        return (model(emb, batch['tokens'].to(device), batch['mask'].to(device)), batch['label'].to(device))
    path = _out('arvaniti', 'fusion', arm)
    infer = _infer_loader(geom, 16, True, load_rgb=True)

    def select():
        return _window_core_qwk(fn, infer)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=80, patience=12, metric='core_qwk', select_fn=select)
    return {'arm': arm, **metrics}

def train_arvaniti_a(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = 'g2' if arm == 'A2' else 'g23'
    fit = ArvanitiWindowDataset('fit', geom=geom, load_dino=True)
    val = ArvanitiWindowDataset('val', geom=geom, load_dino=True)
    node_dim = 38 if arm == 'A2' else 47
    model = ImageNodeExpert(node_dim, 3, classes=4, pool=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=8, shuffle=True, collate_fn=pad_collate_windows, num_workers=0), DataLoader(val, batch_size=8, shuffle=False, collate_fn=pad_collate_windows, num_workers=0))

    def fn(batch):
        logits = []
        for i in range(len(batch['label'])):
            n = int(batch['mask'][i].sum().item())
            tokens = batch['tokens'][i, :n].to(device)
            dino = batch['dino'][i, :n].to(device)
            src = torch.as_tensor(batch['src'][i], device=device, dtype=torch.long).view(-1)
            dst = torch.as_tensor(batch['dst'][i], device=device, dtype=torch.long).view(-1)
            edge = torch.as_tensor(batch['edge'][i], device=device).float()
            if edge.ndim == 1:
                edge = edge.view(-1, 3)
            if src.numel() and int(src.max()) >= n:
                keep = (src < n) & (dst < n)
                src, dst, edge = (src[keep], dst[keep], edge[keep] if len(edge) == len(keep) else edge)
            logits.append(model(dino if n else torch.zeros(0, 384, device=device), tokens if n else torch.zeros(0, node_dim, device=device), edge if edge.numel() else torch.zeros((0, 3), device=device), src, dst, batch['mask'][i, :n].to(device) if n else None))
        return (torch.stack(logits), batch['label'].to(device))
    path = _out('arvaniti', 'interact', arm)
    infer = _infer_loader(geom, 8, True, load_rgb=False)

    def select():
        return _window_core_qwk(fn, infer)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=80, patience=12, metric='core_qwk', select_fn=select)
    return {'arm': arm, **metrics}

def train_arvaniti_c(arm: str, device: str='cuda') -> dict[str, Any]:
    from .fields import arvaniti_field_batch
    torch.manual_seed(SEED)
    fit = ArvanitiWindowDataset('fit', geom='g23')
    val = ArvanitiWindowDataset('val', geom='g23')
    model = SpatialFieldNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=16, shuffle=True, collate_fn=pad_collate_windows, num_workers=0), DataLoader(val, batch_size=16, shuffle=False, collate_fn=pad_collate_windows, num_workers=0))

    def fn(batch):
        field, dino = arvaniti_field_batch(batch, arm, device)
        return (model(field, dino), batch['label'].to(device))
    path = _out('arvaniti', 'spatial', arm)
    infer = _infer_loader('g23', 16, False, load_rgb=False)

    def select():
        return _window_core_qwk(fn, infer)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=80, patience=12, metric='core_qwk', select_fn=select)
    return {'arm': arm, **metrics}

def train_lizard_mlp(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = arm.lower()
    fit = LizardNucleusDataset('fit', geom=geom)
    val = LizardNucleusDataset('val', geom=geom)
    model = NucleusMLP({'n2': 38, 'n3': 9, 'nr': 9, 'n23': 47, 'n2r': 47}[geom]).to(device)
    weights = lizard_class_weights('fit').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    def collate(items):
        return {'token': torch.from_numpy(np.stack([item['token'] for item in items])), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long), 'graph_id': [item['graph_id'] for item in items], 'nucleus_id': np.asarray([item['nucleus_id'] for item in items], np.int64), 'group_id': [item.get('group_id') for item in items]}
    loaders = (DataLoader(fit, batch_size=256, shuffle=True, collate_fn=collate, num_workers=0), DataLoader(val, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0))

    def fn(batch):
        return (model(batch['token'].to(device)), batch['label'].to(device))
    path = _out('lizard', 'geom', arm)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=60, patience=10, weights=weights, metric='macro')
    return {'arm': arm, **metrics}

def train_lizard_dino(device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    fit = LizardNucleusDataset('fit', geom='n2', load_dino=True)
    val = LizardNucleusDataset('val', geom='n2', load_dino=True)
    model = NucleusMLP(384, 6, 128).to(device)
    weights = lizard_class_weights('fit').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    def collate(items):
        return {'token': torch.from_numpy(np.stack([item['dino'] for item in items])), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long)}
    loaders = (DataLoader(fit, batch_size=256, shuffle=True, collate_fn=collate, num_workers=0), DataLoader(val, batch_size=256, shuffle=False, collate_fn=collate, num_workers=0))

    def fn(batch):
        return (model(batch['token'].to(device)), batch['label'].to(device))
    path = _out('lizard', 'baseline', 'B_dino')
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=60, patience=10, weights=weights, metric='macro')
    return {'arm': 'B_dino', **metrics}

def train_lizard_crop(device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    import torchvision
    fit = LizardNucleusDataset('fit', geom='n2', crops=True)
    val = LizardNucleusDataset('val', geom='n2', crops=True)
    net = _packaged_resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Linear(512, 6)
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith('layer4') or name.startswith('fc')
    model = net.to(device)
    weights = lizard_class_weights('fit').to(device)
    opt = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if n.startswith('layer4')], 'lr': 0.0001}, {'params': model.fc.parameters(), 'lr': 0.001}], weight_decay=0.0001)

    def collate(items):
        crops = [item['crop'] if item['crop'] is not None else np.zeros((3, 224, 224), np.float32) for item in items]
        return {'rgb': torch.from_numpy(np.stack(crops).astype(np.float32)), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long)}
    loaders = (DataLoader(fit, batch_size=128, sampler=TileGroupedSampler(fit.items, SEED), collate_fn=collate, num_workers=0), DataLoader(val, batch_size=128, shuffle=False, collate_fn=collate, num_workers=0))

    def fn(batch):
        return (model(batch['rgb'].to(device)), batch['label'].to(device))
    path = _out('lizard', 'baseline', 'B_crop')
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=60, patience=10, weights=weights, metric='macro')
    return {'arm': 'B_crop', **metrics}

def train_lizard_j(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = {'J2': 'j2', 'JS': 'js', 'JR': 'jr'}[arm]
    fit = LizardTileDataset('fit', geom=geom)
    val = LizardTileDataset('val', geom=geom)
    node_dim = 38 if arm == 'J2' else 47
    edge_dim = 7 if arm == 'JR' else 3
    model = RelationExpert(node_dim, edge_dim, classes=6).to(device)
    weights = lizard_class_weights('fit').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda items: items[0]), DataLoader(val, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda items: items[0]))

    def fn(batch):
        item = batch if isinstance(batch, dict) else batch
        tokens = torch.as_tensor(item['tokens'][0] if item['tokens'].ndim == 3 else item['tokens'], device=device)
        if tokens.ndim == 3:
            tokens = tokens[0]
        src = torch.as_tensor(item['src'][0] if torch.is_tensor(item['src']) and item['src'].ndim > 1 else item['src'], device=device, dtype=torch.long)
        dst = torch.as_tensor(item['dst'][0] if torch.is_tensor(item['dst']) and item['dst'].ndim > 1 else item['dst'], device=device, dtype=torch.long)
        edge = torch.as_tensor(item['edge'][0] if torch.is_tensor(item['edge']) and item['edge'].ndim > 2 else item['edge'], device=device)
        if arm == 'JR':
            edge3 = torch.as_tensor(item['edge3'][0] if torch.is_tensor(item['edge3']) and item['edge3'].ndim > 2 else item['edge3'], device=device)
            edge = torch.cat((edge, edge3), -1) if edge.numel() else torch.zeros((0, 7), device=device)
        label = torch.as_tensor(item['label'][0] if torch.is_tensor(item['label']) and item['label'].ndim > 1 else item['label'], device=device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        return (model(tokens.float(), edge.float(), src.view(-1), dst.view(-1)), label.view(-1))
    path = _out('lizard', 'relation', arm)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=60, patience=10, weights=weights, metric='macro')
    return {'arm': arm, **metrics}

def train_lizard_h(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = 'n2' if arm == 'H2' else 'n23'
    fit = LizardNucleusDataset('fit', geom=geom, crops=True)
    val = LizardNucleusDataset('val', geom=geom, crops=True)
    import torchvision
    crop = _packaged_resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    crop.fc = torch.nn.Identity()
    ckpt = _out('lizard', 'baseline', 'B_crop') / 'model.pt'
    if not ckpt.is_file():
        raise SystemExit(f'LH requires B_crop at {ckpt}')
    state = torch.load(ckpt, map_location='cpu')
    crop.load_state_dict({k: v for k, v in state.items() if not k.startswith('fc')}, strict=False)
    for param in crop.parameters():
        param.requires_grad = False
    crop = crop.to(device)
    crop.eval()
    model = FeatureFusion(512 + 384, 38 if arm == 'H2' else 47, 6).to(device)
    opt = torch.optim.AdamW([{'params': model.image.parameters(), 'lr': 2e-05}, {'params': model.geom.parameters(), 'lr': 0.0001}, {'params': model.head.parameters(), 'lr': 2e-05}], weight_decay=0.0001)
    weights = lizard_class_weights('fit').to(device)

    def collate(items):
        crops = [item['crop'] if item['crop'] is not None else np.zeros((3, 224, 224), np.float32) for item in items]
        return {'rgb': torch.from_numpy(np.stack(crops).astype(np.float32)), 'token': torch.from_numpy(np.stack([item['token'] for item in items])), 'dino': torch.from_numpy(np.stack([item['dino'] for item in items])), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long)}
    loaders = (DataLoader(fit, batch_size=64, sampler=TileGroupedSampler(fit.items, SEED), collate_fn=collate, num_workers=0), DataLoader(val, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0))

    def fn(batch):
        with torch.no_grad():
            image = torch.cat((crop(batch['rgb'].to(device)), batch['dino'].to(device)), -1)
        return (model(image, batch['token'].to(device)), batch['label'].to(device))
    name = 'LH2' if arm == 'H2' else 'LH23'
    path = _out('lizard', 'fusion', name)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=80, patience=12, weights=weights, metric='macro')
    return {'arm': name, **metrics}

def train_lizard_a(arm: str, device: str='cuda') -> dict[str, Any]:
    torch.manual_seed(SEED)
    geom = 'a2' if arm == 'A2' else 'a3'
    fit = LizardTileDataset('fit', geom=geom, load_dino=True)
    val = LizardTileDataset('val', geom=geom, load_dino=True)
    node_dim = 38 if arm == 'A2' else 47
    edge_dim = 3 if arm == 'A2' else 7
    model = ImageNodeExpert(node_dim, edge_dim, classes=6, pool=False).to(device)
    weights = lizard_class_weights('fit').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    loaders = (DataLoader(fit, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda items: items[0]), DataLoader(val, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda items: items[0]))

    def fn(batch):
        tokens = torch.as_tensor(batch['tokens'], device=device).float()
        if tokens.ndim == 3:
            tokens = tokens[0]
        dino = torch.as_tensor(batch['dino'], device=device).float()
        if dino.ndim == 3:
            dino = dino[0]
        src = torch.as_tensor(batch['src'], device=device, dtype=torch.long).view(-1)
        dst = torch.as_tensor(batch['dst'], device=device, dtype=torch.long).view(-1)
        edge = torch.as_tensor(batch['edge'], device=device).float()
        if edge.ndim == 3:
            edge = edge[0]
        if arm == 'A3':
            edge3 = torch.as_tensor(batch['edge3'], device=device).float()
            if edge3.ndim == 3:
                edge3 = edge3[0]
            edge = torch.cat((edge, edge3), -1) if edge.numel() else torch.zeros((0, 7), device=device)
        label = torch.as_tensor(batch['label'], device=device, dtype=torch.long).view(-1)
        return (model(dino, tokens, edge, src, dst, None), label)
    name = 'LA2' if arm == 'A2' else 'LA3'
    path = _out('lizard', 'interact', name)
    metrics = _fit(model, fn, fn, loaders[0], loaders[1], opt, device, path, max_epochs=80, patience=12, weights=weights, metric='macro')
    return {'arm': name, **metrics}
ARMS: dict[str, Callable] = {'B_paper': train_arvaniti_baseline, 'G2': lambda device: train_arvaniti_geom('G2', device), 'G3': lambda device: train_arvaniti_geom('G3', device), 'GR': lambda device: train_arvaniti_geom('GR', device), 'G23': lambda device: train_arvaniti_geom('G23', device), 'G2R': lambda device: train_arvaniti_geom('G2R', device), 'H2': lambda device: train_arvaniti_h('H2', device), 'H23': lambda device: train_arvaniti_h('H23', device), 'A2': lambda device: train_arvaniti_a('A2', device), 'AS': lambda device: train_arvaniti_a('AS', device), 'C2': lambda device: train_arvaniti_c('C2', device), 'C3': lambda device: train_arvaniti_c('C3', device), 'B_crop': train_lizard_crop, 'B_dino': train_lizard_dino, 'N2': lambda device: train_lizard_mlp('N2', device), 'N3': lambda device: train_lizard_mlp('N3', device), 'NR': lambda device: train_lizard_mlp('NR', device), 'N23': lambda device: train_lizard_mlp('N23', device), 'N2R': lambda device: train_lizard_mlp('N2R', device), 'J2': lambda device: train_lizard_j('J2', device), 'JS': lambda device: train_lizard_j('JS', device), 'JR': lambda device: train_lizard_j('JR', device), 'LH2': lambda device: train_lizard_h('H2', device), 'LH23': lambda device: train_lizard_h('H23', device), 'LA2': lambda device: train_lizard_a('A2', device), 'LA3': lambda device: train_lizard_a('A3', device)}

def train_arm(arm: str, device: str='cuda') -> dict[str, Any]:
    if arm not in ARMS:
        raise SystemExit(f'unknown arm {arm}; known {sorted(ARMS)}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return ARMS[arm](device)
