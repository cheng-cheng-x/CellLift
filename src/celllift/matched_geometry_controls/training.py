from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Sequence
import numpy as np
from .feature_store import FoldModalStore, ModalRow
from .io_utils import atomic_json, atomic_parquet, atomic_torch
from .protocol import validate_job

def _pad_graphs(samples, device, meanpool: bool):
    import torch
    maximum = max((len(n) for n, _, _ in samples))
    batch = len(samples)
    nucleus = torch.zeros((batch, maximum, 41), device=device)
    cell = torch.zeros_like(nucleus)
    mask = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    counts = torch.empty(batch, device=device)
    for index, (n, c, count) in enumerate(samples):
        length = len(n)
        nucleus[index, :length] = torch.as_tensor(n, device=device)
        cell[index, :length] = torch.as_tensor(c, device=device)
        mask[index, :length] = True
        counts[index] = count
    payload = {'nucleus_tokens': nucleus, 'nucleus_mask': mask, 'cell_tokens': cell, 'cell_mask': mask}
    if meanpool:
        payload.update(nucleus_count=counts, cell_count=counts)
    return payload

def _sample(store, row: ModalRow, arm: str, encoder: str):
    if hasattr(store, 'sample_tokens'):
        return store.sample_tokens(row.graph_id)
    mode = 'direct' if arm in {'D', 'DS'} else 'residual'
    n, c = store.tokens(row.graph_id, mode, arm in {'DS', 'RS'})
    return (n, c, len(n))

def _metric(dataset, labels, logits):
    if dataset == 'sicapv2':
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(labels, logits.argmax(1), weights='quadratic'))
    if dataset == 'bracs':
        from sklearn.metrics import f1_score
        return float(f1_score(labels, logits.argmax(1), average='macro'))
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, logits[:, 0]))

def train_expert(*, dataset: str, rows: Sequence[ModalRow], store: FoldModalStore, fold: int, arm: str, encoder: str, seed: int, destination: str | Path, device: str='cuda', max_epochs: int=100, patience: int=15, learning_rate: float=0.0001, weight_decay: float=0.0001) -> dict:
    import torch
    import torch.nn.functional as F
    from .models import GroupGeometryClassifier, PatchGeometryClassifier
    validate_job(dataset, arm, encoder, seed)
    if arm == 'B0':
        raise ValueError('B0 is frozen/reused and is never trained here')
    torch.manual_seed(seed)
    np.random.seed(seed)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    development = [row for row in rows if row.split == 'train']
    training = [row for row in development if row.fold != fold]
    heldout = [row for row in development if row.fold == fold]
    if not training or not heldout:
        raise RuntimeError('empty outer training or heldout partition')
    classes = 4 if dataset == 'sicapv2' else 7 if dataset == 'bracs' else 1
    grouped = dataset in {'tcga_crc_msi', 'bracs'}
    model = (GroupGeometryClassifier if grouped else PatchGeometryClassifier)(encoder, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best = None
    best_metric = -float('inf')
    best_epoch = 0
    stale = 0
    history = []
    group_field = 'patient_id' if dataset == 'tcga_crc_msi' else 'roi_id'

    def groups(values):
        output = defaultdict(list)
        for row in values:
            output[getattr(row, group_field)].append(row)
        if dataset == 'tcga_crc_msi':
            output = defaultdict(list, {key: value for key, value in output.items() if len(value) >= 10})
        return output
    train_groups, held_groups = (groups(training), groups(heldout))
    if grouped and (not train_groups or not held_groups):
        raise RuntimeError('empty grouped outer training or heldout partition')

    def forward_rows(values):
        samples = [_sample(store, row, arm, encoder) for row in values]
        if not grouped:
            return model(**_pad_graphs(samples, device, encoder == 'meanpool'))
        tile_payload = _pad_graphs(samples, device, encoder == 'meanpool')
        shaped = {key: value[None] for key, value in tile_payload.items()}
        tile_mask = torch.ones((1, len(values)), dtype=torch.bool, device=device)
        return model(tile_mask=tile_mask, **shaped)[0]

    def evaluate():
        model.eval()
        logits = []
        labels = []
        ids = []
        with torch.inference_mode():
            if grouped:
                for key, values in held_groups.items():
                    logits.append(forward_rows(values).cpu().numpy()[0])
                    labels.append(values[0].label_id)
                    ids.append(key)
            else:
                for start in range(0, len(heldout), 128):
                    values = heldout[start:start + 128]
                    logits.extend(forward_rows(values).cpu().numpy())
                    labels.extend((row.label_id for row in values))
                    ids.extend((row.graph_id for row in values))
        array = np.asarray(logits)
        return (_metric(dataset, np.asarray(labels), array), ids, np.asarray(labels), array)
    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        losses = []
        if grouped:
            keys = list(train_groups)
            rng.shuffle(keys)
            iterator = (train_groups[key] for key in keys)
        else:
            order = np.arange(len(training))
            rng.shuffle(order)
            iterator = ([training[index] for index in order[start:start + 64]] for start in range(0, len(order), 64))
        for values in iterator:
            logits = forward_rows(values)
            labels = torch.as_tensor([values[0].label_id] if grouped else [row.label_id for row in values], device=device)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], labels.float()) if dataset == 'tcga_crc_msi' else F.cross_entropy(logits, labels.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metric, _, _, _ = evaluate()
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'metric': metric})
        if np.isfinite(metric) and metric > best_metric + 1e-07:
            best_metric = metric
            best_epoch = epoch + 1
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is None:
        raise RuntimeError('expert training produced no finite heldout metric')
    model.load_state_dict(best)
    metric, ids, labels, logits = evaluate()
    atomic_torch(destination / 'best.pt', {'model': best, 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed, 'best_epoch': best_epoch})
    rows_out = [{'sample_id': str(sample_id), 'label_id': int(label), 'logits': value.tolist(), 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed} for sample_id, label, value in zip(ids, labels, logits)]
    atomic_parquet(destination / 'predictions.parquet', rows_out)
    manifest = {'status': 'PASS', 'dataset': dataset, 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed, 'heldout_metric': metric, 'best_epoch': best_epoch, 'epochs': len(history), 'history': history, 'predictions': str(destination / 'predictions.parquet')}
    atomic_json(destination / 'manifest.json', manifest)
    return manifest
