from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
import random
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch import nn
from celllift.geometry_baselines.io_utils import atomic_json, atomic_parquet, atomic_torch_save, runtime_identity, sha256
from .data import Sample, load_samples
from .models import FeatureFusionModel
from .protocol import ARMS, PROTOCOL_ID

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _batches(rows: Sequence[Sample], size: int, seed: int, shuffle: bool) -> list[list[Sample]]:
    indices = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [[rows[int(i)] for i in indices[start:start + size]] for start in range(0, len(rows), size)]

def _collate(rows: Sequence[Sample], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return ({'rgb_features': torch.as_tensor(np.stack([row.rgb_features for row in rows]), device=device), 'nucleus_summary': torch.as_tensor(np.stack([row.nucleus_summary for row in rows]), device=device), 'cell_summary': torch.as_tensor(np.stack([row.cell_summary for row in rows]), device=device)}, torch.tensor([row.label_id for row in rows], dtype=torch.long, device=device))

@torch.no_grad()
def predict(model: FeatureFusionModel, rows: Sequence[Sample], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    model.eval()
    output = []
    for part in _batches(rows, batch_size, 0, False):
        inputs, labels = _collate(part, device)
        result = model(**inputs)
        logits = result['logits']
        probability = torch.softmax(logits, -1) if logits.ndim == 2 else torch.sigmoid(logits)
        geometry_norm = torch.linalg.vector_norm(result['geometry_embedding'], dim=-1)
        for index, row in enumerate(part):
            output.append({'graph_id': row.graph_id, 'patient_id': row.patient_id, 'label_id': int(labels[index]), 'fold': row.fold, 'paper_logits': row.paper_logits.tolist(), 'final_logits': logits[index].float().cpu().numpy().tolist(), 'probabilities': probability[index].float().cpu().numpy().tolist(), 'geometry_embedding_norm': float(geometry_norm[index].float().cpu())})
    return output

def metric(dataset: str, rows: Sequence[Mapping[str, Any]], *, logits_key: str='final_logits') -> float:
    if dataset == 'sicapv2':
        from sklearn.metrics import cohen_kappa_score
        labels = [int(row['label_id']) for row in rows]
        prediction = [int(np.argmax(row[logits_key])) for row in rows]
        return float(cohen_kappa_score(labels, prediction, weights='quadratic'))
    from sklearn.metrics import roc_auc_score
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row['fold']), str(row['patient_id'])].append(row)
    fold_values: dict[int, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for (fold, _), values in groups.items():
        if len(values) < 10:
            continue
        fold_values[fold][0].append(int(values[0]['label_id']))
        if logits_key == 'paper_logits':
            fold_values[fold][1].append(float(np.mean([int(np.argmax(row[logits_key])) == 1 for row in values])))
        else:
            fold_values[fold][1].append(float(np.mean([float(row[logits_key]) >= 0 for row in values])))
    return float(np.mean([roc_auc_score(labels, scores) for labels, scores in fold_values.values()]))

def _crc_epoch(rows: Sequence[Sample], seed: int) -> list[Sample]:
    by_class = {label: [row for row in rows if row.label_id == label] for label in (0, 1)}
    take = min(len(by_class[0]), len(by_class[1]))
    rng = np.random.default_rng(seed)
    return [by_class[label][int(i)] for label in (0, 1) for i in rng.choice(len(by_class[label]), take, replace=False)]

def _crc_weights(rows: Sequence[Sample], device: torch.device) -> torch.Tensor:
    patient_count = Counter((row.patient_id for row in rows))
    class_patients: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        class_patients[row.label_id].add(row.patient_id)
    return torch.tensor([1.0 / (patient_count[row.patient_id] * len(class_patients[row.label_id])) for row in rows], device=device)

def train_job(cfg: Mapping[str, Any], *, fold: int, seed: int, arm_id: str, device: str) -> dict[str, Any]:
    output = Path(cfg['paths']['result_root']) / 'screen' / arm_id / f'seed_{seed}' / f'fold_{fold}'
    job_path = output / 'job.json'
    if job_path.is_file():
        from celllift.runtime import json
        old = json.loads(job_path.read_text(encoding='utf-8'))
        if old.get('status') == 'PASS' and old.get('protocol_id') == PROTOCOL_ID:
            return {'status': 'REUSED', **old}
        raise RuntimeError(f'incompatible feature_fusion output: {job_path}')
    if arm_id != 'C0':
        c0 = Path(cfg['paths']['result_root']) / 'screen' / 'C0' / f'seed_{seed}' / f'fold_{fold}' / 'best.pt'
        if not c0.is_file():
            raise RuntimeError('C0 must complete before geometry arms')
    train, validation = load_samples(cfg, fold=fold, seed=seed, arm_id=arm_id)
    target = torch.device(device)
    seed_everything(seed + fold * 1009)
    model = FeatureFusionModel(cfg['dataset'], geometry_enabled=ARMS[arm_id].geometry_id is not None, dropout=float(cfg['training']['dropout'])).to(target)
    if arm_id != 'C0':
        saved_c0 = torch.load(c0, map_location=target, weights_only=False)
        model.load_state_dict(saved_c0['model_state_dict'])
        optimizer = torch.optim.AdamW([{'params': list(model.rgb_projection.parameters()) + list(model.fused_norm.parameters()) + list(model.head.parameters()), 'lr': float(cfg['training']['rgb_learning_rate'])}, {'params': list(model.geometry.parameters()) + list(model.geometry_projection.parameters()), 'lr': float(cfg['training']['geometry_learning_rate'])}], weight_decay=float(cfg['training']['weight_decay']))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['head_learning_rate']), weight_decay=float(cfg['training']['weight_decay']))
    if cfg['dataset'] == 'sicapv2':
        counts = np.bincount([row.label_id for row in train], minlength=4)
        criterion: nn.Module = nn.CrossEntropyLoss(weight=torch.as_tensor(len(train) / (4 * counts), dtype=torch.float32, device=target), reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    batch_size = int(cfg['training']['batch_size'])
    checkpoint = output / 'best.pt'
    initial = predict(model, validation, target, batch_size)
    best = metric(cfg['dataset'], initial)
    stale = 0
    history = [{'epoch': -1, 'validation_metric': best, 'warm_start': 'C0' if arm_id != 'C0' else 'random'}]
    atomic_torch_save(checkpoint, {'protocol_id': PROTOCOL_ID, 'model_state_dict': model.state_dict(), 'epoch': -1, 'metric': best})
    scaler = torch.amp.GradScaler('cuda', enabled=target.type == 'cuda')
    for epoch in range(int(cfg['training']['max_epochs'])):
        epoch_rows = list(train) if cfg['dataset'] == 'sicapv2' else _crc_epoch(train, seed + epoch * 97 + fold * 1009)
        model.train()
        loss_sum = 0.0
        for part in _batches(epoch_rows, batch_size, seed + epoch, True):
            inputs, labels = _collate(part, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=target.type == 'cuda'):
                logits = model(**inputs)['logits']
                element = criterion(logits, labels if cfg['dataset'] == 'sicapv2' else labels.float())
                if cfg['dataset'] == 'sicapv2':
                    loss = element.mean()
                else:
                    weights = _crc_weights(part, target)
                    loss = (element * weights).sum() / weights.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
        predictions = predict(model, validation, target, batch_size)
        score = metric(cfg['dataset'], predictions)
        history.append({'epoch': epoch, 'loss_sum': loss_sum, 'validation_metric': score})
        if score > best + 1e-12:
            best = score
            stale = 0
            atomic_torch_save(checkpoint, {'protocol_id': PROTOCOL_ID, 'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': score})
        else:
            stale += 1
            if stale >= int(cfg['training']['patience']):
                break
    saved = torch.load(checkpoint, map_location=target, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    predictions = predict(model, validation, target, batch_size)
    prediction_path = output / 'validation_predictions.parquet'
    atomic_parquet(prediction_path, predictions)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': fold, 'seed': seed, 'arm_id': arm_id, 'geometry_id': ARMS[arm_id].geometry_id, 'metric': metric(cfg['dataset'], predictions), 'paper_rgb_metric': metric(cfg['dataset'], predictions, logits_key='paper_logits'), 'metric_name': 'QWK' if cfg['dataset'] == 'sicapv2' else 'patient_AUROC_hard_vote', 'train_rows': len(train), 'validation_rows': len(validation), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'predictions': str(prediction_path), 'predictions_sha256': sha256(prediction_path), 'history': history, 'runtime': runtime_identity(), 'fold_aligned_rgb_features': True, 'validation_only': True, 'official_test_touched': False}
    atomic_json(job_path, payload)
    return payload

def run_grid(cfg: Mapping[str, Any], *, fold: int, seed: int, arms: Sequence[str], device: str) -> dict[str, Any]:
    ordered = (['C0'] if 'C0' in arms else []) + [arm for arm in arms if arm != 'C0']
    return {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'results': [train_job(cfg, fold=fold, seed=seed, arm_id=arm, device=device) for arm in ordered]}
