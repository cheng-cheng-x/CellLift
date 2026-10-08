from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
import math
from celllift.runtime import ResourcePath as Path
import random
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch import nn
from .geometry_training import ARMS, GraphTokenSample, GeometryFoldTokenStore, audit_geometry_store, configure, preaggregate_meanpool
from .io_utils import atomic_json, atomic_parquet, atomic_torch_save, sha256
from .models import GeometryClassifier
from .protocol import PROTOCOL_ID, SEEDS

def _seed_job(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _batches(samples: Sequence[GraphTokenSample], budget: int, *, seed: int, shuffle: bool) -> list[list[GraphTokenSample]]:
    ordered = sorted(samples, key=lambda row: (len(row.nucleus_tokens), row.graph_id))
    output, current, maximum = ([], [], 0)
    for sample in ordered:
        next_max = max(maximum, len(sample.nucleus_tokens))
        if current and next_max * (len(current) + 1) > budget:
            output.append(current)
            current, maximum = ([], 0)
        current.append(sample)
        maximum = max(maximum, len(sample.nucleus_tokens))
    if current:
        output.append(current)
    if shuffle:
        np.random.default_rng(seed).shuffle(output)
    return output

def _collate(samples: Sequence[GraphTokenSample], device: torch.device, *, patient_tile_counts: Mapping[str, int] | None=None, class_patient_counts: Mapping[int, int] | None=None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    from .tokens import collate, GeometrySample
    batch = collate([GeometrySample(row.graph_id, row.nucleus_tokens, row.cell_tokens, row.metadata) for row in samples])
    inputs = {key: torch.as_tensor(value, dtype=torch.bool if key.endswith('mask') else torch.float32, device=device) for key, value in batch.items() if key != 'graph_ids'}
    labels = torch.tensor([int(row.metadata['label_id']) for row in samples], dtype=torch.float32, device=device)
    patients = [str(row.metadata['patient_id']) for row in samples]
    patient_counts = patient_tile_counts or Counter(patients)
    if class_patient_counts is None:
        by_class: dict[int, set[str]] = defaultdict(set)
        for patient, label in zip(patients, labels.detach().cpu().tolist()):
            by_class[int(label)].add(patient)
        class_patient_counts = {label: len(values) for label, values in by_class.items()}
    weights = torch.tensor([1.0 / (patient_counts[patient] * class_patient_counts[int(label)]) for patient, label in zip(patients, labels.detach().cpu().tolist())], dtype=torch.float32, device=device)
    return (inputs, labels, weights)

@torch.no_grad()
def _predict(model: GeometryClassifier, samples: Sequence[GraphTokenSample], device: torch.device, budget: int) -> list[dict[str, Any]]:
    model.eval()
    output = []
    for batch in _batches(samples, budget, seed=0, shuffle=False):
        inputs, labels, _ = _collate(batch, device)
        scores = torch.sigmoid(model(**inputs)['logits']).cpu().numpy()
        for sample, label, score in zip(batch, labels.cpu().numpy(), scores):
            output.append({'graph_id': sample.graph_id, 'patient_id': str(sample.metadata['patient_id']), 'label_id': int(label), 'score': float(score)})
    return output

def _patient_auc(rows: Sequence[Mapping[str, Any]]) -> float:
    from sklearn.metrics import roc_auc_score
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row['patient_id'])].append(row)
    labels, scores = ([], [])
    for values in groups.values():
        labels.append(int(values[0]['label_id']))
        scores.append(float(np.mean([float(row['score']) for row in values])))
    return float(roc_auc_score(labels, scores))

def _load_store(cfg: Mapping[str, Any], fold: int, seed: int) -> GeometryFoldTokenStore:
    audit_geometry_store(cfg, fold, seed=seed)
    data_root = Path(cfg['paths']['data_root'])
    graph_manifest = data_root / '00_manifest' / 'downstream_manifest.parquet'
    import pyarrow.parquet as pq
    rows = pq.read_table(graph_manifest, partitioning=None).to_pylist()
    eligible = [row for row in rows if str(row['split']).lower() == 'train']
    train_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) != fold]
    val_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) == fold]
    partition = {graph: 'train' for graph in train_ids} | {graph: 'validation' for graph in val_ids}
    configure(cfg)
    return GeometryFoldTokenStore.from_parquet(Path(cfg['paths']['set_encoding_data_root']) / '02_nucleus_tokens', Path(cfg['paths']['set_encoding_data_root']) / '03_cell_tokens', Path(cfg['paths']['set_encoding_data_root']) / '04_ncr_features', graph_manifest, training_graph_ids=train_ids, seed=seed, included_graph_ids=train_ids + val_ids, partition_by_graph=partition)

def train_tile_geometry(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', encoders: Sequence[str] | None=None, seeds: Sequence[int] | None=None, geometry_ids: Sequence[str] | None=None) -> dict[str, Any]:
    if cfg['dataset'] != 'tcga_crc_msi':
        raise RuntimeError('tile geometry route is registered only for CRC')
    target = torch.device(device)
    budget = int(cfg['geometry_training'].get('token_budget', 32000))
    completed = []
    for seed in tuple(seeds or SEEDS):
        store = _load_store(cfg, fold, int(seed))
        train_ids = [graph for graph in store.graph_ids if store.partition_by_graph[graph] == 'train']
        val_ids = [graph for graph in store.graph_ids if store.partition_by_graph[graph] == 'validation']
        for encoder in tuple(encoders or ('meanpool', 'deepsets')):
            for geometry_id in tuple(geometry_ids or ('G1', 'G1S', 'G2', 'G2S', 'G3', 'G3S', 'G4', 'G4S', 'G5', 'G5S')):
                arm = ARMS['O' + geometry_id[1:]]
                output = Path(cfg['paths']['result_root']) / 'tile_fusion' / 'experts' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{fold}'
                job_path = output / 'job.json'
                if job_path.is_file():
                    from celllift.runtime import json
                    previous = json.loads(job_path.read_text(encoding='utf-8'))
                    if previous.get('status') == 'PASS' and previous.get('protocol_id') == PROTOCOL_ID:
                        completed.append({'status': 'REUSED', 'path': str(job_path)})
                        continue
                    raise RuntimeError(f'non-PASS or incompatible existing tile job: {job_path}')
                train_samples = store.samples(arm, train_ids)
                validation_samples = store.samples(arm, val_ids)
                if encoder == 'meanpool':
                    train_samples = preaggregate_meanpool(train_samples)
                    validation_samples = preaggregate_meanpool(validation_samples)
                _seed_job(int(seed))
                model = GeometryClassifier('tcga_crc_msi', encoder, dropout=float(cfg['geometry_training']['dropout'])).to(target)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
                criterion = nn.BCEWithLogitsLoss(reduction='none')
                best, stale, history = (-math.inf, 0, [])
                checkpoint = output / 'best.pt'
                for epoch in range(int(cfg['geometry_training']['max_epochs'])):
                    by_class = {label: [row for row in train_samples if int(row.metadata['label_id']) == label] for label in (0, 1)}
                    take = min(map(len, by_class.values()))
                    rng = np.random.default_rng(int(seed) + epoch * 1009)
                    epoch_samples = []
                    for label in (0, 1):
                        chosen = rng.choice(len(by_class[label]), take, replace=False)
                        epoch_samples.extend((by_class[label][int(index)] for index in chosen))
                    epoch_patient_counts = Counter((str(row.metadata['patient_id']) for row in epoch_samples))
                    epoch_class_patients: dict[int, set[str]] = defaultdict(set)
                    for row in epoch_samples:
                        epoch_class_patients[int(row.metadata['label_id'])].add(str(row.metadata['patient_id']))
                    epoch_class_patient_counts = {label: len(patients) for label, patients in epoch_class_patients.items()}
                    model.train()
                    loss_sum = 0.0
                    for batch in _batches(epoch_samples, budget, seed=int(seed) + epoch, shuffle=True):
                        inputs, labels, patient_weight = _collate(batch, target, patient_tile_counts=epoch_patient_counts, class_patient_counts=epoch_class_patient_counts)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(**inputs)['logits']
                        loss = (criterion(logits, labels) * patient_weight).sum() / patient_weight.sum()
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(loss.detach())
                    predictions = _predict(model, validation_samples, target, budget)
                    metric = _patient_auc(predictions)
                    history.append({'epoch': epoch, 'loss_sum': loss_sum, 'validation_patient_auc': metric})
                    if metric > best + 1e-12:
                        best, stale = (metric, 0)
                        atomic_torch_save(checkpoint, {'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric})
                    else:
                        stale += 1
                        if stale >= int(cfg['geometry_training']['patience']):
                            break
                saved = torch.load(checkpoint, map_location=target, weights_only=False)
                model.load_state_dict(saved['model_state_dict'])
                predictions = _predict(model, validation_samples, target, budget)
                prediction_path = output / 'validation_tile_predictions.parquet'
                atomic_parquet(prediction_path, predictions)
                job = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'fold': fold, 'seed': int(seed), 'encoder': encoder, 'arm_id': arm.arm_id, 'validation_patient_auc': _patient_auc(predictions), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'predictions': str(prediction_path), 'predictions_sha256': sha256(prediction_path), 'deterministic_job_seed': int(seed), 'patient_normalized_loss': True, 'class_balanced_sampling': True, 'history': history}
                atomic_json(job_path, job)
                completed.append(job)
    return {'status': 'PASS', 'jobs': len(completed), 'completed': completed}
