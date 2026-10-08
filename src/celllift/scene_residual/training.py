from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import math
import numpy as np
from .baseline import baseline_probability
from .data import Sample, SceneCache, bag_batches, build_batch, ncr_fill_value
from .dataset import DATASET_SPEC
from .io_utils import atomic_torch
from .models import SceneCorrectionNet
DEFAULT_WIDTH = 64
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_MAX_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_DELTA_L2 = 0.001
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_BAGS_PER_BATCH = 8
DEFAULT_TILES_PER_BATCH = 256
DEFAULT_TRAIN_TILES = 32
DEFAULT_EPOCH_SEED_STRIDE = 1009
PARTITION_SEED = 20260917
ARM_MODEL = {'G-Zperm': 'G'}

def classes_for(dataset: str) -> int:
    return int(DATASET_SPEC[dataset]['classes'])

def build_samples_for_cache(dataset: str, rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], available: set[str] | None=None, tile_baseline: Mapping[str, Any] | None=None) -> list[Sample]:

    def usable(graph_id: str) -> bool:
        return available is None or graph_id in available
    output: list[Sample] = []
    if dataset == 'sicapv2':
        for row in rows:
            if not usable(row['graph_id']):
                continue
            entry = baseline.get(row['graph_id'])
            if entry is None:
                continue
            probability = np.asarray(baseline_probability(dataset, entry), np.float32)
            output.append(Sample(row['graph_id'], [row['graph_id']], int(entry['label_id']), int(entry['fold']), str(entry['group_id']), probability))
        return output
    key = 'roi_id' if dataset == 'bracs' else 'patient_id'
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if usable(row['graph_id']):
            grouped[str(row[key])].append(row)
    minimum_tiles = int(DATASET_SPEC['tcga_crc_msi'].get('min_tiles', 10))
    for bag_id, members in sorted(grouped.items()):
        if dataset == 'tcga_crc_msi':
            entry = baseline.get(bag_id)
            if entry is None or len(members) < minimum_tiles:
                continue
            probability = np.asarray([baseline_probability(dataset, entry)], np.float32)
            tile_decisions = None
            if tile_baseline is not None:
                tile_decisions = np.asarray([(float(tile_baseline.get(str(member['graph_id']), {}).get('rgb_probability', 0.5)), 1.0 if float(tile_baseline.get(str(member['graph_id']), {}).get('rgb_probability', 0.5)) >= 0.5 else 0.0) for member in members], np.float32)
        else:
            entry = baseline.get(bag_id)
            if entry is None:
                continue
            probability = np.asarray(baseline_probability(dataset, entry), np.float32)
            tile_decisions = None
        output.append(Sample(bag_id, [str(member['graph_id']) for member in members], int(entry['label_id']), int(entry['fold']), str(entry['group_id']), probability, tile_decisions))
    return output

def fold_split(samples: Sequence[Sample], fold: int) -> tuple[list[Sample], list[Sample]]:
    training = [sample for sample in samples if sample.fold != fold]
    heldout = [sample for sample in samples if sample.fold == fold]
    if not training or not heldout:
        raise RuntimeError(f'empty train/heldout partition for fold {fold}')
    return (training, heldout)

def partition(training: Sequence[Sample], seed: int=PARTITION_SEED) -> tuple[list[Sample], list[Sample]]:
    unique = sorted({sample.group_id for sample in training})
    if len(unique) < 2:
        return (list(training), list(training))
    generator = np.random.default_rng(seed)
    order = np.asarray(unique, object)
    generator.shuffle(order)
    count = min(max(1, int(round(len(unique) * DEFAULT_VALIDATION_FRACTION))), len(unique) - 1)
    validation_groups = set(order[:count].tolist())
    train = [sample for sample in training if sample.group_id not in validation_groups]
    validation = [sample for sample in training if sample.group_id in validation_groups]
    return (train, validation)

def subsample(sample: Sample, generator: np.random.Generator, maximum: int) -> Sample:
    if len(sample.graph_ids) <= maximum:
        return sample
    chosen = np.sort(generator.choice(len(sample.graph_ids), maximum, replace=False))
    return Sample(sample.bag_id, [sample.graph_ids[int(index)] for index in chosen], sample.label_id, sample.fold, sample.group_id, sample.baseline, None if sample.tile_rgb is None else sample.tile_rgb[chosen])

def _score(classes: int, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(probabilities, float)[:, 0] if classes == 1 else np.asarray(probabilities, float)

def selection_score(classes: int, probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(probabilities, float), 1e-07, 1.0)
    labels = np.asarray(labels)
    if classes == 1:
        positive = probabilities[:, 0]
        return float(np.mean(labels * np.log(positive) + (1 - labels) * np.log(1 - positive)))
    return float(np.mean(np.log(probabilities[np.arange(len(labels)), labels])))

def metric(dataset: str, labels: np.ndarray, probabilities: np.ndarray, folds: np.ndarray | None=None) -> float:
    probabilities = np.asarray(probabilities, float)
    if not np.isfinite(probabilities).all():
        return float('nan')
    if dataset == 'sicapv2':
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(labels, probabilities.argmax(1), weights='quadratic'))
    if dataset == 'bracs':
        from sklearn.metrics import f1_score
        return float(f1_score(labels, probabilities.argmax(1), average='macro'))
    from sklearn.metrics import roc_auc_score
    if folds is None:
        return float(roc_auc_score(labels, probabilities))
    values = []
    for fold in sorted(set(map(int, folds))):
        selected = folds == fold
        if len(np.unique(labels[selected])) < 2:
            continue
        values.append(float(roc_auc_score(labels[selected], probabilities[selected])))
    return float(np.mean(values)) if values else float('nan')

class Trainer:

    def __init__(self, *, dataset: str, cache: SceneCache, arm: str, seed: int, device: str, width: int=DEFAULT_WIDTH, layers: int=DEFAULT_LAYERS, dropout: float=DEFAULT_DROPOUT, learning_rate: float=DEFAULT_LR, weight_decay: float=DEFAULT_WEIGHT_DECAY, max_epochs: int=DEFAULT_MAX_EPOCHS, patience: int=DEFAULT_PATIENCE, delta_l2: float=DEFAULT_DELTA_L2, bags_per_batch: int=DEFAULT_BAGS_PER_BATCH, tiles_per_batch: int=DEFAULT_TILES_PER_BATCH, train_tiles: int=DEFAULT_TRAIN_TILES, z_offset: Mapping[str, np.ndarray] | None=None):
        import torch
        self.torch = torch
        self.dataset = dataset
        self.cache = cache
        self.arm = arm
        self.seed = int(seed)
        self.device = device
        self.classes = classes_for(dataset)
        self.bag = dataset in {'bracs', 'tcga_crc_msi'}
        self.train_tiles = int(train_tiles)
        self.bags_per_batch = int(bags_per_batch)
        self.tiles_per_batch = int(tiles_per_batch)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if arm == 'B':
            raise ValueError('arm B is the frozen baseline and has no trainable network')
        self.model = SceneCorrectionNet(classes=self.classes, width=width, layers=layers, dropout=dropout, arm=ARM_MODEL.get(arm, arm), bag=self.bag, tile_rgb=self.dataset == 'tcga_crc_msi' and arm not in {'Recal', 'B'}, baseline_dim=1 if self.classes == 1 else self.classes).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.delta_l2 = float(delta_l2)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.ncr_fill = 0.0
        self.z_offset = z_offset

    def parameter_count(self) -> int:
        return int(sum((parameter.numel() for parameter in self.model.parameters())))

    def _batch(self, samples: Sequence[Sample]):
        return build_batch(self.cache, samples, device=self.device, arm=self.arm, ncr_fill=self.ncr_fill, z_offset=self.z_offset)

    def _loss(self, batch):
        torch = self.torch
        output = self.model(batch)
        logits = output['logits']
        if self.classes == 1:
            supervised = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.labels.float())
        else:
            supervised = torch.nn.functional.cross_entropy(logits, batch.labels)
        return (supervised + self.delta_l2 * output['delta'].square().mean(), output)

    def predict(self, samples: Sequence[Sample], *, residual: bool=False):
        torch = self.torch
        self.model.eval()
        if len({sample.bag_id for sample in samples}) != len(samples):
            raise RuntimeError('predict requires unique bag_id values')
        index = {sample.bag_id: position for position, sample in enumerate(samples)}
        width = 1 if self.classes == 1 else self.classes
        baseline = np.zeros((len(samples), width), np.float32)
        final = np.zeros((len(samples), width), np.float32)
        labels = np.zeros((len(samples),), np.int64)
        delta = np.zeros((len(samples), width), np.float32) if residual else None
        with torch.inference_mode():
            for part in bag_batches(samples, self.bags_per_batch, self.tiles_per_batch):
                batch = self._batch(part)
                output = self.model(batch)
                logits = output['logits']
                part_baseline = batch.baseline.float().cpu().numpy()
                part_delta = output['delta'].float().cpu().numpy()
                if self.classes == 1:
                    part_final = torch.sigmoid(logits).float().cpu().numpy()
                    part_delta = np.asarray(part_delta, np.float32).reshape(-1, 1)
                else:
                    part_final = torch.softmax(logits, -1).float().cpu().numpy()
                    part_delta = np.asarray(part_delta, np.float32).reshape(-1, width)
                part_labels = batch.labels.cpu().numpy()
                for offset, sample in enumerate(batch.samples):
                    position = index[sample.bag_id]
                    baseline[position] = np.asarray(part_baseline[offset], np.float32).reshape(width)
                    final[position] = np.asarray(part_final[offset], np.float32).reshape(width)
                    labels[position] = int(part_labels[offset])
                    if delta is not None:
                        delta[position] = part_delta[offset]
        if residual:
            return (baseline, final, labels, delta)
        return (baseline, final, labels)

    def fit(self, train: Sequence[Sample], validation: Sequence[Sample], *, destination: str | Path | None=None, max_steps: int | None=None) -> dict[str, Any]:
        import torch
        graph_ids = sorted({graph_id for sample in list(train) + list(validation) for graph_id in sample.graph_ids})
        self.ncr_fill = ncr_fill_value(self.cache, graph_ids)
        destination = Path(destination) if destination is not None else None
        checkpoint = destination / 'best.pt' if destination else None
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
        initial_baseline, initial_final, initial_labels = self.predict(validation)
        best = selection_score(self.classes, initial_final, initial_labels)
        history = [{'epoch': -1, 'loss': None, 'selection': best, 'task_metric': metric(self.dataset, initial_labels, _score(self.classes, initial_final), self._fold_vector(validation)), 'baseline_identity': True}]
        if checkpoint is not None:
            atomic_torch(checkpoint, {'model': self.model.state_dict(), 'epoch': -1, 'metric': best})
        stale = 0
        steps_done = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            generator = np.random.default_rng(self.seed + epoch * DEFAULT_EPOCH_SEED_STRIDE)
            epoch_samples = list(train)
            if self.dataset != 'sicapv2':
                epoch_samples = [subsample(sample, generator, self.train_tiles) for sample in epoch_samples]
            order = generator.permutation(len(epoch_samples))
            running, steps = (0.0, 0)
            for part in bag_batches([epoch_samples[int(i)] for i in order], self.bags_per_batch, self.tiles_per_batch):
                batch = self._batch(part)
                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self._loss(batch)
                loss.backward()
                self.optimizer.step()
                running += float(loss.detach())
                steps += 1
                steps_done += 1
                if max_steps is not None and steps_done >= max_steps:
                    break
            _, final_probability, labels = self.predict(validation)
            score = selection_score(self.classes, final_probability, labels)
            history.append({'epoch': epoch, 'loss': running / max(1, steps), 'selection': score, 'task_metric': metric(self.dataset, labels, _score(self.classes, final_probability), self._fold_vector(validation))})
            if math.isfinite(score) and score > best + 1e-12:
                best, stale = (score, 0)
                if checkpoint is not None:
                    atomic_torch(checkpoint, {'model': self.model.state_dict(), 'epoch': epoch, 'metric': score})
            else:
                stale += 1
                if stale >= self.patience:
                    break
            if max_steps is not None and steps_done >= max_steps:
                break
        if checkpoint is not None and checkpoint.is_file():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(saved['model'])
        return {'best_metric': best, 'epochs': len(history), 'history': history, 'ncr_fill': self.ncr_fill}

    @staticmethod
    def _fold_vector(samples: Sequence[Sample]) -> np.ndarray:
        return np.asarray([sample.fold for sample in samples], np.int64)
