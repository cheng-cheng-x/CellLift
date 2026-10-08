from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from celllift.runtime import ResourcePath as Path
import random
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from .io_utils import atomic_json, atomic_parquet, atomic_torch_save, runtime_identity, sha256
from .models import PaperFSConv, build_paper_crc_resnet18, crc_adam_groups
from .protocol import PROTOCOL_ID, require_test_gate

@dataclass(frozen=True)
class PaperRGBSpec:
    dataset: str
    batch_size: int
    max_epochs: int
    workers: int
    source_size: int = 512
    input_size: int = 224
    fp32: bool = True

    @classmethod
    def for_dataset(cls, dataset: str, workers: int=8) -> 'PaperRGBSpec':
        if dataset == 'sicapv2':
            return cls(dataset, 32, 200, workers)
        if dataset == 'tcga_crc_msi':
            return cls(dataset, 256, 100, workers)
        raise ValueError(dataset)

def seed_everything(seed: int) -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _stable_shard(graph_id: str, shards: int) -> int:
    return int.from_bytes(hashlib.sha256(graph_id.encode()).digest()[:8], 'little') % shards

def cache_paper_rgb(labels_path: str | Path, destination: str | Path, *, include_test: bool, result_root: str | Path, shards: int=16) -> dict[str, Any]:
    from PIL import Image
    labels_path, destination = (Path(labels_path), Path(destination))
    if include_test:
        require_test_gate(result_root)
    rows = read_rows(labels_path)
    target_split = 'test' if include_test else 'train'
    selected = [row for row in rows if str(row['official_split']).lower() == target_split]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[_stable_shard(str(row['graph_id']), shards)].append(row)
    index_rows: list[dict[str, Any]] = []
    shard_meta: list[dict[str, Any]] = []
    destination.mkdir(parents=True, exist_ok=True)
    for shard in range(shards):
        current = sorted(grouped.get(shard, []), key=lambda row: str(row['graph_id']))
        final = destination / f'rgb224_{shard:02d}.npy'
        temporary = destination / f'.rgb224_{shard:02d}.tmp.{os.getpid()}.npy'
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(current), 224, 224, 3))
        for offset, row in enumerate(current):
            path = Path(str(row['source_path']))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise RuntimeError(f'paper source must be 512x512: {path} is {image.size}')
                value = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), np.uint8)
            array[offset] = value
            index_row = {'graph_id': str(row['graph_id']), 'patch_id': str(row['patch_id']), 'patient_id': str(row['patient_id']), 'official_split': str(row['official_split']).lower(), 'validation_fold': None if include_test else int(row['validation_fold']), 'source_path': str(path), 'source_sha256': str(row['source_sha256']), 'shard': shard, 'offset': offset}
            if not include_test:
                index_row['label_id'] = int(row['label_id'])
            index_rows.append(index_row)
        array.flush()
        del array
        os.replace(temporary, final)
        shard_meta.append({'shard': shard, 'rows': len(current), 'path': str(final), 'sha256': sha256(final)})
    index_path = destination / 'index.parquet'
    atomic_parquet(index_path, index_rows)
    manifest = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'include_test': include_test, 'cache_phase': 'official_test_only' if include_test else 'validation_train_only', 'rows': len(index_rows), 'shards': shard_meta, 'index': str(index_path), 'index_sha256': sha256(index_path), 'source': str(labels_path), 'source_sha256': sha256(labels_path), 'resize': 'PIL bilinear 512x512 -> 224x224', 'storage': 'uint8 NHWC npy memmap; no JPEG recompression'}
    atomic_json(destination / 'manifest.json', manifest)
    return manifest

def write_paper_fidelity(dataset: str, result_root: str | Path) -> dict[str, Any]:
    result_root = Path(result_root)
    if dataset == 'sicapv2':
        model = PaperFSConv()
        payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'architecture': 'FSConv(3-32-128-512)+GMP+Linear(512,4)', 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'expected_parameters': PaperFSConv.expected_parameters, 'initialization': 'Xavier uniform weights; zero bias'}
    elif dataset == 'tcga_crc_msi':
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        from celllift.runtime import ROOT as PACKAGE_ROOT
        cache = PACKAGE_ROOT / 'weights/pretrained' / Path(weights.url).name
        if not cache.is_file():
            raise FileNotFoundError(f'trained ResNet18 weight cache is absent: {cache}')
        digest = sha256(cache)
        if digest != 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec':
            raise RuntimeError(f'ResNet18 IMAGENET1K_V1 SHA drift: {digest}')
        payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'weights_enum': 'ResNet18_Weights.IMAGENET1K_V1', 'weights_url': weights.url, 'weights_cache': str(cache), 'weights_sha256': digest, 'weights_bytes': cache.stat().st_size, 'trainable_scope': 'layer4.1 + Linear(512,2)'}
    else:
        raise ValueError(dataset)
    destination = result_root / 'paper_baseline' / 'fidelity.json'
    atomic_json(destination, payload)
    return {**payload, 'path': str(destination), 'sha256': sha256(destination)}

class RGBMemmapDataset(Dataset):

    def __init__(self, rows: Sequence[dict[str, Any]], cache_root: str | Path, *, include_label: bool=True) -> None:
        self.rows = list(rows)
        self.cache_root = Path(cache_root)
        self.include_label = bool(include_label)
        self._arrays: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard = int(row['shard'])
        if shard not in self._arrays:
            self._arrays[shard] = np.load(self.cache_root / f'rgb224_{shard:02d}.npy', mmap_mode='r')
        image = torch.from_numpy(np.array(self._arrays[shard][int(row['offset'])], copy=True)).permute(2, 0, 1)
        output = {'image': image}
        if self.include_label:
            output['label'] = int(row['label_id'])
        return output

class ShardBatchSampler(Sampler[list[int]]):

    def __init__(self, rows: Sequence[dict[str, Any]], batch_size: int, seed: int) -> None:
        self.batch_size, self.seed = (int(batch_size), int(seed))
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.groups[int(row['shard'])].append(index)

    def __len__(self) -> int:
        return sum((math.ceil(len(indices) / self.batch_size) for indices in self.groups.values()))

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches: list[list[int]] = []
        for shard in sorted(self.groups):
            indices = np.asarray(self.groups[shard], np.int64)
            rng.shuffle(indices)
            batches.extend((indices[start:start + self.batch_size].tolist() for start in range(0, len(indices), self.batch_size)))
        rng.shuffle(batches)
        yield from batches

def _loader(rows: Sequence[dict[str, Any]], cache_root: Path, spec: PaperRGBSpec, *, shuffle: bool, seed: int, include_labels: bool=True, persistent_workers: bool=False) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    workers = int(spec.workers)
    keep = bool(persistent_workers) and workers > 0
    common = dict(dataset=RGBMemmapDataset(rows, cache_root, include_label=include_labels), num_workers=workers, pin_memory=True, persistent_workers=keep, prefetch_factor=4 if workers > 0 else None)
    if shuffle:
        return DataLoader(batch_sampler=ShardBatchSampler(rows, spec.batch_size, seed), **common)
    return DataLoader(batch_size=spec.batch_size, shuffle=False, generator=generator, **common)

def _integer_translate_white(images: Tensor, shifts: Tensor) -> Tensor:
    batch, _, height, width = images.shape
    pad = 5
    canvas = images.new_ones(batch, images.shape[1], height + 2 * pad, width + 2 * pad)
    canvas[:, :, pad:pad + height, pad:pad + width] = images
    origin_y = pad - shifts[:, 0]
    origin_x = pad - shifts[:, 1]
    rows = torch.arange(height, device=images.device)
    cols = torch.arange(width, device=images.device)
    y = origin_y[:, None, None] + rows[None, :, None]
    x = origin_x[:, None, None] + cols[None, None, :]
    index = torch.arange(batch, device=images.device)[:, None, None]
    return canvas.permute(0, 2, 3, 1)[index, y, x].permute(0, 3, 1, 2)

def _gpu_preprocess(images: Tensor, dataset: str, *, train: bool, generator: torch.Generator) -> Tensor:
    images = images.cuda(non_blocking=True).float().div_(255.0)
    batch = len(images)
    if train and dataset == 'sicapv2':
        angles = (torch.rand(batch, device=images.device, generator=generator) * 2 - 1) * math.pi
        translations = (torch.rand(batch, 2, device=images.device, generator=generator) * 2 - 1) * 0.2
        theta = torch.zeros(batch, 2, 3, device=images.device)
        theta[:, 0, 0], theta[:, 0, 1] = (torch.cos(angles), -torch.sin(angles))
        theta[:, 1, 0], theta[:, 1, 1] = (torch.sin(angles), torch.cos(angles))
        theta[:, :, 2] = translations
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        transformed = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        valid = F.grid_sample(torch.ones_like(images[:, :1]), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        images = transformed + (1.0 - valid)
    elif train and dataset == 'tcga_crc_msi':
        flip_x = torch.rand(batch, device=images.device, generator=generator) < 0.5
        flip_y = torch.rand(batch, device=images.device, generator=generator) < 0.5
        images[flip_x] = torch.flip(images[flip_x], dims=(3,))
        images[flip_y] = torch.flip(images[flip_y], dims=(2,))
        shifts = torch.randint(-5, 6, (batch, 2), device=images.device, generator=generator)
        images = _integer_translate_white(images, shifts)
    if dataset == 'tcga_crc_msi':
        mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        images = (images - mean) / std
    return images

def _select_outer(rows: Sequence[dict[str, Any]], fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_train = [row for row in rows if str(row['official_split']).lower() == 'train']
    train = [row for row in official_train if int(row['validation_fold']) != fold]
    validation = [row for row in official_train if int(row['validation_fold']) == fold]
    if not train or not validation:
        raise RuntimeError('empty outer train or validation')
    return (train, validation)

def _balanced_crc_inner(rows: Sequence[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    by_class = {label: [row for row in rows if int(row['label_id']) == label] for label in (0, 1)}
    target = min(map(len, by_class.values()))
    balanced: list[dict[str, Any]] = []
    for label in (0, 1):
        indices = rng.choice(len(by_class[label]), target, replace=False)
        selected = [by_class[label][int(index)] for index in indices]
        rng.shuffle(selected)
        n_train = int(math.floor(0.85 * target))
        n_validation = int(math.floor(0.125 * target))
        for position, row in enumerate(selected):
            copy = dict(row)
            copy['_inner'] = 'train' if position < n_train else 'validation' if position < n_train + n_validation else 'unused'
            balanced.append(copy)
    rng.shuffle(balanced)
    return tuple(([row for row in balanced if row['_inner'] == role] for role in ('train', 'validation', 'unused')))

def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))

def _patient_auc(rows: Sequence[dict[str, Any]], scores: np.ndarray, *, minimum_tiles: int=10) -> float:
    from sklearn.metrics import roc_auc_score
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[str(row['patient_id'])].append((int(row['label_id']), float(score)))
    labels, patient_scores = ([], [])
    for values in groups.values():
        if len(values) < minimum_tiles:
            continue
        labels.append(values[0][0])
        patient_scores.append(float(np.mean([score >= 0.5 for _, score in values])))
    return float(roc_auc_score(labels, patient_scores))

@torch.no_grad()
def _predict(model: nn.Module, rows: Sequence[dict[str, Any]], cache_root: Path, spec: PaperRGBSpec, seed: int, *, collect_labels: bool=True) -> tuple[np.ndarray, np.ndarray | None]:
    loader = _loader(rows, cache_root, spec, shuffle=False, seed=seed, include_labels=collect_labels)
    logits, labels = ([], [])
    generator = torch.Generator(device='cuda').manual_seed(seed)
    model.eval()
    for batch in loader:
        images = _gpu_preprocess(batch['image'], spec.dataset, train=False, generator=generator)
        output = model(images)
        value = output['logits'] if isinstance(output, dict) else output
        logits.append(value.float().cpu().numpy())
        if collect_labels:
            labels.append(batch['label'].numpy())
    return (np.concatenate(logits), np.concatenate(labels) if collect_labels else None)

def train_paper_rgb(*, dataset: str, fold: int | None, seed: int, cache_root: str | Path, result_root: str | Path, workers: int=8, imagenet_weights: bool=True, official_test: bool=False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError('paper-RGB formal training requires CUDA')
    write_paper_fidelity(dataset, result_root)
    spec = PaperRGBSpec.for_dataset(dataset, workers)
    cache_root, result_root = (Path(cache_root), Path(result_root))
    if official_test:
        require_test_gate(result_root)
        if fold is not None:
            raise ValueError('official full-TRAIN paper model must use fold=None')
        output_root = result_root / 'official_test' / 'paper_baseline' / f'seed_{seed}'
    else:
        if fold is None:
            raise ValueError('validation paper model requires a fold')
        output_root = result_root / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}'
    completed_path = output_root / 'job.json'
    if completed_path.is_file():
        from celllift.runtime import json
        previous = json.loads(completed_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            expected = {'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test}
            if any((previous.get(key) != value for key, value in expected.items())):
                raise RuntimeError(f'paper-RGB completed job provenance mismatch: {completed_path}')
            if not Path(previous['checkpoint']).is_file() or sha256(previous['checkpoint']) != previous.get('checkpoint_sha256'):
                raise RuntimeError(f'paper-RGB completed checkpoint checksum mismatch: {completed_path}')
            if not Path(previous['predictions']).is_file() or sha256(previous['predictions']) != previous.get('predictions_sha256'):
                raise RuntimeError(f'paper-RGB completed prediction checksum mismatch: {completed_path}')
            return {**previous, 'reused': True}
    rows = read_rows(cache_root / 'index.parquet')
    prediction_cache_root = cache_root
    if official_test:
        if any((str(row['official_split']).lower() != 'train' for row in rows)):
            raise RuntimeError('official full-TRAIN model received a non-TRAIN row from the frozen TRAIN cache')
        train_rows = rows
        prediction_cache_root = cache_root / 'official_test'
        validation_rows = read_rows(prediction_cache_root / 'index.parquet')
        if any((str(row['official_split']).lower() != 'test' for row in validation_rows)):
            raise RuntimeError('official prediction cache contains a non-TEST row')
        if not train_rows or not validation_rows:
            raise RuntimeError('official full-TRAIN or TEST RGB cache is empty')
        fold_for_seed = 0
    else:
        train_rows, validation_rows = _select_outer(rows, int(fold))
        fold_for_seed = int(fold)
    seed_everything(seed)
    if dataset == 'sicapv2':
        model = PaperFSConv().cuda()
        counts = Counter((int(row['label_id']) for row in train_rows))
        weights = torch.tensor([len(train_rows) / (4 * counts[index]) for index in range(4)], device='cuda')
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0)
        inner_train = train_rows
    else:
        model, trainable_names = build_paper_crc_resnet18(imagenet_weights=imagenet_weights)
        model = model.cuda()
        optimizer = torch.optim.Adam(crc_adam_groups(model))
        criterion = nn.CrossEntropyLoss()
        inner_train, internal_validation, unused = _balanced_crc_inner(train_rows, seed + fold_for_seed * 1009)
    checkpoint = output_root / 'best.pt'
    last_checkpoint = output_root / 'last.pt'
    progress_path = output_root / 'progress.jsonl'
    history: list[dict[str, Any]] = []
    generator = torch.Generator(device='cuda').manual_seed(seed + 73)
    best_metric, best_step, stale = (-math.inf, -1, 0)
    global_step = 0
    start_epoch = 0
    if last_checkpoint.is_file():
        resume = torch.load(last_checkpoint, map_location='cuda', weights_only=False)
        if resume.get('protocol_id') != PROTOCOL_ID or resume.get('dataset') != dataset or resume.get('fold') != fold or (int(resume.get('seed', -1)) != seed) or (bool(resume.get('official_test', False)) != official_test):
            raise RuntimeError('paper-RGB last checkpoint provenance mismatch')
        model.load_state_dict(resume['model_state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        history = list(resume['history'])
        best_metric, best_step, stale = (float(resume['best_metric']), int(resume['best_step']), int(resume['stale']))
        global_step, start_epoch = (int(resume['global_step']), int(resume['epoch']) + 1)
        generator_state = resume['augmentation_generator_state']
        if not torch.is_tensor(generator_state) or generator_state.dtype != torch.uint8:
            raise RuntimeError('paper-RGB augmentation generator state is invalid')
        generator.set_state(generator_state.detach().cpu())
    output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, spec.max_epochs):
        model.train()
        loader = _loader(inner_train, cache_root, spec, shuffle=True, seed=seed + epoch * 97)
        running = 0.0
        for batch in loader:
            images = _gpu_preprocess(batch['image'], dataset, train=True, generator=generator)
            labels = batch['label'].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            global_step += 1
            if dataset == 'tcga_crc_msi' and global_step % 256 == 0:
                val_logits, val_labels = _predict(model, internal_validation, cache_root, spec, seed)
                metric = float((val_logits.argmax(1) == val_labels).mean())
                history.append({'epoch': epoch, 'step': global_step, 'train_loss': running / max(1, len(inner_train)), 'internal_tile_accuracy': metric})
                if metric > best_metric + 1e-12:
                    best_metric, best_step, stale = (metric, global_step, 0)
                    atomic_torch_save(checkpoint, {'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test, 'best_step': global_step, 'selection_metric': 'internal_validation_tile_accuracy', 'selection_value': metric, 'model_state_dict': model.state_dict(), 'trainable_names': trainable_names, 'spec': asdict(spec)})
                else:
                    stale += 1
                atomic_torch_save(last_checkpoint, {'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test, 'epoch': epoch, 'global_step': global_step, 'best_metric': best_metric, 'best_step': best_step, 'stale': stale, 'history': history, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'augmentation_generator_state': generator.get_state()})
                if stale >= 3:
                    break
        if dataset == 'sicapv2':
            if official_test:
                metric = None
                history.append({'epoch': epoch, 'train_loss': running / len(inner_train)})
            else:
                val_logits, val_labels = _predict(model, validation_rows, cache_root, spec, seed)
                metric = _qwk(val_labels, val_logits.argmax(1))
                history.append({'epoch': epoch, 'train_loss': running / len(inner_train), 'outer_validation_qwk_audit': metric})
            if epoch == spec.max_epochs - 1:
                best_metric, best_step = (float('nan') if metric is None else metric, epoch)
                atomic_torch_save(checkpoint, {'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test, 'best_step': epoch, 'selection_metric': 'fixed_epoch_200', 'selection_value': metric, 'model_state_dict': model.state_dict(), 'spec': asdict(spec)})
            atomic_torch_save(last_checkpoint, {'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test, 'epoch': epoch, 'global_step': global_step, 'best_metric': best_metric, 'best_step': best_step, 'stale': stale, 'history': history, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'augmentation_generator_state': generator.get_state()})
            with progress_path.open('a', encoding='utf-8') as stream:
                if metric is None:
                    stream.write(f'{{"epoch":{epoch},"epochs":{spec.max_epochs},"official_full_train":true}}\n')
                else:
                    stream.write(f'{{"epoch":{epoch},"epochs":{spec.max_epochs},"outer_validation_qwk_audit":{metric}}}\n')
        elif stale >= 3:
            break
    if not checkpoint.is_file():
        raise RuntimeError('paper-RGB training did not write a checkpoint')
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    val_logits, val_labels = _predict(model, validation_rows, prediction_cache_root, spec, seed, collect_labels=not official_test)
    if official_test:
        outer_metric = None
        probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    elif dataset == 'sicapv2':
        assert val_labels is not None
        outer_metric = _qwk(val_labels, val_logits.argmax(1))
        probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    else:
        probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        outer_metric = _patient_auc(validation_rows, probabilities[:, 1])
    prediction_rows = []
    for index, row in enumerate(validation_rows):
        value = {'graph_id': row['graph_id'], 'patient_id': row['patient_id'], 'fold': fold, 'seed': seed, 'logits': val_logits[index].tolist(), 'probabilities': probabilities[index].tolist()}
        if not official_test:
            value['label_id'] = int(row['label_id'])
        prediction_rows.append(value)
    prediction_path = checkpoint.with_name('official_test_predictions.parquet' if official_test else 'validation_predictions.parquet')
    atomic_parquet(prediction_path, prediction_rows)
    manifest = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test': official_test, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'predictions': str(prediction_path), 'predictions_sha256': sha256(prediction_path), 'outer_validation_metric': outer_metric, 'metric': 'not_computed_during_prediction' if official_test else 'QWK' if dataset == 'sicapv2' else 'patient_AUROC_hard_vote', 'training_rows': len(inner_train), 'validation_rows': len(validation_rows), 'history': history, 'runtime': runtime_identity(), 'spec': asdict(spec), 'source_input': 'RGB224 uint8 cache derived only from labels_splits.source_path', 'test_labels_read_during_prediction': False if official_test else None}
    atomic_json(checkpoint.with_name('job.json'), manifest)
    return manifest
