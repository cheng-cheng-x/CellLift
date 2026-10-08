from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import hashlib
from celllift.runtime import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.models import CRCTileClassifier, SICAPFSConv
from celllift.set_encoding.experiment import SCREENING_PROTOCOL_ID
from celllift.set_encoding.tasks import binary_auroc, historical_tile_hard_vote, inverse_frequency_class_weights, quadratic_weighted_kappa
DATASETS = frozenset({'sicapv2', 'tcga_crc_msi'})

@dataclass(frozen=True)
class RGBTrainingSpec:
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 32
    num_workers: int = 4
    amp: bool = True

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('invalid optimizer parameters')
        if self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError('max_epochs and patience must be positive')
        if self.batch_size <= 0 or self.num_workers < 0:
            raise ValueError('invalid data-loader parameters')

def seed_everything(seed: int) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass

def _sha256(path: Path, chunk_size: int=8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _load_eligible_graphs(path: Path | None) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    result: dict[str, str] = {}
    for row in _read_parquet_rows(path):
        patch_id = str(row['patch_id'])
        if patch_id in result:
            raise RuntimeError(f'duplicate patch_id in graph index: {patch_id}')
        result[patch_id] = str(row['graph_id'])
    if not result:
        raise RuntimeError(f'empty graph index: {path}')
    return result

def select_fold_rows(rows: Sequence[dict[str, Any]], *, fold: int, eligible_graphs: dict[str, str] | None=None, include_test: bool=False) -> dict[str, list[dict[str, Any]]]:
    selected = {'train': [], 'validation': [], 'official_test': []}
    for original in rows:
        row = dict(original)
        patch_id = str(row['patch_id'])
        if eligible_graphs is not None:
            if patch_id not in eligible_graphs:
                continue
            row['graph_id'] = eligible_graphs[patch_id]
        else:
            row.setdefault('graph_id', patch_id)
        official = str(row['official_split']).lower()
        if official == 'train':
            if row.get('validation_fold') is None:
                raise RuntimeError(f'training row lacks validation_fold: {patch_id}')
            logical = 'validation' if int(row['validation_fold']) == fold else 'train'
            row['rgb_split'] = logical
            selected[logical].append(row)
        elif official == 'test':
            if include_test:
                row['rgb_split'] = 'official_test'
                selected['official_test'].append(row)
        else:
            raise RuntimeError(f'unexpected official_split {official!r}')
    if not selected['train'] or not selected['validation']:
        raise RuntimeError(f'fold {fold} has an empty training or validation split')
    if not include_test and selected['official_test']:
        raise AssertionError('official TEST leaked into a default fold selection')
    return selected

class RGBPatchDataset(Dataset[dict[str, Any]]):

    def __init__(self, rows: Sequence[dict[str, Any]], dataset: str) -> None:
        if dataset not in DATASETS:
            raise ValueError(f'unsupported dataset {dataset!r}')
        self.rows = list(rows)
        self.dataset = dataset
        from torchvision import transforms
        operations: list[Any] = [transforms.Resize((224, 224)), transforms.ToTensor()]
        if dataset == 'tcga_crc_msi':
            operations.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image
        row = self.rows[index]
        image_path = Path(str(row.get('rgb_path') or row.get('source_path') or ''))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert('RGB'))
        return {'image': tensor, 'label': int(row['label_id']), 'patch_id': str(row['patch_id']), 'graph_id': str(row['graph_id']), 'patient_id': str(row['patient_id']), 'rgb_split': str(row['rgb_split']), 'official_split': str(row['official_split']), 'validation_fold': -1 if row.get('validation_fold') is None else int(row['validation_fold'])}

def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _loader(rows: Sequence[dict[str, Any]], dataset: str, spec: RGBTrainingSpec, seed: int, *, shuffle: bool, device: torch.device) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(RGBPatchDataset(rows, dataset), batch_size=spec.batch_size, shuffle=shuffle, num_workers=spec.num_workers, pin_memory=device.type == 'cuda', persistent_workers=spec.num_workers > 0, worker_init_fn=_worker_seed, generator=generator)

def build_rgb_model(dataset: str, *, imagenet_weights: bool=True) -> nn.Module:
    if dataset == 'sicapv2':
        return SICAPFSConv()
    if dataset == 'tcga_crc_msi':
        return CRCTileClassifier(imagenet_weights=imagenet_weights)
    raise ValueError(f'unsupported dataset {dataset!r}')

def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)

def _train_epoch(model: nn.Module, loader: DataLoader[dict[str, Any]], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, dataset: str, device: torch.device, class_weights: Tensor | None, amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            output = model(images)
            if dataset == 'sicapv2':
                loss = F.cross_entropy(output['logits'], targets, weight=class_weights)
            else:
                loss = F.binary_cross_entropy_with_logits(output['logits'], targets.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * images.shape[0]
        total_examples += images.shape[0]
    return total_loss / max(1, total_examples)

@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader[dict[str, Any]], dataset: str, device: torch.device, class_weights: Tensor | None, amp: bool) -> tuple[float, float]:
    model.eval()
    losses: list[tuple[float, int]] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    patient_ids: list[str] = []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)
        with _autocast(device, amp):
            output = model(images)
            if dataset == 'sicapv2':
                loss = F.cross_entropy(output['logits'], targets, weight=class_weights)
                probability = torch.softmax(output['logits'].float(), dim=1)
            else:
                loss = F.binary_cross_entropy_with_logits(output['logits'], targets.float())
                probability = torch.sigmoid(output['logits'].float())
        losses.append((float(loss), images.shape[0]))
        labels.append(targets.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        patient_ids.extend((str(value) for value in batch['patient_id']))
    truth = np.concatenate(labels)
    prediction = np.concatenate(probabilities)
    if dataset == 'sicapv2':
        metric = quadratic_weighted_kappa(truth, prediction.argmax(axis=1), 4)
    else:
        ordered_patients, patient_scores = historical_tile_hard_vote(prediction, patient_ids)
        patient_truth: list[int] = []
        ids = np.asarray(patient_ids, dtype=object)
        for patient in ordered_patients:
            values = np.unique(truth[ids == patient])
            if values.size != 1:
                raise RuntimeError(f'CRC patient has inconsistent validation labels: {patient}')
            patient_truth.append(int(values[0]))
        metric = binary_auroc(patient_truth, patient_scores)
    denominator = sum((count for _, count in losses))
    mean_loss = sum((value * count for value, count in losses)) / max(1, denominator)
    return (float(mean_loss), float(metric))

def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get('best_frozen') is not True:
        raise RuntimeError(f'checkpoint is not a frozen-best RGB baseline: {path}')
    return payload

def train_rgb_baseline(*, dataset: str, train_rows: Sequence[dict[str, Any]], validation_rows: Sequence[dict[str, Any]], fold: int, seed: int, checkpoint_path: Path, spec: RGBTrainingSpec, device: torch.device, imagenet_weights: bool=True) -> dict[str, Any]:
    seed_everything(seed)
    model = build_rgb_model(dataset, imagenet_weights=imagenet_weights).to(device)
    trainable_names = tuple((name for name, parameter in model.named_parameters() if parameter.requires_grad))
    trainable_backbone_names = tuple(getattr(model, 'trainable_backbone_parameter_names', ()))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    amp = bool(spec.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    class_weights = None
    if dataset == 'sicapv2':
        class_weights = inverse_frequency_class_weights([int(row['label_id']) for row in train_rows]).to(device)
    train_loader = _loader(train_rows, dataset, spec, seed, shuffle=True, device=device)
    validation_loader = _loader(validation_rows, dataset, spec, seed + 1, shuffle=False, device=device)
    best_metric = -float('inf')
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(spec.max_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, dataset, device, class_weights, amp)
        validation_loss, validation_metric = _validate(model, validation_loader, dataset, device, class_weights, amp)
        if not np.isfinite(validation_metric):
            raise RuntimeError('validation selection metric is not finite')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss, 'validation_metric': validation_metric})
        if validation_metric > best_metric + 1e-12:
            best_metric = validation_metric
            best_epoch = epoch
            stale_epochs = 0
            payload = {'best_frozen': True, 'dataset': dataset, 'fold': fold, 'seed': seed, 'epoch': epoch, 'selection_split': 'validation', 'selection_metric': 'qwk' if dataset == 'sicapv2' else 'patient_auroc_hard_vote', 'selection_value': validation_metric, 'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'trainable_parameter_names': trainable_names, 'trainable_backbone_parameter_names': trainable_backbone_names, 'training_spec': asdict(spec)}
            _atomic_torch_save(checkpoint_path, payload)
        else:
            stale_epochs += 1
            if stale_epochs >= spec.patience:
                break
    if best_epoch < 0 or not checkpoint_path.is_file():
        raise RuntimeError('training failed to produce a frozen-best checkpoint')
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return {'model': model, 'checkpoint': checkpoint, 'checkpoint_sha256': _sha256(checkpoint_path), 'history': history, 'epochs_ran': len(history)}

def load_rgb_baseline(*, dataset: str, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if checkpoint.get('dataset') != dataset:
        raise RuntimeError('checkpoint dataset does not match requested dataset')
    model = build_rgb_model(dataset, imagenet_weights=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return {'model': model, 'checkpoint': checkpoint, 'checkpoint_sha256': _sha256(checkpoint_path), 'history': [], 'epochs_ran': 0}

def _feature_table(batch: dict[str, Any], features: np.ndarray, *, fold: int, seed: int, checkpoint_sha256: str, rgb_scores: np.ndarray | None=None):
    import pyarrow as pa
    if features.ndim != 2 or features.shape[1] != 512:
        raise RuntimeError(f'RGB feature width must be 512, got {features.shape}')
    flattened = pa.array(features.astype(np.float32, copy=False).reshape(-1), type=pa.float32())
    feature_array = pa.FixedSizeListArray.from_arrays(flattened, 512)
    columns = {'graph_id': pa.array([str(value) for value in batch['graph_id']]), 'patch_id': pa.array([str(value) for value in batch['patch_id']]), 'patient_id': pa.array([str(value) for value in batch['patient_id']]), 'split': pa.array([str(value) for value in batch['rgb_split']]), 'official_split': pa.array([str(value) for value in batch['official_split']]), 'validation_fold': pa.array(np.asarray(batch['validation_fold'], dtype=np.int16)), 'label_id': pa.array(np.asarray(batch['label'], dtype=np.int8)), 'fold': pa.array(np.full(features.shape[0], fold, dtype=np.int16)), 'seed': pa.array(np.full(features.shape[0], seed, dtype=np.int32)), 'checkpoint_sha256': pa.array([checkpoint_sha256] * features.shape[0]), 'rgb_feature': feature_array}
    if rgb_scores is not None:
        scores = np.asarray(rgb_scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != features.shape[0] or not np.isfinite(scores).all():
            raise RuntimeError('RGB audit scores must be finite and row aligned')
        columns['rgb_score'] = pa.array(scores, type=pa.float32())
    return pa.table(columns)

@torch.no_grad()
def cache_rgb_features(*, model: nn.Module, rows: Sequence[dict[str, Any]], dataset: str, fold: int, seed: int, checkpoint_sha256: str, destination: Path, spec: RGBTrainingSpec, device: torch.device) -> dict[str, Any]:
    import pyarrow.parquet as pq
    loader = _loader(rows, dataset, spec, seed + 2, shuffle=False, device=device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    writer = None
    count = 0
    split_counts: dict[str, int] = {}
    model.eval()
    try:
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            output = model(images)
            features = output['features'].float().cpu().numpy()
            rgb_scores = None
            if dataset == 'tcga_crc_msi':
                rgb_scores = torch.sigmoid(output['logits'].float()).cpu().numpy()
            table = _feature_table(batch, features, fold=fold, seed=seed, checkpoint_sha256=checkpoint_sha256, rgb_scores=rgb_scores)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
            count += features.shape[0]
            for split in batch['rgb_split']:
                split_counts[str(split)] = split_counts.get(str(split), 0) + 1
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError('cannot write an empty RGB feature cache')
    os.replace(temporary, destination)
    return {'path': str(destination), 'sha256': _sha256(destination), 'rows': count, 'split_counts': split_counts, 'feature_dim': 512, 'schema': 'fixed_size_list<float32>[512]', 'audit_score_column': 'rgb_score' if dataset == 'tcga_crc_msi' else None}

def _require_test_gate(result_root: Path) -> dict[str, Any]:
    gate_path = result_root / 'selection_frozen.json'
    if not gate_path.is_file():
        raise RuntimeError('official TEST is locked: selection_frozen.json is absent')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('status') != 'PASS':
        raise RuntimeError('official TEST is locked: selection_frozen status is not PASS')
    if gate.get('protocol_id') != SCREENING_PROTOCOL_ID:
        raise RuntimeError('official TEST is locked: incompatible downstream protocol')
    return gate

def extract_rgb(cfg: dict[str, Any], *, fold: int, seed: int, include_test: bool=False, train_baseline: bool=True, device: str | None=None, batch_size: int=32, num_workers: int=4, imagenet_weights: bool=True) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    if dataset not in DATASETS:
        raise ValueError(f'unsupported dataset {dataset!r}')
    validation_folds = int(cfg['split']['validation_folds'])
    if fold < 0 or fold >= validation_folds:
        raise ValueError(f'fold must be in [0, {validation_folds - 1}]')
    if include_test and train_baseline:
        raise RuntimeError('include_test requires train_baseline=False so TEST reuses an already frozen best')
    data_root = Path(cfg['paths']['data_root'])
    result_root = Path(cfg['paths']['result_root'])
    model_input_root = Path(cfg['paths']['model_input_root'])
    labels_path = model_input_root / '04_labels_splits' / 'labels_splits.parquet'
    graph_index_path = model_input_root / '03_graph_cache' / 'graph_index.parquet'
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)
    if include_test:
        gate = _require_test_gate(result_root)
    else:
        gate = None
    spec = RGBTrainingSpec(learning_rate=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']), max_epochs=int(cfg['training']['max_epochs']), patience=int(cfg['training']['patience']), batch_size=batch_size, num_workers=num_workers, amp=str(cfg['training'].get('amp', 'fp16')).lower() == 'fp16')
    target = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    seed_everything(seed)
    rows = _read_parquet_rows(labels_path)
    eligible = _load_eligible_graphs(graph_index_path)
    selected = select_fold_rows(rows, fold=fold, eligible_graphs=eligible, include_test=include_test)
    run_key = f'fold_{fold:02d}/seed_{seed}'
    checkpoint_path = result_root / 'checkpoints' / 'rgb' / run_key / 'best.pt'
    if train_baseline:
        trained = train_rgb_baseline(dataset=dataset, train_rows=selected['train'], validation_rows=selected['validation'], fold=fold, seed=seed, checkpoint_path=checkpoint_path, spec=spec, device=target, imagenet_weights=imagenet_weights)
    else:
        trained = load_rgb_baseline(dataset=dataset, checkpoint_path=checkpoint_path, device=target)
        checkpoint = trained['checkpoint']
        if int(checkpoint['fold']) != fold or int(checkpoint['seed']) != seed:
            raise RuntimeError('frozen checkpoint fold/seed does not match request')
    extraction_rows = selected['train'] + selected['validation']
    if include_test:
        extraction_rows += selected['official_test']
    cache_path = data_root / '05_rgb_features' / run_key / 'rgb_features.parquet'
    cache = cache_rgb_features(model=trained['model'], rows=extraction_rows, dataset=dataset, fold=fold, seed=seed, checkpoint_sha256=trained['checkpoint_sha256'], destination=cache_path, spec=spec, device=target)
    checkpoint = trained['checkpoint']
    manifest = {'status': 'PASS', 'dataset': dataset, 'fold': fold, 'seed': seed, 'official_test_included': include_test, 'test_gate': gate, 'checkpoint_path': str(checkpoint_path), 'checkpoint_sha256': trained['checkpoint_sha256'], 'checkpoint_best_epoch': int(checkpoint['epoch']), 'checkpoint_selection_metric': checkpoint['selection_metric'], 'checkpoint_selection_value': float(checkpoint['selection_value']), 'checkpoint_best_frozen': bool(checkpoint['best_frozen']), 'trainable_parameter_names': list(checkpoint['trainable_parameter_names']), 'trainable_backbone_parameter_names': list(checkpoint.get('trainable_backbone_parameter_names', ())), 'model_class': trained['model'].__class__.__name__, 'rgb_preprocessing': {'resize_hw': [224, 224], 'range': 'ToTensor [0,1]', 'normalization': 'ImageNet mean/std' if dataset == 'tcga_crc_msi' else 'none'}, 'training_spec': asdict(spec), 'epochs_ran': int(trained['epochs_ran']), 'history': trained['history'], 'labels_manifest': str(labels_path), 'labels_manifest_sha256': _sha256(labels_path), 'eligible_graph_index': str(graph_index_path) if eligible is not None else None, 'eligible_graph_count': len(eligible) if eligible is not None else None, 'cache': cache}
    manifest_path = cache_path.with_name('manifest.json')
    _atomic_json(manifest_path, manifest)
    manifest['manifest_path'] = str(manifest_path)
    return manifest

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--reuse-best', action='store_true', help='do not train; load the frozen best checkpoint')
    parser.add_argument('--no-imagenet-weights', action='store_true', help='offline smoke testing only')
    return parser

def main(argv: list[str] | None=None) -> int:
    from celllift.runtime import yaml
    args = build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    result = extract_rgb(cfg, fold=args.fold, seed=args.seed, include_test=args.include_test, train_baseline=not args.reuse_best, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers, imagenet_weights=not args.no_imagenet_weights)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
