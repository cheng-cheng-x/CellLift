from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import random
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterator, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from celllift.breast_roi_baseline.cache import atomic_json, sha256_file
from celllift.breast_roi_baseline.features import FoldFeatureNormalizer, build_token_pair
from celllift.breast_roi_baseline.models import BRACSROIModel, GeometryTileEncoder
from celllift.breast_roi_baseline.probe import ProbeArrays, load_anchor_arrays
from celllift.breast_roi_baseline.rgb import BRACSRGBMaskROIModel
from celllift.breast_roi_baseline.shuffle import DonorRow, TileRecord, assign_count_deciles, build_donor_mapping, mapping_checksum, validate_donor_mapping
EXPERT_TO_ARM = {'E0_RGB_MASK2D': 'mask2d', 'E1_MASK2D': 'mask2d', 'E2_MASK_DIRECT3D': 'direct3d', 'E3_MASK_SHUF_DIRECT3D': 'shuffled_direct3d', 'E4_MASK_RESIDUAL3D': 'residual3d', 'E5_MASK_SHUF_RESIDUAL3D': 'shuffled_residual3d'}

def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _fixed_list_numpy(column: Any, width: int) -> np.ndarray:
    values = column.combine_chunks()
    return np.asarray(values.values.to_numpy(zero_copy_only=False), np.float32).reshape(len(values), width)

def _load_residual(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['graph_id', 'anchor_id', 'residual9'], partitioning=None)
    values = table.to_pydict()
    residual = _fixed_list_numpy(table['residual9'], 9)
    return {(str(graph), int(anchor)): residual[index] for index, (graph, anchor) in enumerate(zip(values['graph_id'], values['anchor_id']))}

def _align_residual(values: Mapping[tuple[str, int], np.ndarray], graph_id: np.ndarray, anchor_id: np.ndarray, roles: np.ndarray, valid_ncr: np.ndarray) -> np.ndarray:
    residual = np.zeros((len(graph_id), 9), dtype=np.float32)
    missing: list[tuple[str, int]] = []
    for index in np.flatnonzero(roles != 'excluded'):
        key = (str(graph_id[index]), int(anchor_id[index]))
        value = values.get(key)
        if value is None:
            missing.append(key)
            if len(missing) >= 10:
                break
        else:
            residual[index] = value
    if missing:
        raise RuntimeError(f'probe residual coverage is missing required non-excluded anchors: {missing}')
    residual[~valid_ncr, 8] = 0.0
    return residual

def _load_or_build_shuffle_mapping(path: str | Path, records: Sequence[TileRecord], *, seed: int, fold: int, phase: str) -> tuple[DonorRow, ...]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    materialized = tuple(records)
    if any((row.count_decile is None for row in materialized)):
        if any((row.count_decile is not None for row in materialized)):
            raise ValueError('count_decile must be supplied for every tile or no tile')
        materialized = assign_count_deciles(materialized)
    destination = Path(path)
    if destination.is_file():
        rows = tuple((DonorRow(**row) for row in pq.read_table(destination, partitioning=None).to_pylist()))
        validate_donor_mapping(materialized, rows)
        return rows
    rows = build_donor_mapping(materialized, seed=seed, fold=fold, role=phase)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist([row.as_dict() for row in rows]), temporary, compression='zstd', row_group_size=65536)
    os.replace(temporary, destination)
    reloaded = tuple((DonorRow(**row) for row in pq.read_table(destination, partitioning=None).to_pylist()))
    validate_donor_mapping(materialized, reloaded)
    if mapping_checksum(reloaded) != mapping_checksum(rows):
        raise RuntimeError('persisted shuffle mapping checksum changed')
    return reloaded

def _logical_roles(arrays: ProbeArrays, phase: str, fold: int) -> np.ndarray:
    roles = np.full(len(arrays.rays), 'excluded', dtype=object)
    folds = arrays.validation_fold if phase == 'development' else arrays.final_validation_fold
    eligible_split = np.isin(arrays.split, ['train'] if phase == 'development' else ['train', 'val'])
    roles[eligible_split & (folds != fold)] = 'train'
    roles[eligible_split & (folds == fold)] = 'heldout'
    roles[arrays.split == ('val' if phase == 'development' else 'test')] = 'external'
    return roles

class BRACSFeatureStore:

    def __init__(self, *, anchor_manifest: str | Path, tile_manifest: str | Path, expert: str, phase: str, fold: int, seed: int, residual_path: str | Path | None=None, shuffle_map_path: str | Path | None=None) -> None:
        if expert not in EXPERT_TO_ARM:
            raise ValueError(f'unknown expert {expert!r}')
        if phase not in {'development', 'final_test'}:
            raise ValueError('invalid phase')
        self.expert, self.arm = (expert, EXPERT_TO_ARM[expert])
        self.arrays = load_anchor_arrays(anchor_manifest)
        self.roles = _logical_roles(self.arrays, phase, int(fold))
        training = self.roles == 'train'
        self.normalizer = FoldFeatureNormalizer.fit(self.arrays.raw_geometry, self.arrays.valid_ncr, training)
        ray_mean = self.arrays.rays[training].mean(0)
        ray_std = np.maximum(self.arrays.rays[training].std(0), 1e-06)
        self.rays = ((self.arrays.rays - ray_mean) / ray_std).astype(np.float32)
        self.direct = self.normalizer.transform_direct(self.arrays.raw_geometry, self.arrays.valid_ncr)
        self.residual = None
        if 'residual' in self.arm:
            if residual_path is None:
                raise ValueError('residual expert requires probe residual_path')
            values = _load_residual(residual_path)
            self.residual = _align_residual(values, self.arrays.graph_id, self.arrays.anchor_id, self.roles, self.arrays.valid_ncr)
        self.tile_rows = {str(row['graph_id']): row for row in _read_rows(tile_manifest)}
        self.graph_indices: dict[str, np.ndarray] = {}
        graph_code = self.arrays.graph_code
        direct_boundaries = np.flatnonzero(np.r_[True, graph_code[1:] != graph_code[:-1], True])
        if len(direct_boundaries) - 1 == len(np.unique(graph_code)):
            order = None
            boundaries = direct_boundaries
        else:
            order = np.lexsort((self.arrays.anchor_id, graph_code))
            ordered_code = graph_code[order]
            boundaries = np.flatnonzero(np.r_[True, ordered_code[1:] != ordered_code[:-1], True])
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            selected = np.arange(begin, end, dtype=np.int64) if order is None else order[begin:end]
            if len(selected) > 1:
                selected = selected[np.argsort(self.arrays.anchor_id[selected], kind='stable')]
            graph = str(self.arrays.graph_id[selected[0]])
            self.graph_indices[graph] = selected
        if set(self.graph_indices) - set(self.tile_rows):
            raise RuntimeError('anchor graphs are absent from tile manifest')
        self.donor = np.empty(0, dtype=np.int64)
        self._shuffled: np.ndarray | None = None
        if self.arm.startswith('shuffled'):
            records = []
            for graph, indices in self.graph_indices.items():
                role = str(self.roles[indices[0]])
                if role == 'excluded':
                    continue
                row = self.tile_rows[graph]
                records.append(TileRecord(graph, str(row['roi_id']), str(row['wsi_id']), role, len(indices)))
            if shuffle_map_path is None:
                raise ValueError('shuffled expert requires a persisted shuffle_map_path')
            mapping = _load_or_build_shuffle_mapping(shuffle_map_path, records, seed=seed, fold=fold, phase=phase)
            donor_index = np.full(len(self.arrays.rays), -1, dtype=np.int64)
            for row in mapping:
                target = self.graph_indices[row.target_tile_id][row.target_anchor_index]
                donor = self.graph_indices[row.donor_tile_id][row.donor_anchor_index]
                donor_index[target] = donor
            mapped = self.roles != 'excluded'
            if np.any(donor_index[mapped] < 0) or np.any(donor_index[~mapped] >= 0):
                raise RuntimeError('shuffle mapping coverage disagrees with logical roles')
            source = self.direct if self.arm == 'shuffled_direct3d' else self.residual
            assert source is not None
            self._shuffled = np.zeros_like(source, dtype=np.float32)
            self._shuffled[mapped] = source[donor_index[mapped]]
            self.donor = donor_index[mapped]
        self.statistics = {'ray_mean': ray_mean.tolist(), 'ray_std': ray_std.tolist(), 'geometry': self.normalizer.as_dict(), 'training_anchors': int(training.sum())}

    def tokens(self, graph: str) -> tuple[np.ndarray, np.ndarray]:
        indices = self.graph_indices[graph]
        kwargs: dict[str, np.ndarray] = {}
        if self.arm == 'direct3d':
            kwargs['direct3d'] = self.direct[indices]
        elif self.arm == 'residual3d':
            assert self.residual is not None
            kwargs['residual3d'] = self.residual[indices]
        elif self.arm in {'shuffled_direct3d', 'shuffled_residual3d'}:
            assert self._shuffled is not None
            kwargs[self.arm] = self._shuffled[indices]
        return build_token_pair(self.rays[indices], self.arm, **kwargs)

class BRACSROIDataset(Dataset[dict[str, Any]]):

    def __init__(self, store: BRACSFeatureStore, role: str, task: str) -> None:
        if role not in {'train', 'heldout', 'external'} or task not in {'t7', 't3'}:
            raise ValueError('invalid ROI dataset role/task')
        self.store, self.role, self.task = (store, role, task)
        grouped: dict[str, list[str]] = defaultdict(list)
        for graph, indices in store.graph_indices.items():
            if str(store.roles[indices[0]]) == role:
                grouped[str(store.tile_rows[graph]['roi_id'])].append(graph)
        self.rois = tuple(sorted(grouped))
        self.graphs = {roi: tuple(sorted(grouped[roi])) for roi in self.rois}
        if not self.rois:
            raise RuntimeError(f'empty ROI dataset role: {role}')

    def __len__(self) -> int:
        return len(self.rois)

    def tile_count(self, index: int) -> int:
        return len(self.graphs[self.rois[index]])

    def object_count(self, index: int) -> int:
        return sum((len(self.store.graph_indices[graph]) for graph in self.graphs[self.rois[index]]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        roi = self.rois[index]
        graphs = self.graphs[roi]
        rows = [self.store.tile_rows[graph] for graph in graphs]
        labels = {int(row['label_7'] if self.task == 't7' else row['label_3']) for row in rows}
        wsis = {str(row['wsi_id']) for row in rows}
        if len(labels) != 1 or len(wsis) != 1:
            raise RuntimeError('ROI tiles have inconsistent label or WSI')
        pairs = [self.store.tokens(graph) for graph in graphs]
        result = {'roi_id': roi, 'wsi_id': next(iter(wsis)), 'label': next(iter(labels)), 'graphs': graphs, 'nucleus': [pair[0] for pair in pairs], 'cell': [pair[1] for pair in pairs]}
        if self.store.expert == 'E0_RGB_MASK2D':
            result['image_paths'] = [str(row.get('tile_path') or row.get('rgb_path')) for row in rows]
        return result

class TileBudgetBatchSampler(Sampler[list[int]]):

    def __init__(self, dataset: BRACSROIDataset, budget: int, object_budget: int, shuffle: bool, seed: int) -> None:
        self.dataset, self.budget = (dataset, int(budget))
        self.object_budget, self.shuffle, self.seed = (int(object_budget), bool(shuffle), int(seed))
        if self.budget <= 0:
            raise ValueError('tile budget must be positive')
        if self.object_budget <= 0:
            raise ValueError('object budget must be positive')
        self.max_single_roi_tiles = max((dataset.tile_count(i) for i in range(len(dataset))))
        self.max_single_roi_objects = max((dataset.object_count(i) for i in range(len(dataset))))
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        batch, cost, objects = ([], 0, 0)
        for index in indices:
            value = self.dataset.tile_count(index)
            object_value = self.dataset.object_count(index)
            if batch and (cost + value > self.budget or objects + object_value > self.object_budget):
                yield batch
                batch, cost, objects = ([], 0, 0)
            batch.append(index)
            cost += value
            objects += object_value
        if batch:
            yield batch

    def __len__(self) -> int:
        tiles = sum((self.dataset.tile_count(i) for i in range(len(self.dataset))))
        objects = sum((self.dataset.object_count(i) for i in range(len(self.dataset))))
        return max(1, (tiles + self.budget - 1) // self.budget, (objects + self.object_budget - 1) // self.object_budget)

def collate_rois(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError('cannot collate empty ROI batch')
    nucleus = [value for sample in samples for value in sample['nucleus']]
    cell = [value for sample in samples for value in sample['cell']]
    maximum = max((len(value) for value in nucleus))
    n = np.zeros((len(nucleus), maximum, 41), np.float32)
    c = np.zeros_like(n)
    nm = np.zeros((len(nucleus), maximum), bool)
    cm = np.zeros_like(nm)
    for index, (nv, cv) in enumerate(zip(nucleus, cell)):
        n[index, :len(nv)] = nv
        c[index, :len(cv)] = cv
        nm[index, :len(nv)] = True
        cm[index, :len(cv)] = True
    lengths = [len(sample['graphs']) for sample in samples]
    output: dict[str, Any] = {'roi_ids': [str(sample['roi_id']) for sample in samples], 'wsi_ids': [str(sample['wsi_id']) for sample in samples], 'labels': torch.tensor([int(sample['label']) for sample in samples], dtype=torch.long), 'roi_offsets': torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.long), 'nucleus_tokens': torch.from_numpy(n), 'cell_tokens': torch.from_numpy(c), 'nucleus_mask': torch.from_numpy(nm), 'cell_mask': torch.from_numpy(cm)}
    if 'image_paths' in samples[0]:
        from PIL import Image
        from torchvision import transforms
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        images = []
        for sample in samples:
            for path in sample['image_paths']:
                with Image.open(path) as image:
                    images.append(transform(image.convert('RGB')))
        output['images'] = torch.stack(images)
    return output

def _macro_f1(truth: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability.argmax(1)
    values = []
    for label in range(probability.shape[1]):
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        values.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    return float(np.mean(values))

def _forward(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    kwargs = {name: batch[name].to(device, non_blocking=True) for name in ('roi_offsets', 'nucleus_tokens', 'cell_tokens', 'nucleus_mask', 'cell_mask')}
    if 'images' in batch:
        return model(images=batch['images'].to(device, non_blocking=True), **kwargs)
    return model(**kwargs)

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    rows = []
    for batch in loader:
        probability = torch.softmax(_forward(model, batch, device)['logits'].float(), 1).cpu().numpy()
        for index, roi in enumerate(batch['roi_ids']):
            rows.append({'roi_id': roi, 'wsi_id': batch['wsi_ids'][index], 'label': int(batch['labels'][index]), 'probability': probability[index]})
    truth = np.asarray([row['label'] for row in rows])
    probability = np.stack([row['probability'] for row in rows])
    return (_macro_f1(truth, probability), rows)

def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)

def train_expert(*, anchor_manifest: str | Path, tile_manifest: str | Path, residual_path: str | Path | None, shuffle_map_path: str | Path | None, expert: str, task: str, encoder: str, phase: str, fold: int, seed: int, output_dir: str | Path, device: str='cuda', tile_budget: int=256, object_budget: int=32000, max_epochs: int=100, patience: int=15) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(output_dir)
    manifest_path = destination / 'manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            return previous
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    store = BRACSFeatureStore(anchor_manifest=anchor_manifest, tile_manifest=tile_manifest, expert=expert, phase=phase, fold=fold, seed=seed, residual_path=residual_path, shuffle_map_path=shuffle_map_path)
    train_data = BRACSROIDataset(store, 'train', task)
    heldout_data = BRACSROIDataset(store, 'heldout', task)
    external_data = BRACSROIDataset(store, 'external', task)
    workers = 4
    train_loader = DataLoader(train_data, batch_sampler=TileBudgetBatchSampler(train_data, tile_budget, object_budget, True, seed), collate_fn=collate_rois, num_workers=workers, pin_memory=True)
    heldout_loader = DataLoader(heldout_data, batch_sampler=TileBudgetBatchSampler(heldout_data, tile_budget, object_budget, False, seed), collate_fn=collate_rois, num_workers=workers, pin_memory=True)
    external_loader = DataLoader(external_data, batch_sampler=TileBudgetBatchSampler(external_data, tile_budget, object_budget, False, seed), collate_fn=collate_rois, num_workers=workers, pin_memory=True)
    classes = 7 if task == 't7' else 3
    if expert == 'E0_RGB_MASK2D':
        model: nn.Module = BRACSRGBMaskROIModel(set_encoder=encoder, num_classes=classes)
    else:
        model = BRACSROIModel(GeometryTileEncoder(encoder), num_classes=classes)
    target = torch.device(device)
    model.to(target)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler('cuda', enabled=target.type == 'cuda')
    counts = Counter((int(train_data[i]['label']) for i in range(len(train_data))))
    weights = torch.tensor([len(train_data) / (classes * counts[label]) for label in range(classes)], dtype=torch.float32, device=target)
    best, best_epoch, stale = (-float('inf'), -1, 0)
    history = []
    best_path = destination / 'best.pt'
    last_path = destination / 'last.pt'
    start_epoch = 0
    if last_path.is_file():
        resumed = torch.load(last_path, map_location=target, weights_only=False)
        model.load_state_dict(resumed['state_dict'])
        optimizer.load_state_dict(resumed['optimizer'])
        scaler.load_state_dict(resumed['scaler'])
        start_epoch = int(resumed['epoch']) + 1
        best = float(resumed['best'])
        best_epoch = int(resumed['best_epoch'])
        stale = int(resumed['stale'])
        history = list(resumed['history'])
        train_loader.batch_sampler.epoch = start_epoch
        if stale >= patience:
            start_epoch = max_epochs
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total = 0.0
        examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=target.type == 'cuda'):
                logits = _forward(model, batch, target)['logits']
                loss = F.cross_entropy(logits, batch['labels'].to(target), weight=weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch['labels'])
            examples += len(batch['labels'])
        metric, _ = _evaluate(model, heldout_loader, target)
        history.append({'epoch': epoch, 'train_loss': total / max(1, examples), 'heldout_macro_f1': metric})
        if metric > best + 1e-12:
            best, best_epoch, stale = (metric, epoch, 0)
            _atomic_torch(best_path, {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'epoch': epoch, 'metric': metric})
        else:
            stale += 1
        _atomic_torch(last_path, {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'stale': stale, 'history': history, 'tile_budget': tile_budget, 'object_budget': object_budget})
        print(json.dumps({'event': 'expert_epoch', 'expert': expert, 'task': task, 'encoder': encoder, 'fold': int(fold), 'seed': int(seed), **history[-1], 'best': best, 'best_epoch': best_epoch, 'stale': stale}), flush=True)
        if stale >= patience:
            break
    payload = torch.load(best_path, map_location=target, weights_only=False)
    model.load_state_dict(payload['state_dict'])
    heldout_metric, heldout_rows = _evaluate(model, heldout_loader, target)
    external_metric, external_rows = _evaluate(model, external_loader, target)
    prediction_path = destination / 'predictions.parquet'
    temporary = prediction_path.with_name(f'.{prediction_path.name}.tmp.{os.getpid()}')
    combined = [('heldout', row) for row in heldout_rows] + [('external', row) for row in external_rows]
    probabilities = np.stack([row[1]['probability'] for row in combined]).astype(np.float32)
    fixed = pa.FixedSizeListArray.from_arrays(pa.array(probabilities.reshape(-1)), classes)
    pq.write_table(pa.table({'roi_id': pa.array([row[1]['roi_id'] for row in combined]), 'wsi_id': pa.array([row[1]['wsi_id'] for row in combined]), 'label': pa.array(np.asarray([row[1]['label'] for row in combined], np.int8)), 'role': pa.array([row[0] for row in combined]), 'probability': fixed, 'expert': pa.array([expert] * len(combined)), 'task': pa.array([task] * len(combined)), 'encoder': pa.array([encoder] * len(combined)), 'phase': pa.array([phase] * len(combined)), 'fold': pa.array(np.full(len(combined), fold, np.int8)), 'seed': pa.array(np.full(len(combined), seed, np.int32))}), temporary, compression='zstd')
    os.replace(temporary, prediction_path)
    result = {'status': 'PASS', 'expert': expert, 'task': task, 'encoder': encoder, 'phase': phase, 'fold': fold, 'seed': seed, 'best_epoch': best_epoch, 'heldout_macro_f1': heldout_metric, 'external_macro_f1': external_metric, 'statistics': store.statistics, 'history': history, 'checkpoint': str(best_path), 'checkpoint_sha256': sha256_file(best_path), 'predictions': str(prediction_path), 'predictions_sha256': sha256_file(prediction_path)}
    atomic_json(manifest_path, result)
    return result
