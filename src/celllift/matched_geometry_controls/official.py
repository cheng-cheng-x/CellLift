from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import importlib
from celllift.runtime import json
import os
import sys
import time
import numpy as np
from .evaluation import _rows, _seed_set, _softmax, load_crc_baseline, load_geometry, load_sicap_baseline
from .feature_store import FoldModalStore, MeanPoolStore, ModalRow, read_index, write_meanpool_cache
from .fusion import apply, fit_crc, fit_multiclass_alpha
from .io_utils import atomic_json, atomic_npz, atomic_parquet, atomic_torch, safe_component, sha256_file
from .metrics import macro_f1, paired_cluster_bootstrap, qwk, registered_interpretation, t7_to_t3
from .pipeline import all_dataset_rows
from .protocol import require_frozen
from .residual import crossfit_residual
from .shuffle import GraphShuffleRecord, build_donor_map
from .training import _pad_graphs, _sample
ARMS = ('D', 'DS', 'R', 'RS')

def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'PASS'
    except Exception:
        return False

def _projected_rows(path: str | Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=list(columns), partitioning=None).to_pylist()

def _combined_rows(cfg: Mapping[str, Any]):
    root = Path(cfg['paths']['data_root'])
    development = read_index(root / '03_modal_features/index.parquet')
    test = read_index(root / '09_official_test/03_modal_features/index.parquet')
    if any((row.label_id is not None for row in test)):
        raise RuntimeError('official TEST modal index contains labels before the one-time evaluation transaction')
    ids = [row.graph_id for row in development + test]
    if len(ids) != len(set(ids)):
        raise RuntimeError('development/TEST graph IDs overlap')
    return (development, test, development + test)

def _load_arrays(rows: Sequence[ModalRow]):
    arrays = []
    for row in rows:
        with np.load(row.path) as value:
            arrays.append({key: value[key].copy() for key in value.files})
    return arrays

def _scope(dataset: str, fold: int | None):
    return f'fold_{int(fold):02d}' if dataset == 'bracs' else 'final'

def run_official_residual(cfg: Mapping[str, Any], fold: int | None, device: str) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    dataset = cfg['dataset']
    if dataset == 'bracs' and fold is None:
        raise ValueError('BRACS official residual requires a final fold')
    development, test, rows = _combined_rows(cfg)
    arrays = _load_arrays(rows)
    counts = np.asarray([len(value['rays36']) for value in arrays])
    graph_code = np.repeat(np.arange(len(rows)), counts)
    rays = np.concatenate([value['rays36'] for value in arrays])
    direct = np.concatenate([value['direct9'] for value in arrays])
    valid = np.concatenate([value['valid_ncr'] for value in arrays])
    groups = np.repeat([row.wsi_id if dataset == 'bracs' else row.patient_id for row in rows], counts)
    if dataset == 'bracs':
        train_graph = np.asarray([row.split in {'train', 'val', 'validation'} and row.final_fold != fold for row in rows])
        held_graph = np.asarray([row.split in {'train', 'val', 'validation'} and row.final_fold == fold for row in rows])
        external_graph = np.asarray([row.split == 'test' for row in rows])
    else:
        train_graph = np.asarray([row.split == 'train' for row in rows])
        held_graph = np.asarray([row.split == 'test' for row in rows])
        external_graph = np.zeros(len(rows), bool)
    train, held, external = (np.repeat(value, counts) for value in (train_graph, held_graph, external_graph))
    scope = _scope(dataset, fold)
    root = Path(cfg['paths']['data_root']) / f'09_official_test/04_residual/{scope}'
    global_path = root / 'all_residuals.npz'
    manifest = crossfit_residual(rays36=rays, direct9=direct, valid_ncr=valid, graph_code=graph_code, groups=groups, outer_train=train, outer_heldout=held, external=external, conditional_geometry_root=cfg['paths']['mask_probe_conditional_geometry_root'], destination=global_path, device=device)
    with np.load(global_path) as value:
        residual, written = (value['residual9'], value['written'])
    cursor = offset = 0
    for row, count in zip(rows, counts):
        selected = written[offset:offset + count]
        if not selected.all():
            raise RuntimeError(f'official residual coverage incomplete for {row.graph_id}')
        atomic_npz(root / f'{safe_component(row.graph_id)}.npz', residual9=residual[cursor:cursor + count])
        cursor += count
        offset += count
    manifest.update(scope=scope, graphs=len(rows), development_graphs=len(development), test_graphs=len(test))
    atomic_json(root / 'manifest.json', manifest)
    return manifest

def run_official_shuffle(cfg: Mapping[str, Any], fold: int | None) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    dataset = cfg['dataset']
    if dataset == 'bracs' and fold is None:
        raise ValueError('BRACS official shuffle requires a final fold')
    _, _, rows = _combined_rows(cfg)
    records = []
    for row in rows:
        group = row.wsi_id if dataset == 'bracs' else row.patient_id
        role = 'test' if row.split == 'test' else 'heldout' if dataset == 'bracs' and row.final_fold == fold else 'train'
        records.append(GraphShuffleRecord(row.graph_id, group, role, row.count))
    mapping = build_donor_map(records)
    scope = _scope(dataset, fold)
    destination = Path(cfg['paths']['data_root']) / f'09_official_test/04_shuffle/{scope}/mapping.parquet'
    atomic_parquet(destination, [{'graph_id': key, 'donor_graph_id': value} for key, value in sorted(mapping.items())])
    payload = {'status': 'PASS', 'scope': scope, 'graphs': len(mapping), 'sha256': sha256_file(destination)}
    atomic_json(destination.with_suffix('.json'), payload)
    return payload

def _mapping(path: Path):
    return {str(row['graph_id']): str(row['donor_graph_id']) for row in _rows(path)}

def _official_store(cfg: Mapping[str, Any], fold: int | None):
    dataset = cfg['dataset']
    _, _, rows = _combined_rows(cfg)
    scope = _scope(dataset, fold)
    data_root = Path(cfg['paths']['data_root'])
    store = FoldModalStore(rows, fold if dataset == 'bracs' else None, dataset=dataset, residual_root=data_root / f'09_official_test/04_residual/{scope}', donor_map=_mapping(data_root / f'09_official_test/04_shuffle/{scope}/mapping.parquet'), training_splits=('train', 'val', 'validation') if dataset == 'bracs' else ('train',), use_final_fold=dataset == 'bracs')
    return (rows, scope, store)

def run_official_meanpool_all(cfg: Mapping[str, Any], fold: int | None) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    rows, scope, store = _official_store(cfg, fold)
    root = Path(cfg['paths']['data_root']) / f'09_official_test/05_meanpool_cache/{scope}'
    arms = {arm: write_meanpool_cache(store, rows, arm, root / arm) for arm in ARMS}
    payload = {'status': 'PASS', 'scope': scope, 'arms': arms}
    atomic_json(root / 'manifest.json', payload)
    return payload

def _group_rows(rows: Sequence[ModalRow], dataset: str):
    if dataset == 'sicapv2':
        return {row.graph_id: [row] for row in rows}
    field = 'patient_id' if dataset == 'tcga_crc_msi' else 'roi_id'
    output = defaultdict(list)
    for row in rows:
        output[getattr(row, field)].append(row)
    if dataset == 'tcga_crc_msi':
        output = defaultdict(list, {key: value for key, value in output.items() if len(value) >= 10})
    return dict(output)

def _metric(dataset: str, labels: np.ndarray, logits: np.ndarray) -> float:
    if dataset == 'sicapv2':
        return qwk(labels, _softmax(logits))
    if dataset == 'bracs':
        return macro_f1(labels, _softmax(logits))
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, logits[:, 0]))

def _fixed_epochs(cfg: Mapping[str, Any], arm: str, encoder: str, seed: int) -> int:
    root = Path(cfg['paths']['result_root'])
    values = []
    for fold in range(int(cfg['split']['folds'])):
        payload = json.loads((root / f'06_experts/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}/manifest.json').read_text(encoding='utf-8'))
        values.append(int(payload['best_epoch']))
    return max(1, int(round(float(np.median(values)))))

def _train_geometry_model(*, dataset: str, rows: Sequence[ModalRow], store, arm: str, encoder: str, seed: int, destination: Path, device: str, fold: int | None, max_epochs: int, patience: int, fixed_epochs: int | None) -> dict:
    import torch
    import torch.nn.functional as F
    from .models import GroupGeometryClassifier, PatchGeometryClassifier
    torch.manual_seed(seed)
    np.random.seed(seed)
    grouped = dataset in {'tcga_crc_msi', 'bracs'}
    classes = 4 if dataset == 'sicapv2' else 7 if dataset == 'bracs' else 1
    if dataset == 'bracs':
        training_rows = [row for row in rows if row.split in {'train', 'val', 'validation'} and row.final_fold != fold]
        heldout_rows = [row for row in rows if row.split in {'train', 'val', 'validation'} and row.final_fold == fold]
    else:
        training_rows = [row for row in rows if row.split == 'train']
        heldout_rows = []
    test_rows = [row for row in rows if row.split == 'test']
    train_groups, held_groups, test_groups = (_group_rows(value, dataset) for value in (training_rows, heldout_rows, test_rows))
    model = (GroupGeometryClassifier if grouped else PatchGeometryClassifier)(encoder, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    def forward(values):
        samples = [_sample(store, row, arm, encoder) for row in values]
        if not grouped:
            return model(**_pad_graphs(samples, device, encoder == 'meanpool'))
        payload = _pad_graphs(samples, device, encoder == 'meanpool')
        shaped = {key: value[None] for key, value in payload.items()}
        return model(tile_mask=torch.ones((1, len(values)), dtype=torch.bool, device=device), **shaped)[0]

    def predict(groups):
        output = []
        model.eval()
        with torch.inference_mode():
            for sample_id, values in groups.items():
                output.append((str(sample_id), values, np.asarray(forward(values).cpu().numpy()[0])))
        return output
    best = None
    best_metric = -float('inf')
    best_epoch = 0
    stale = 0
    history = []
    epochs = int(fixed_epochs or max_epochs)
    for epoch in range(epochs):
        model.train()
        keys = list(train_groups)
        np.random.default_rng(seed + epoch).shuffle(keys)
        losses = []
        iterator = (train_groups[key] for key in keys) if grouped else ([train_groups[key][0] for key in keys[start:start + 64]] for start in range(0, len(keys), 64))
        for values in iterator:
            logits = forward(values)
            labels = torch.as_tensor([values[0].label_id] if grouped else [row.label_id for row in values], device=device)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], labels.float()) if dataset == 'tcga_crc_msi' else F.cross_entropy(logits, labels.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {'epoch': epoch + 1, 'loss': float(np.mean(losses))}
        if held_groups:
            held = predict(held_groups)
            metric = _metric(dataset, np.asarray([item[1][0].label_id for item in held]), np.stack([item[2] for item in held]))
            record['heldout_metric'] = metric
            if metric > best_metric + 1e-07:
                best_metric, best_epoch, stale = (metric, epoch + 1, 0)
                best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if fixed_epochs is None and held_groups and (stale >= patience):
            break
    if best is None:
        best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        best_epoch = len(history)
    model.load_state_dict(best)
    held = predict(held_groups) if held_groups else []
    test = predict(test_groups)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_torch(destination / 'best.pt', {'model': best, 'best_epoch': best_epoch, 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed})
    rows_out = []
    for role, values in (('heldout', held), ('test', test)):
        for sample_id, samples, logits in values:
            rows_out.append({'sample_id': sample_id, 'group_id': samples[0].wsi_id if dataset == 'bracs' else samples[0].patient_id, 'label_id': samples[0].label_id if role == 'heldout' else None, 'role': role, 'logits': logits.tolist(), 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed})
    atomic_parquet(destination / 'predictions.parquet', rows_out)
    payload = {'status': 'PASS', 'dataset': dataset, 'fold': fold, 'arm': arm, 'encoder': encoder, 'seed': seed, 'best_epoch': best_epoch, 'epochs': len(history), 'history': history, 'heldout_samples': len(held), 'test_samples': len(test)}
    atomic_json(destination / 'manifest.json', payload)
    return payload

def run_official_geometry(cfg: Mapping[str, Any], fold: int | None, arm: str, encoder: str, seed: int, device: str) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    dataset = cfg['dataset']
    if dataset == 'bracs' and fold is None:
        raise ValueError('BRACS official geometry requires a final fold')
    rows, scope, store = _official_store(cfg, fold)
    if encoder == 'meanpool':
        store = MeanPoolStore(Path(cfg['paths']['data_root']) / f'09_official_test/05_meanpool_cache/{scope}/{arm}/manifest.json')
    fixed = None if dataset == 'bracs' else _fixed_epochs(cfg, arm, encoder, seed)
    destination = Path(cfg['paths']['result_root']) / f'09_official_test/geometry/{encoder}/{arm}/{scope}/seed_{seed}'
    return _train_geometry_model(dataset=dataset, rows=rows, store=store, arm=arm, encoder=encoder, seed=seed, destination=destination, device=device, fold=fold, max_epochs=int(cfg['training']['max_epochs']), patience=int(cfg['training']['patience']), fixed_epochs=fixed)

def _bracs_safe_anchor_loader(manifest_path: str | Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    probe = importlib.import_module('probe')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    columns = ['graph_id', 'anchor_id', 'roi_id', 'wsi_id', 'split_new', 'validation_fold', 'final_validation_fold', 'rays36', 'raw_geometry9', 'valid_ncr3d']
    table = pa.concat_tables([pq.read_table(item['path'], columns=columns, partitioning=None) for item in manifest['shards']], promote_options='default')

    def strings(name):
        return np.asarray(table[name].combine_chunks().to_pylist(), object)

    def numeric(name, dtype):
        return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype)

    def fixed(name, width):
        values = table[name].combine_chunks()
        return np.asarray(values.values.to_numpy(zero_copy_only=False), np.float32).reshape(len(values), width)
    graph = table['graph_id'].combine_chunks()
    encoded = graph.dictionary_encode()

    def folds(name):
        return np.asarray([-1 if value is None else int(value) for value in table[name].combine_chunks().to_pylist()], np.int8)
    return probe.ProbeArrays(graph_id=np.asarray(graph.to_pylist(), object), graph_code=np.asarray(encoded.indices.to_numpy(zero_copy_only=False), np.int32), anchor_id=numeric('anchor_id', np.int64), roi_id=strings('roi_id'), wsi_id=strings('wsi_id'), split=strings('split_new'), label7=np.full(len(table), -1, np.int8), validation_fold=folds('validation_fold'), final_validation_fold=folds('final_validation_fold'), rays=fixed('rays36', 36), raw_geometry=fixed('raw_geometry9', 9), valid_ncr=numeric('valid_ncr3d', bool))

def _sanitized_bracs_tile_manifest(source: Path, development_index: Path, destination: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    sidecar = destination.with_suffix('.json')
    if destination.is_file() and _passed(sidecar):
        payload = json.loads(sidecar.read_text(encoding='utf-8'))
        if payload.get('schema_version') == 2:
            return destination
    schema = pq.read_schema(source)
    base_columns = [name for name in schema.names if name not in {'label_7', 'label_3'}]
    base = pq.read_table(source, columns=base_columns, partitioning=None)
    development = read_index(development_index)
    labels = {row.graph_id: int(row.label_id) for row in development if row.label_id is not None}
    if len(labels) != len(development):
        raise RuntimeError('BRACS development label cache contains duplicate or missing graph labels')
    graph_ids = [str(value) for value in base['graph_id'].combine_chunks().to_pylist()]
    split = [str(value).lower() for value in base['split_new'].combine_chunks().to_pylist()]
    label7 = [labels.get(graph, -1) for graph in graph_ids]
    missing_development = [graph for graph, role, label in zip(graph_ids, split, label7) if role in {'train', 'val', 'validation'} and graph in labels and (label < 0)]
    if missing_development:
        raise RuntimeError(f'BRACS sanitized manifest lost development labels: {missing_development[:10]}')
    label3 = [-1 if label < 0 else 0 if label <= 2 else 1 if label <= 4 else 2 for label in label7]
    if any((role == 'test' and label >= 0 for role, label in zip(split, label7))):
        raise RuntimeError('BRACS sanitized manifest exposed an official TEST label')
    table = base.append_column('label_7', pa.array(label7, type=pa.int8())).append_column('label_3', pa.array(label3, type=pa.int8()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    pq.write_table(table, temporary, compression='zstd')
    temporary.replace(destination)
    atomic_json(sidecar, {'status': 'PASS', 'schema_version': 2, 'source': str(source), 'source_sha256': sha256_file(source), 'development_index': str(development_index), 'development_index_sha256': sha256_file(development_index), 'rows': len(graph_ids), 'development_labels': len(labels), 'hidden_test_rows': sum((role == 'test' for role in split))})
    return destination

def run_official_bracs_b0(cfg: Mapping[str, Any], fold: int, encoder: str, seed: int, device: str) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    if cfg['dataset'] != 'bracs':
        raise ValueError('official BRACS B0 wrapper is BRACS-only')
    old_code = Path(cfg['paths'].get('breast_roi_baseline_code_root', _resource_path('artifact_0053')))
    if str(old_code) not in sys.path:
        sys.path.insert(0, str(old_code))
    training = importlib.import_module('training')
    training.load_anchor_arrays = _bracs_safe_anchor_loader
    old_data = Path(cfg['paths']['model_input_root'])
    data_root = Path(cfg['paths']['data_root'])
    sanitized = _sanitized_bracs_tile_manifest(old_data / '00_manifest/tile_manifest.parquet', data_root / '03_modal_features/index.parquet', data_root / '09_official_test/00_manifest/bracs_tile_manifest_labels_hidden.parquet')
    destination = Path(cfg['paths']['result_root']) / f'09_official_test/b0/bracs/{encoder}/fold_{fold:02d}/seed_{seed}'
    return training.train_expert(anchor_manifest=old_data / '04_direct3d/anchor_cache_manifest.json', tile_manifest=sanitized, residual_path=None, shuffle_map_path=None, expert='E0_RGB_MASK2D', task='t7', encoder=encoder, phase='final_test', fold=fold, seed=seed, output_dir=destination, device=device, tile_budget=256, object_budget=32000, max_epochs=int(cfg['training']['max_epochs']), patience=int(cfg['training']['patience']))

def _official_geometry_predictions(cfg, encoder: str, arm: str):
    dataset = cfg['dataset']
    root = Path(cfg['paths']['result_root'])
    seeds = _seed_set(encoder)
    grouped = defaultdict(list)
    if dataset == 'bracs':
        for fold in range(5):
            for seed in seeds:
                path = root / f'09_official_test/geometry/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}/predictions.parquet'
                for source in _rows(path):
                    row = dict(source, _source_fold=fold, _source_seed=seed)
                    role = 'heldout' if row['role'] == 'heldout' else 'external'
                    grouped[role, fold if role == 'heldout' else -1, str(row['sample_id'])].append(row)
    else:
        for seed in seeds:
            path = root / f'09_official_test/geometry/{encoder}/{arm}/final/seed_{seed}/predictions.parquet'
            for source in _rows(path):
                grouped['test', -1, str(source['sample_id'])].append(dict(source, _source_fold=-1, _source_seed=seed))
    output = {}
    for key, values in grouped.items():
        observed = {(int(row['_source_fold']), int(row['_source_seed'])) for row in values}
        expected = {(fold, seed) for fold in range(5) for seed in seeds} if dataset == 'bracs' and key[0] == 'external' else {(key[1] if dataset == 'bracs' else -1, seed) for seed in seeds}
        if observed != expected or len(values) != len(expected):
            raise RuntimeError(f'official geometry ensemble coverage mismatch: {dataset}/{encoder}/{arm}/{key}')
        logits = np.stack([np.asarray(row['logits'], float) for row in values])
        if dataset == 'bracs':
            probability = np.mean(_softmax(logits), 0)
            output[key] = {'logits': np.log(np.clip(probability, 1e-07, 1)), 'group': str(values[0]['group_id']), 'label': values[0].get('label_id')}
        else:
            output[key] = {'logits': logits.mean(0), 'group': str(values[0]['group_id']), 'label': None}
    return output

def _official_baseline_predictions(cfg, encoder: str, eligible_test: set[str]):
    dataset = cfg['dataset']
    seeds = _seed_set(encoder)
    root = Path(cfg['paths']['baseline_predictions'])
    if dataset == 'sicapv2':
        grouped = defaultdict(list)
        for seed in seeds:
            path = root / f'official_test/paper_baseline/seed_{seed}/official_test_predictions.parquet'
            for source in _projected_rows(path, ('graph_id', 'patient_id', 'logits')):
                if str(source['graph_id']) in eligible_test:
                    grouped[str(source['graph_id'])].append(dict(source, _source_seed=seed))
        output = {}
        for key, values in grouped.items():
            if {row['_source_seed'] for row in values} != set(seeds) or len(values) != len(seeds):
                raise RuntimeError(f'official SICAP baseline seed coverage mismatch: {key}')
            output[key] = {'logits': np.mean([np.asarray(row['logits'], float) for row in values], 0), 'group': str(values[0]['patient_id'])}
        return output
    if dataset == 'tcga_crc_msi':
        tiles = defaultdict(list)
        for seed in seeds:
            path = root / f'official_test/paper_baseline/seed_{seed}/official_test_predictions.parquet'
            for source in _projected_rows(path, ('graph_id', 'patient_id', 'probabilities')):
                if str(source['graph_id']) in eligible_test:
                    tiles[str(source['graph_id'])].append(dict(source, _source_seed=seed))
        if set(tiles) != eligible_test:
            raise RuntimeError(f'official CRC baseline tile coverage mismatch: missing={sorted(eligible_test - set(tiles))[:10]}, extra={sorted(set(tiles) - eligible_test)[:10]}')
        patients = defaultdict(list)
        for graph, values in tiles.items():
            if {row['_source_seed'] for row in values} != set(seeds) or len(values) != len(seeds):
                raise RuntimeError(f'official CRC baseline seed coverage mismatch: {graph}')
            patients[str(values[0]['patient_id'])].append(float(np.mean([row['probabilities'][1] for row in values])))
        output = {}
        for patient, values in patients.items():
            if len(values) >= 10:
                positive = sum((value >= 0.5 for value in values))
                output[patient] = {'logits': float(np.log((positive + 0.5) / (len(values) - positive + 0.5))), 'group': patient}
        return output
    grouped = defaultdict(list)
    result = Path(cfg['paths']['result_root'])
    for fold in range(5):
        for seed in seeds:
            path = result / f'09_official_test/b0/bracs/{encoder}/fold_{fold:02d}/seed_{seed}/predictions.parquet'
            for source in _rows(path):
                row = dict(source, _source_fold=fold, _source_seed=seed)
                grouped[row['role'], fold if row['role'] == 'heldout' else -1, str(row['roi_id'])].append(row)
    output = {}
    for key, values in grouped.items():
        observed = {(row['_source_fold'], row['_source_seed']) for row in values}
        expected = {(fold, seed) for fold in range(5) for seed in seeds} if key[0] == 'external' else {(key[1], seed) for seed in seeds}
        if observed != expected or len(values) != len(expected):
            raise RuntimeError(f'official BRACS baseline ensemble coverage mismatch: {key}')
        probability = np.mean([np.asarray(row['probability'], float) for row in values], 0)
        output[key] = {'logits': np.log(np.clip(probability, 1e-07, 1)), 'group': str(values[0]['wsi_id']), 'label': int(values[0]['label']) if key[0] == 'heldout' else None}
    return output

def _development_calibration(cfg, encoder: str, arm: str):
    dataset = cfg['dataset']
    baseline_root = Path(cfg['paths']['baseline_predictions'])
    modal = read_index(Path(cfg['paths']['data_root']) / '03_modal_features/index.parquet')
    if dataset == 'sicapv2':
        baseline, folds = (load_sicap_baseline(baseline_root, encoder), 4)
    elif dataset == 'tcga_crc_msi':
        baseline, folds = (load_crc_baseline(baseline_root, encoder, {row.graph_id for row in modal}), 5)
    else:
        raise ValueError('BRACS uses final-fold calibration')
    geometry = load_geometry(Path(cfg['paths']['result_root']), encoder, arm, folds)
    keys = sorted(baseline)
    labels = np.asarray([baseline[key]['label'] for key in keys])
    if dataset == 'tcga_crc_msi':
        return fit_crc(np.asarray([baseline[key]['logit'] for key in keys]), np.asarray([geometry[key]['logits'][0] for key in keys]), labels)
    return fit_multiclass_alpha(np.stack([baseline[key]['logits'] for key in keys]), np.stack([geometry[key]['logits'] for key in keys]), labels)

def _expected_test_sample_ids(dataset: str, rows: Sequence[ModalRow]) -> set[str]:
    if dataset == 'sicapv2':
        return {row.graph_id for row in rows}
    grouped = defaultdict(int)
    field = 'patient_id' if dataset == 'tcga_crc_msi' else 'roi_id'
    for row in rows:
        grouped[getattr(row, field)] += 1
    if dataset == 'tcga_crc_msi':
        return {key for key, count in grouped.items() if count >= 10}
    return set(grouped)

def run_official_fusion(cfg: Mapping[str, Any], encoder: str) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    dataset = cfg['dataset']
    _, test, _ = _combined_rows(cfg)
    eligible = {row.graph_id for row in test}
    baseline = _official_baseline_predictions(cfg, encoder, eligible)
    geometry = {arm: _official_geometry_predictions(cfg, encoder, arm) for arm in ARMS}
    test_keys = sorted((key for key in baseline if key[0] == 'external')) if dataset == 'bracs' else sorted(baseline)
    actual_ids = {key[2] for key in test_keys} if dataset == 'bracs' else set(test_keys)
    expected_ids = _expected_test_sample_ids(dataset, test)
    if actual_ids != expected_ids:
        raise RuntimeError(f'official baseline TEST coverage mismatch: missing={sorted(expected_ids - actual_ids)[:10]}, extra={sorted(actual_ids - expected_ids)[:10]}')
    rows = []
    for key in test_keys:
        sample_id = key[2] if dataset == 'bracs' else key
        b = baseline[key]['logits']
        p = float(1 / (1 + np.exp(-b))) if dataset == 'tcga_crc_msi' else _softmax(np.asarray(b)[None])[0]
        rows.append({'sample_id': sample_id, 'group_id': baseline[key]['group'], 'arm': 'B0', 'score': float(p) if np.ndim(p) == 0 else None, 'probabilities': None if np.ndim(p) == 0 else np.asarray(p).tolist(), 'calibration': None})
    calibrations = {}
    for arm in ARMS:
        if dataset == 'bracs':
            fit_keys = sorted((key for key in baseline if key[0] == 'heldout'))
            b_fit = np.stack([baseline[key]['logits'] for key in fit_keys])
            g_fit = np.stack([geometry[arm][key]['logits'] for key in fit_keys])
            labels = np.asarray([baseline[key]['label'] for key in fit_keys])
            calibration = fit_multiclass_alpha(b_fit, g_fit, labels)
        else:
            calibration = _development_calibration(cfg, encoder, arm)
        calibrations[arm] = calibration.__dict__
        for key in test_keys:
            geometry_key = key if dataset == 'bracs' else ('test', -1, key)
            if geometry_key not in geometry[arm]:
                raise RuntimeError(f'official baseline/geometry join mismatch: {dataset}/{encoder}/{arm}/{key}')
            fused = apply(np.asarray(baseline[key]['logits']), np.asarray(geometry[arm][geometry_key]['logits']), calibration, dataset != 'tcga_crc_msi')
            probability = float(1 / (1 + np.exp(-fused))) if dataset == 'tcga_crc_msi' else _softmax(np.asarray(fused)[None])[0]
            sample_id = key[2] if dataset == 'bracs' else key
            rows.append({'sample_id': sample_id, 'group_id': baseline[key]['group'], 'arm': arm, 'score': float(probability) if np.ndim(probability) == 0 else None, 'probabilities': None if np.ndim(probability) == 0 else np.asarray(probability).tolist(), 'calibration': calibration.__dict__})
    destination = Path(cfg['paths']['result_root']) / f'09_official_test/predictions/{encoder}.parquet'
    atomic_parquet(destination, rows)
    payload = {'status': 'PASS', 'dataset': dataset, 'encoder': encoder, 'samples': len(test_keys), 'rows': len(rows), 'calibrations': calibrations, 'labels_present': False}
    atomic_json(destination.with_suffix('.json'), payload)
    return payload

def _prediction_inventory(prediction_paths: Mapping[str, Path]) -> tuple[dict[str, dict[str, dict]], list[str]]:
    inventories: dict[str, dict[str, dict]] = {}
    reference: list[str] | None = None
    for encoder, path in prediction_paths.items():
        grouped: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in _rows(path):
            sample_id, arm = (str(row['sample_id']), str(row['arm']))
            if arm in grouped[sample_id]:
                raise RuntimeError(f'duplicate official prediction: {encoder}/{sample_id}/{arm}')
            grouped[sample_id][arm] = row
        keys = sorted(grouped)
        if not keys or any((set(grouped[key]) != {'B0', *ARMS} for key in keys)):
            raise RuntimeError(f'official prediction arm coverage mismatch for {encoder}')
        if reference is None:
            reference = keys
        elif keys != reference:
            raise RuntimeError('MeanPool/DeepSets official prediction sample sets differ')
        inventories[encoder] = dict(grouped)
    return (inventories, list(reference or []))

def _label_rows_from_source(cfg: Mapping[str, Any], expected_ids: set[str]) -> list[dict[str, Any]]:
    dataset = cfg['dataset']
    source_rows = [row for row in all_dataset_rows(cfg, include_labels=True) if row.split == 'test']
    labels: dict[str, tuple[int | None, str]] = {}
    for row in source_rows:
        if dataset == 'sicapv2':
            sample_id, group_id = (row.graph_id, row.patient_id)
        elif dataset == 'tcga_crc_msi':
            sample_id, group_id = (row.patient_id, row.patient_id)
        else:
            sample_id, group_id = (row.roi_id, row.wsi_id)
        if sample_id not in expected_ids:
            continue
        value = (row.label_id, group_id)
        previous = labels.setdefault(sample_id, value)
        if previous != value:
            raise RuntimeError(f'official TEST label/group mismatch within sample {sample_id}')
    if set(labels) != expected_ids:
        missing = sorted(expected_ids - set(labels))[:10]
        extra = sorted(set(labels) - expected_ids)[:10]
        raise RuntimeError(f'official prediction/label join mismatch: missing={missing}, extra={extra}')
    if any((value[0] is None for value in labels.values())):
        raise RuntimeError('official TEST source contains missing labels')
    return [{'sample_id': key, 'label_id': int(labels[key][0]), 'group_id': labels[key][1]} for key in sorted(labels)]

def _load_or_create_test_labels(cfg: Mapping[str, Any], expected_ids: Sequence[str]) -> tuple[dict[str, tuple[int, str]], Path, Path]:
    result_root = Path(cfg['paths']['result_root'])
    transaction_root = result_root / '09_official_test'
    labels_path = transaction_root / 'test_labels.parquet'
    marker_path = transaction_root / 'labels_read_once.json'
    expected = set(map(str, expected_ids))
    if marker_path.is_file() and (not labels_path.is_file()):
        raise RuntimeError('TEST label marker exists but compact label artifact is missing')
    created_from_source = not labels_path.is_file()
    if created_from_source:
        rows = _label_rows_from_source(cfg, expected)
        atomic_parquet(labels_path, rows)
    rows = _rows(labels_path)
    labels = {str(row['sample_id']): (int(row['label_id']), str(row['group_id'])) for row in rows}
    if len(labels) != len(rows) or set(labels) != expected:
        raise RuntimeError('compact TEST label artifact does not match frozen predictions')
    manifest_hashes = {str(path): sha256_file(path) for path in cfg['paths']['manifest_sources'] if Path(path).is_file()}
    marker = {'status': 'PASS', 'time': time.time(), 'dataset': cfg['dataset'], 'labels_artifact': str(labels_path), 'labels_sha256': sha256_file(labels_path), 'samples': len(labels), 'manifest_hashes': manifest_hashes, 'source_labels_read': created_from_source}
    if not marker_path.is_file():
        atomic_json(marker_path, marker)
    else:
        existing = json.loads(marker_path.read_text(encoding='utf-8'))
        if existing.get('labels_sha256') != marker['labels_sha256'] or existing.get('dataset') != cfg['dataset']:
            raise RuntimeError('compact TEST label artifact no longer matches its read-once marker')
        if manifest_hashes and existing.get('manifest_hashes') != manifest_hashes:
            raise RuntimeError('official TEST source manifests changed after the label transaction')
    return (labels, labels_path, marker_path)

def run_official_evaluate(cfg: Mapping[str, Any]) -> dict:
    require_frozen(cfg['paths']['global_result_root'])
    dataset = cfg['dataset']
    result_root = Path(cfg['paths']['result_root'])
    prediction_paths = {encoder: result_root / f'09_official_test/predictions/{encoder}.parquet' for encoder in ('meanpool', 'deepsets')}
    if not all((_passed(path.with_suffix('.json')) and path.is_file() for path in prediction_paths.values())):
        raise RuntimeError('all registered official predictions must exist before TEST labels are read')
    inventories, keys = _prediction_inventory(prediction_paths)
    labels_by_id, labels_path, label_marker = _load_or_create_test_labels(cfg, keys)
    summaries = {}
    for encoder, grouped in inventories.items():
        labels = np.asarray([labels_by_id[key][0] for key in keys])
        groups = np.asarray([labels_by_id[key][1] for key in keys], object)
        scores = {}
        for arm in ('B0', *ARMS):
            scores[arm] = np.asarray([grouped[key][arm]['score'] for key in keys]) if dataset == 'tcga_crc_msi' else np.stack([grouped[key][arm]['probabilities'] for key in keys])
        replicates = int(cfg['runtime']['bootstrap_replicates'])
        if dataset == 'sicapv2':
            observed, distributions = paired_cluster_bootstrap(labels=labels, groups=groups, scores=scores, metric=qwk, replicates=replicates)
        elif dataset == 'bracs':
            observed, distributions = paired_cluster_bootstrap(labels=labels, groups=groups, scores=scores, metric=macro_f1, replicates=replicates)
        else:
            from sklearn.metrics import roc_auc_score
            metric = lambda y, s: float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')
            observed, distributions = paired_cluster_bootstrap(labels=labels, groups=groups, scores=scores, metric=metric, replicates=replicates)
        interpretation = registered_interpretation(observed, distributions)
        output = result_root / f'09_official_test/metrics/{encoder}'
        atomic_npz(output / 'bootstrap_replicates.npz', **distributions)
        payload = {'status': 'PASS', 'dataset': dataset, 'encoder': encoder, 'samples': len(keys), 'observed': observed, **interpretation}
        if dataset == 'bracs':
            labels_t3 = np.where(labels <= 2, 0, np.where(labels <= 4, 1, 2))
            payload['descriptive_t3_macro_f1'] = {arm: macro_f1(labels_t3, t7_to_t3(value)) for arm, value in scores.items()}
        atomic_json(output / 'summary.json', payload)
        summaries[encoder] = payload
    final = {'status': 'PASS', 'dataset': dataset, 'labels_artifact': str(labels_path), 'labels_read_marker': str(label_marker), 'encoders': summaries}
    atomic_json(result_root / '09_official_test/summary.json', final)
    return final
