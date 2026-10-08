from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import math
import os
import random
import socket
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Sequence
import numpy as np
from celllift.runtime import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.common.token_store import FoldTokenStore, GraphTokenSample, collate_variable_tokens
from celllift.set_encoding.experiment import ExperimentArm, SCREENING_PROTOCOL_ID
from celllift.set_encoding.models import CRCFusionMIL, DualSetFusion, SICAPClassifier
from celllift.set_encoding.tasks import crc_patient_loss, crc_patient_metrics, inverse_frequency_class_weights, sicap_loss, sicap_metrics, validation_threshold

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)

def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _progress(output: Path, event: str, **values: Any) -> None:
    output.mkdir(parents=True, exist_ok=True)
    row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event, **values}
    line = json.dumps(row, sort_keys=True)
    with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
        stream.flush()
    print(line, flush=True)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _code_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        dirty = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet'], timeout=10, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
        return {'git_commit': commit, 'git_dirty': dirty}
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': None, 'git_dirty': None}

def _gpu_identity(device: torch.device) -> dict[str, Any] | None:
    if device.type != 'cuda':
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    physical = visible.split(',')[0].strip() if visible else str(torch.cuda.current_device())
    uuid = None
    try:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=10).splitlines()
        lookup = {line.split(',', 1)[0].strip(): line.split(',', 1)[1].strip() for line in lines}
        uuid = lookup.get(physical)
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {'visible_index': torch.cuda.current_device(), 'physical_index': physical, 'uuid': uuid, 'name': properties.name, 'total_memory_bytes': int(properties.total_memory)}

def _runtime_start(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    return {'host': socket.gethostname(), 'gpu': _gpu_identity(device), 'started_at': datetime.now(timezone.utc).isoformat(), 'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest(), 'code_state': _code_state()}

def _runtime_finish(runtime: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(runtime)
    result['ended_at'] = datetime.now(timezone.utc).isoformat()
    result['peak_allocated_bytes'] = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
    result['peak_reserved_bytes'] = int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0
    return result

def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (isinstance(error, RuntimeError) and 'out of memory' in str(error).lower())

def _persist_shuffle_mapping(store: FoldTokenStore, data_root: Path, *, phase: str, fold: int, seed: int) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = data_root / '00_manifest' / 'controls' / phase / f'fold_{fold:02d}' / f'seed_{seed}'
    destination, manifest_path = (root / 'anchor_shuffle.parquet', root / 'manifest.json')
    if manifest_path.is_file() and destination.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS' and previous.get('seed') == seed and (previous.get('fold') == fold) and (previous.get('phase') == phase) and (previous.get('sha256') == _sha256(destination)):
            return previous
        raise RuntimeError(f'existing shuffle mapping fails provenance validation: {manifest_path}')
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    writer = None
    rows = store.shuffle_rows
    try:
        for start in range(0, len(rows), 100000):
            table = pa.Table.from_pylist([row.as_dict() for row in rows[start:start + 100000]])
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError('shuffle mapping is empty')
    os.replace(temporary, destination)
    payload = {'status': 'PASS', 'phase': phase, 'fold': fold, 'seed': seed, 'rows': len(rows), 'path': str(destination), 'sha256': _sha256(destination), 'constraints': 'cross-graph,same-partition,same-count-decile,prefer-exact-count'}
    _atomic_json(manifest_path, payload)
    return payload

def _seed_everything(seed: int) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def _epoch_seed(seed: int, epoch: int) -> int:
    return int((int(seed) * 1000003 + int(epoch) * 97409 + 17) % (2 ** 31 - 1))

def _read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path, partitioning=None).to_pylist()

def _load_rgb(path: Path) -> dict[str, np.ndarray]:
    rows = _read_rows(path)
    result: dict[str, np.ndarray] = {}
    for row in rows:
        graph_id = str(row['graph_id'])
        if graph_id in result:
            raise RuntimeError(f'duplicate RGB graph ID {graph_id!r}')
        if 'rgb_feature' in row:
            feature = np.asarray(row['rgb_feature'], np.float32)
        elif 'features' in row:
            feature = np.asarray(row['features'], np.float32)
        else:
            feature = np.asarray([row[f'f{index:03d}'] for index in range(512)], np.float32)
        if feature.shape != (512,) or not np.isfinite(feature).all():
            raise RuntimeError(f'invalid RGB feature for {graph_id!r}')
        result[graph_id] = feature
    return result

def _tensor(value: np.ndarray | None, device: torch.device, *, boolean: bool=False) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.bool if boolean else torch.float32, device=device)

def _fusion_inputs(batch: dict[str, Any], rgb: dict[str, np.ndarray], arm: ExperimentArm, device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if arm.use_rgb:
        missing = [graph for graph in batch['graph_ids'] if graph not in rgb]
        if missing:
            raise KeyError(f'RGB cache misses graph IDs: {missing[:3]}')
        result['rgb_features'] = torch.as_tensor(np.stack([rgb[graph] for graph in batch['graph_ids']]), dtype=torch.float32, device=device)
    for name in ('nucleus_tokens', 'cell_tokens'):
        value = _tensor(batch[name], device)
        if value is not None:
            result[name] = value
    for name in ('nucleus_mask', 'cell_mask'):
        value = _tensor(batch[name], device, boolean=True)
        if value is not None:
            result[name] = value
    return result

def _token_batches(samples: Sequence[GraphTokenSample], token_budget: int, *, shuffle: bool, seed: int) -> Iterable[list[GraphTokenSample]]:

    def sample_cost(sample: GraphTokenSample) -> int:
        return max(1, 0 if sample.nucleus_tokens is None else len(sample.nucleus_tokens), 0 if sample.cell_tokens is None else len(sample.cell_tokens))
    ordered = sorted(samples, key=lambda sample: (sample_cost(sample), sample.graph_id))
    batches: list[list[GraphTokenSample]] = []
    current: list[GraphTokenSample] = []
    current_max = 0
    for sample in ordered:
        size = sample_cost(sample)
        if size > token_budget:
            raise RuntimeError(f'graph {sample.graph_id} exceeds token budget {token_budget}')
        next_max = max(current_max, size)
        if current and (len(current) + 1) * next_max > token_budget:
            batches.append(current)
            current, current_max = ([], 0)
        current.append(sample)
        current_max = max(current_max, size)
    if current:
        batches.append(current)
    if shuffle:
        np.random.default_rng(seed).shuffle(batches)
    yield from batches

def _build_models(dataset: str, arm: ExperimentArm, encoder: str, dropout: float) -> nn.Module:
    effective_encoder = 'meanpool' if arm.object_mode == 'rgb' else encoder
    fusion = DualSetFusion(set_encoder=effective_encoder, object_mode=arm.object_mode, use_rgb=arm.use_rgb, dropout=dropout)
    if dataset == 'sicapv2':
        return SICAPClassifier(fusion)
    return CRCFusionMIL(fusion, dropout=dropout)

def _autocast(device: torch.device):
    return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda')

def _grad_scaler(device: torch.device):
    try:
        return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

def _train_sicap(model: SICAPClassifier, train_samples: list[GraphTokenSample], val_samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, cfg: dict[str, Any], device: torch.device, seed: int, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = cfg['training']
    labels = np.asarray([int(sample.metadata['label_id']) for sample in train_samples])
    class_weights = inverse_frequency_class_weights(labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']), weight_decay=float(training['weight_decay']))
    scaler = _grad_scaler(device)
    token_budget = int(getattr(model, 'token_budget', 32000))
    best_metric = -math.inf
    best_epoch = -1
    stale = 0
    start_epoch = 0
    last_path = output / 'last.pt'
    if last_path.is_file():
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        scaler.load_state_dict(resumed['scaler'])
        start_epoch = int(resumed['next_epoch'])
        best_metric = float(resumed['best_metric'])
        best_epoch = int(resumed['best_epoch'])
        stale = int(resumed['stale'])

    def evaluate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        model.eval()
        rows: list[dict[str, Any]] = []
        probabilities, truth = ([], [])
        with torch.no_grad():
            for group in _token_batches(val_samples, token_budget, shuffle=False, seed=seed):
                batch = collate_variable_tokens(group)
                inputs = _fusion_inputs(batch, rgb, arm, device)
                with _autocast(device):
                    outputs = model(**inputs)
                probs = torch.softmax(outputs['grade_logits'].float(), dim=1).cpu().numpy()
                crib = torch.sigmoid(outputs['cribriform_logits'].float()).cpu().numpy()
                probabilities.append(probs)
                truth.extend((int(sample.metadata['label_id']) for sample in group))
                for sample, values, crib_score in zip(group, probs, crib):
                    rows.append({'sample_id': sample.graph_id, 'graph_id': sample.graph_id, 'patient_id': str(sample.metadata['patient_id']), 'y_true': int(sample.metadata['label_id']), 'prob_0': float(values[0]), 'prob_1': float(values[1]), 'prob_2': float(values[2]), 'prob_3': float(values[3]), 'cribriform_score': float(crib_score), 'g4c_valid': bool(sample.metadata.get('g4c_valid', False)), 'g4c_label': int(sample.metadata['g4c_label']) if sample.metadata.get('g4c_valid', False) else None})
        metrics = sicap_metrics(np.asarray(truth), np.concatenate(probabilities))
        return (metrics, rows)
    epochs_ran = start_epoch
    for epoch in range(start_epoch, int(training['max_epochs'])):
        _seed_everything(_epoch_seed(seed, epoch))
        model.train()
        for group in _token_batches(train_samples, token_budget, shuffle=True, seed=seed + epoch):
            batch = collate_variable_tokens(group)
            inputs = _fusion_inputs(batch, rgb, arm, device)
            grade = torch.as_tensor([int(sample.metadata['label_id']) for sample in group], device=device)
            crib_values = [sample.metadata.get('g4c_label') for sample in group]
            crib_target = torch.as_tensor([float(value) if value is not None else float('nan') for value in crib_values], device=device)
            crib_valid = torch.as_tensor([bool(sample.metadata.get('g4c_valid', False)) for sample in group], device=device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                outputs = model(**inputs)
                losses = sicap_loss(outputs, grade, class_weights=class_weights, cribriform_targets=crib_target, cribriform_valid=crib_valid)
            scaler.scale(losses['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics, _ = evaluate()
        score = float(metrics['qwk'])
        if score > best_metric:
            best_metric, best_epoch, stale = (score, epoch, 0)
            _atomic_torch_save(output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics})
        else:
            stale += 1
        epochs_ran = epoch + 1
        _progress(output, 'epoch', epoch=epoch, train_loss=None, validation_metric=score, best_metric=best_metric, best_epoch=best_epoch, stale_epochs=stale, token_budget=token_budget)
        _atomic_torch_save(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'next_epoch': epochs_ran, 'best_metric': best_metric, 'best_epoch': best_epoch, 'stale': stale})
        if stale >= int(training['patience']):
            break
    payload = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(payload['model'])
    metrics, rows = evaluate()
    metrics.update({'best_epoch': best_epoch, 'epochs_ran': epochs_ran})
    return (metrics, rows)

def _patient_groups(samples: Sequence[GraphTokenSample]) -> dict[str, list[GraphTokenSample]]:
    groups: dict[str, list[GraphTokenSample]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.metadata['patient_id'])].append(sample)
    return {patient: sorted(values, key=lambda sample: sample.graph_id) for patient, values in sorted(groups.items())}

def _crc_patient_batch(samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    flat = collate_variable_tokens(samples)
    base = _fusion_inputs(flat, rgb, arm, device)
    result: dict[str, torch.Tensor] = {'tile_mask': torch.ones((1, len(samples)), dtype=torch.bool, device=device)}
    for name, value in base.items():
        result[name] = value.unsqueeze(0)
    labels = {int(sample.metadata['label_id']) for sample in samples}
    if len(labels) != 1:
        raise RuntimeError('CRC patient tiles disagree on inherited label')
    return (result, torch.tensor([float(next(iter(labels)))], device=device))

def _crc_patient_forward(model: CRCFusionMIL, samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, device: torch.device, token_budget: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    for group in _token_batches(samples, token_budget, shuffle=False, seed=0):
        batch = collate_variable_tokens(group)
        embeddings.append(model.fusion(**_fusion_inputs(batch, rgb, arm, device)))
    tile_embeddings = torch.cat(embeddings, dim=0).unsqueeze(0)
    mask = torch.ones((1, tile_embeddings.shape[1]), dtype=torch.bool, device=device)
    output = model.mil(tile_embeddings, mask)
    output['tile_embeddings'] = tile_embeddings
    labels = {int(sample.metadata['label_id']) for sample in samples}
    if len(labels) != 1:
        raise RuntimeError('CRC patient tiles disagree on inherited label')
    target = torch.tensor([float(next(iter(labels)))], device=device)
    return (output, target)

def _train_crc(model: CRCFusionMIL, train_samples: list[GraphTokenSample], val_samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, cfg: dict[str, Any], device: torch.device, seed: int, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training['learning_rate']), weight_decay=float(training['weight_decay']))
    scaler = _grad_scaler(device)
    train_groups, val_groups = (_patient_groups(train_samples), _patient_groups(val_samples))
    token_budget = int(getattr(model, 'token_budget', 32000))
    best_metric, best_epoch, stale = (-math.inf, -1, 0)
    start_epoch = 0
    last_path = output / 'last.pt'
    if last_path.is_file():
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed['model'], strict=True)
        optimizer.load_state_dict(resumed['optimizer'])
        scaler.load_state_dict(resumed['scaler'])
        start_epoch = int(resumed['next_epoch'])
        best_metric = float(resumed['best_metric'])
        best_epoch = int(resumed['best_epoch'])
        stale = int(resumed['stale'])

    def evaluate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        model.eval()
        rows, truth, scores = ([], [], [])
        with torch.no_grad():
            for patient, samples in val_groups.items():
                with _autocast(device):
                    output_values, target = _crc_patient_forward(model, samples, rgb, arm, device, token_budget)
                score = float(torch.sigmoid(output_values['logits'].float())[0].cpu())
                label = int(target.item())
                truth.append(label)
                scores.append(score)
                rows.append({'sample_id': patient, 'patient_id': patient, 'y_true': label, 'score': score})
        threshold = validation_threshold(truth, scores)
        for row in rows:
            row['validation_threshold'] = threshold
        return (crc_patient_metrics(truth, scores, threshold=threshold), rows)
    epochs_ran = start_epoch
    for epoch in range(start_epoch, int(training['max_epochs'])):
        _seed_everything(_epoch_seed(seed, epoch))
        model.train()
        patients = list(train_groups)
        np.random.default_rng(seed + epoch).shuffle(patients)
        for patient in patients:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output_values, target = _crc_patient_forward(model, train_groups[patient], rgb, arm, device, token_budget)
                loss = crc_patient_loss(output_values['logits'], target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics, _ = evaluate()
        score = float(metrics['auroc'])
        if score > best_metric:
            best_metric, best_epoch, stale = (score, epoch, 0)
            _atomic_torch_save(output / 'best.pt', {'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics})
        else:
            stale += 1
        epochs_ran = epoch + 1
        _progress(output, 'epoch', epoch=epoch, validation_metric=score, best_metric=best_metric, best_epoch=best_epoch, stale_epochs=stale, token_budget=token_budget)
        _atomic_torch_save(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'next_epoch': epochs_ran, 'best_metric': best_metric, 'best_epoch': best_epoch, 'stale': stale})
        if stale >= int(training['patience']):
            break
    payload = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(payload['model'])
    metrics, rows = evaluate()
    metrics.update({'best_epoch': best_epoch, 'epochs_ran': epochs_ran})
    return (metrics, rows)

def _run_job(cfg: dict[str, Any], store: FoldTokenStore, rgb: dict[str, np.ndarray], arm: ExperimentArm, encoder: str, seed: int, fold: int, output: Path, device: torch.device, shuffle_mapping: dict[str, Any] | None=None) -> dict[str, Any]:
    manifest_path = output / 'job.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('protocol_id') != SCREENING_PROTOCOL_ID:
            raise RuntimeError(f'existing job is from an incompatible protocol; archive it before rerun: {manifest_path}')
        if previous.get('status') == 'PASS':
            return previous
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    runtime = _runtime_start(cfg, device)
    _progress(output, 'job_start', dataset=cfg['dataset'], encoder=encoder, effective_encoder='meanpool' if arm.object_mode == 'rgb' else encoder, arm=arm.arm_id, seed=seed, fold=fold, protocol_id=SCREENING_PROTOCOL_ID)
    train_ids = sorted(store.training_graph_ids)
    val_ids = sorted((graph for graph in store.graph_ids if store.partition_by_graph[graph] == 'validation'))
    train_samples = store.samples(arm, train_ids)
    val_samples = store.samples(arm, val_ids)
    if not train_samples or not val_samples:
        raise RuntimeError('empty training or validation partition')
    token_budget = int(cfg['training']['token_budget'][encoder])
    oom_retries: list[dict[str, Any]] = []
    try:
        while True:
            model = _build_models(cfg['dataset'], arm, encoder, float(cfg['training']['dropout']))
            setattr(model, 'token_budget', token_budget)
            model.to(device)
            try:
                if cfg['dataset'] == 'sicapv2':
                    metrics, predictions = _train_sicap(model, train_samples, val_samples, rgb, arm, cfg, device, seed, output)
                else:
                    metrics, predictions = _train_crc(model, train_samples, val_samples, rgb, arm, cfg, device, seed, output)
                break
            except Exception as error:
                if not _is_cuda_oom(error) or token_budget <= 1:
                    raise
                reduced = max(1, token_budget // 2)
                oom_retries.append({'attempt': len(oom_retries) + 1, 'from_token_budget': token_budget, 'to_token_budget': reduced, 'error': str(error), 'resume_checkpoint': str(output / 'last.pt') if (output / 'last.pt').is_file() else None, 'time': datetime.now(timezone.utc).isoformat()})
                token_budget = reduced
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        prediction_path = output / 'validation_predictions.parquet'
        metrics_path = output / 'metrics.json'
        _atomic_parquet(prediction_path, predictions)
        _atomic_json(metrics_path, metrics)
    except Exception as error:
        failed = {'status': 'FAIL', 'dataset': cfg['dataset'], 'arm': arm.to_dict(), 'protocol_id': SCREENING_PROTOCOL_ID, 'encoder': encoder, 'seed': seed, 'fold': fold, 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(), 'oom_retries': oom_retries, 'runtime': _runtime_finish(runtime, device)}
        _atomic_json(manifest_path, failed)
        raise
    payload = {'status': 'PASS', 'dataset': cfg['dataset'], 'arm': arm.to_dict(), 'encoder': encoder, 'protocol_id': SCREENING_PROTOCOL_ID, 'batching': 'count_bucketed_dense_budget_set_encoding', 'job_seed_reset': True, 'effective_encoder': 'meanpool' if arm.object_mode == 'rgb' else encoder, 'seed': seed, 'fold': fold, 'train_graphs': len(train_samples), 'validation_graphs': len(val_samples), 'ncr_statistics': None if store.ncr_statistics(arm) is None else store.ncr_statistics(arm).as_dict(), 'shuffle_mapping': shuffle_mapping if arm.control_mode != 'none' else None, 'metrics': metrics, 'final_token_budget': token_budget, 'oom_retries': oom_retries, 'runtime': _runtime_finish(runtime, device), 'outputs': {'checkpoint': {'path': str(output / 'best.pt'), 'sha256': _sha256(output / 'best.pt')}, 'predictions': {'path': str(prediction_path), 'sha256': _sha256(prediction_path)}, 'metrics': {'path': str(metrics_path), 'sha256': _sha256(metrics_path)}}}
    _atomic_json(manifest_path, payload)
    _progress(output, 'job_complete', status='PASS', metrics=metrics)
    return payload

def _predict_sicap_rows(model: SICAPClassifier, samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, device: torch.device, token_budget: int) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for group in _token_batches(samples, token_budget, shuffle=False, seed=0):
            batch = collate_variable_tokens(group)
            with _autocast(device):
                outputs = model(**_fusion_inputs(batch, rgb, arm, device))
            probs = torch.softmax(outputs['grade_logits'].float(), dim=1).cpu().numpy()
            crib = torch.sigmoid(outputs['cribriform_logits'].float()).cpu().numpy()
            for sample, values, crib_score in zip(group, probs, crib):
                rows.append({'sample_id': sample.graph_id, 'graph_id': sample.graph_id, 'patient_id': str(sample.metadata['patient_id']), 'y_true': int(sample.metadata['label_id']), 'prob_0': float(values[0]), 'prob_1': float(values[1]), 'prob_2': float(values[2]), 'prob_3': float(values[3]), 'cribriform_score': float(crib_score), 'g4c_valid': bool(sample.metadata.get('g4c_valid', False)), 'g4c_label': int(sample.metadata['g4c_label']) if sample.metadata.get('g4c_valid', False) else None})
    return rows

def _predict_crc_rows(model: CRCFusionMIL, samples: list[GraphTokenSample], rgb: dict[str, np.ndarray], arm: ExperimentArm, device: torch.device, threshold: float, token_budget: int) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for patient, tiles in _patient_groups(samples).items():
            with _autocast(device):
                outputs, target = _crc_patient_forward(model, tiles, rgb, arm, device, token_budget)
            rows.append({'sample_id': patient, 'patient_id': patient, 'y_true': int(target.item()), 'score': float(torch.sigmoid(outputs['logits'].float())[0].cpu()), 'validation_threshold': float(threshold)})
    return rows

def _validation_threshold_from_job(path: Path) -> float:
    rows = _read_rows(path)
    if not rows:
        raise RuntimeError(f'empty validation predictions: {path}')
    values = {float(row['validation_threshold']) for row in rows}
    if len(values) != 1:
        raise RuntimeError(f'validation threshold is not constant in {path}')
    return next(iter(values))

def run_official_test(cfg: dict[str, Any], arms: Sequence[ExperimentArm], *, encoders: Sequence[str], seeds: Sequence[int], fold: int | None, device: str) -> dict[str, Any]:
    result_root = Path(cfg['paths']['result_root'])
    gate_path = result_root / 'selection_frozen.json'
    if not gate_path.is_file():
        raise RuntimeError('official TEST is locked: selection_frozen.json is absent')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('status') != 'PASS' or not gate.get('official_test_unlocked'):
        raise RuntimeError('official TEST is locked: frozen selection is not PASS')
    if gate.get('protocol_id') != SCREENING_PROTOCOL_ID:
        raise RuntimeError('official TEST is locked: incompatible downstream protocol')
    invalid_encoders = sorted(set(encoders) - set(gate['retained_encoders']))
    if invalid_encoders:
        raise RuntimeError(f'encoders were not retained before TEST: {invalid_encoders}')
    allowed_arms = set(gate['confirmation_arms'])
    invalid_arms = sorted({arm.arm_id for arm in arms} - allowed_arms)
    if invalid_arms:
        raise RuntimeError(f'arms were not frozen before TEST: {invalid_arms}')
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    data_root = Path(cfg['paths']['data_root'])
    model_input = Path(cfg['paths']['model_input_root'])
    manifest_rows = _read_rows(model_input / '03_graph_cache' / 'graph_index.parquet')
    patch_rows = {str(row['graph_id']): row for row in _read_rows(model_input / '00_manifest' / 'patch_manifest.parquet')}
    for row in manifest_rows:
        patch = patch_rows.get(str(row['graph_id']))
        if patch is None:
            raise RuntimeError(f"graph index row is absent from patch manifest: {row['graph_id']}")
        for name in ('g4c_label', 'g4c_valid', 'rgb_path', 'physical_width_um', 'physical_height_um'):
            row[name] = patch.get(name)
    graph_manifest = data_root / '00_manifest' / 'downstream_manifest.parquet'
    _atomic_parquet(graph_manifest, manifest_rows)
    official_ids = sorted((str(row['graph_id']) for row in manifest_rows if str(row['split']).lower() == 'test'))
    if not official_ids:
        raise RuntimeError('official TEST graph set is empty')
    folds = range(int(cfg['split']['validation_folds'])) if fold is None else (fold,)
    completed: list[dict[str, Any]] = []
    for current_fold in folds:
        train_ids = sorted((str(row['graph_id']) for row in manifest_rows if str(row['split']).lower() == 'train' and int(row['validation_fold']) != current_fold))
        partition = {graph: 'train' for graph in train_ids} | {graph: 'official_test' for graph in official_ids}
        rgb_path = data_root / '05_rgb_features' / f'fold_{current_fold:02d}' / 'seed_42' / 'rgb_features.parquet'
        rgb = _load_rgb(rgb_path)
        missing_rgb = sorted(set(official_ids) - rgb.keys())
        if missing_rgb:
            raise RuntimeError(f'fold {current_fold} RGB cache does not include frozen official TEST: {missing_rgb[:3]}')
        for seed in seeds:
            _seed_everything(int(seed))
            store = FoldTokenStore.from_parquet(data_root / '02_nucleus_tokens', data_root / '03_cell_tokens', data_root / '04_ncr_features', graph_manifest, training_graph_ids=train_ids, seed=int(seed), included_graph_ids=train_ids + official_ids, partition_by_graph=partition)
            shuffle_mapping = None
            if any((arm.control_mode != 'none' for arm in arms)):
                shuffle_mapping = _persist_shuffle_mapping(store, data_root, phase='official_test', fold=current_fold, seed=int(seed))
            for encoder in encoders:
                for arm in arms:
                    output = result_root / 'official_test' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{current_fold}'
                    job_path = output / 'job.json'
                    if job_path.is_file():
                        previous = json.loads(job_path.read_text(encoding='utf-8'))
                        if previous.get('status') == 'PASS':
                            completed.append(previous)
                            continue
                    output.mkdir(parents=True, exist_ok=True)
                    runtime = _runtime_start(cfg, target)
                    checkpoint_path = result_root / 'confirm' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{current_fold}' / 'best.pt'
                    prediction_path = output / 'official_test_predictions.parquet'
                    try:
                        if not checkpoint_path.is_file():
                            raise FileNotFoundError(checkpoint_path)
                        model = _build_models(cfg['dataset'], arm, encoder, float(cfg['training']['dropout']))
                        payload = torch.load(checkpoint_path, map_location=target, weights_only=False)
                        model.load_state_dict(payload['model'], strict=True)
                        model.to(target)
                        samples = store.samples(arm, official_ids)
                        if cfg['dataset'] == 'sicapv2':
                            predictions = _predict_sicap_rows(model, samples, rgb, arm, target, int(cfg['training']['token_budget'][encoder]))
                        else:
                            validation_path = checkpoint_path.with_name('validation_predictions.parquet')
                            predictions = _predict_crc_rows(model, samples, rgb, arm, target, _validation_threshold_from_job(validation_path), int(cfg['training']['token_budget'][encoder]))
                        _atomic_parquet(prediction_path, predictions)
                    except Exception as error:
                        _atomic_json(job_path, {'status': 'FAIL', 'dataset': cfg['dataset'], 'encoder': encoder, 'arm': arm.to_dict(), 'seed': int(seed), 'fold': int(current_fold), 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(), 'runtime': _runtime_finish(runtime, target)})
                        raise
                    record = {'status': 'PASS', 'dataset': cfg['dataset'], 'encoder': encoder, 'protocol_id': SCREENING_PROTOCOL_ID, 'arm': arm.to_dict(), 'seed': int(seed), 'fold': int(current_fold), 'official_test_samples': len(predictions), 'selection_gate': str(gate_path), 'checkpoint': str(checkpoint_path), 'shuffle_mapping': shuffle_mapping if arm.control_mode != 'none' else None, 'runtime': _runtime_finish(runtime, target), 'outputs': {'checkpoint_sha256': _sha256(checkpoint_path), 'predictions_sha256': _sha256(prediction_path)}}
                    _atomic_json(job_path, record)
                    completed.append(record)
    return {'status': 'PASS', 'phase': 'official_test', 'jobs': len(completed), 'completed': completed}

def run_grid(cfg: dict[str, Any], arms: Sequence[ExperimentArm], *, encoders: Sequence[str], seeds: Sequence[int], fold: int | None, validation_only: bool, device: str) -> dict[str, Any]:
    if not validation_only:
        pass
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable')
    data_root = Path(cfg['paths']['data_root'])
    model_input = Path(cfg['paths']['model_input_root'])
    graph_index = model_input / '03_graph_cache' / 'graph_index.parquet'
    manifest_rows = _read_rows(graph_index)
    patch_rows = {str(row['graph_id']): row for row in _read_rows(model_input / '00_manifest' / 'patch_manifest.parquet')}
    for row in manifest_rows:
        patch = patch_rows.get(str(row['graph_id']))
        if patch is None:
            raise RuntimeError(f"graph index row is absent from patch manifest: {row['graph_id']}")
        for name in ('g4c_label', 'g4c_valid', 'rgb_path', 'physical_width_um', 'physical_height_um'):
            row[name] = patch.get(name)
    graph_manifest = data_root / '00_manifest' / 'downstream_manifest.parquet'
    _atomic_parquet(graph_manifest, manifest_rows)
    folds = range(int(cfg['split']['validation_folds'])) if fold is None else (fold,)
    phase = 'screen' if validation_only else 'confirm'
    completed: list[dict[str, Any]] = []
    for current_fold in folds:
        eligible = [row for row in manifest_rows if str(row['split']).lower() == 'train']
        train_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) != current_fold]
        val_ids = [str(row['graph_id']) for row in eligible if int(row['validation_fold']) == current_fold]
        included = train_ids + val_ids
        partition = {graph: 'train' for graph in train_ids} | {graph: 'validation' for graph in val_ids}
        rgb_path = data_root / '05_rgb_features' / f'fold_{current_fold:02d}' / 'seed_42' / 'rgb_features.parquet'
        rgb = _load_rgb(rgb_path)
        for seed in seeds:
            _seed_everything(int(seed))
            store = FoldTokenStore.from_parquet(data_root / '02_nucleus_tokens', data_root / '03_cell_tokens', data_root / '04_ncr_features', graph_manifest, training_graph_ids=train_ids, seed=int(seed), included_graph_ids=included, partition_by_graph=partition)
            shuffle_mapping = None
            if any((arm.control_mode != 'none' for arm in arms)):
                shuffle_mapping = _persist_shuffle_mapping(store, data_root, phase='validation', fold=current_fold, seed=int(seed))
            for encoder in encoders:
                for arm in arms:
                    output = Path(cfg['paths']['result_root']) / phase / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{current_fold}'
                    completed.append(_run_job(cfg, store, rgb, arm, encoder, int(seed), current_fold, output, target, shuffle_mapping=shuffle_mapping))
    return {'status': 'PASS', 'phase': phase, 'jobs': len(completed), 'completed': completed}
