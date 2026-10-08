from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from celllift.geometry_baselines.io_utils import atomic_json, sha256
from celllift.geometry_baselines.models import PaperFSConv, build_paper_crc_resnet18
from celllift.geometry_baselines.paper_rgb import PaperRGBSpec, _gpu_preprocess, _loader, _select_outer, read_rows
from .protocol import PROTOCOL_ID, assert_validation_only_path

def _forward(model: torch.nn.Module, images: torch.Tensor, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset == 'sicapv2':
        output = model(images)
        return (output['features'], output['logits'])
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = torch.flatten(model.avgpool(x), 1)
    return (features, model.fc(features))

@torch.no_grad()
def _extract(model: torch.nn.Module, rows: Sequence[dict[str, Any]], cache_root: Path, spec: PaperRGBSpec, seed: int, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    loader = _loader(rows, cache_root, spec, shuffle=False, seed=seed, include_labels=True)
    features, logits = ([], [])
    generator = torch.Generator(device='cuda').manual_seed(seed)
    model.eval()
    for batch in loader:
        images = _gpu_preprocess(batch['image'], dataset, train=False, generator=generator)
        value, score = _forward(model, images, dataset)
        features.append(value.float().cpu().numpy())
        logits.append(score.float().cpu().numpy())
    return (np.concatenate(features), np.concatenate(logits))

def cache_fold_features(cfg: Mapping[str, Any], *, fold: int, seed: int, workers: int=8) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError('paper feature extraction requires CUDA')
    dataset = cfg['dataset']
    result_root = assert_validation_only_path(cfg['paths']['result_root'])
    destination = result_root / 'rgb_features' / f'seed_{seed}' / f'fold_{fold}.npz'
    manifest_path = destination.with_suffix('.json')
    if manifest_path.is_file() and destination.is_file():
        from celllift.runtime import json
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('status') == 'PASS' and old.get('npz_sha256') == sha256(destination):
            return {**old, 'reused': True}
    geometry_baselines_data = Path(cfg['paths']['geometry_baselines_data_root'])
    cache_root = geometry_baselines_data / '01_rgb224_cache'
    rows = read_rows(cache_root / 'index.parquet')
    train_rows, validation_rows = _select_outer(rows, fold)
    all_rows = list(train_rows) + list(validation_rows)
    checkpoint = Path(cfg['paths']['geometry_baselines_result_root']) / 'paper_baseline' / f'fold_{fold:02d}' / f'seed_{seed}' / 'best.pt'
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    if dataset == 'sicapv2':
        model = PaperFSConv().cuda()
    else:
        model, _ = build_paper_crc_resnet18(imagenet_weights=False)
        model = model.cuda()
    model.load_state_dict(saved['model_state_dict'])
    spec = PaperRGBSpec.for_dataset(dataset, workers)
    features, logits = _extract(model, all_rows, cache_root, spec, seed, dataset)
    if features.shape != (len(all_rows), 512) or not np.isfinite(features).all():
        raise RuntimeError('invalid paper-RGB feature cache')
    expected_path = checkpoint.with_name('validation_predictions.parquet')
    import pyarrow.parquet as pq
    expected_rows = pq.read_table(expected_path, partitioning=None).to_pylist()
    expected = {str(row['graph_id']): np.asarray(row['logits'], np.float32) for row in expected_rows}
    offset = len(train_rows)
    expected_matrix = np.stack([expected[str(row['graph_id'])] for row in validation_rows])
    observed_matrix = logits[offset:]
    maximum_error = float(np.max(np.abs(observed_matrix - expected_matrix)))
    decision_mismatches = int(np.count_nonzero(observed_matrix.argmax(1) != expected_matrix.argmax(1)))
    if decision_mismatches or maximum_error > 0.005:
        raise RuntimeError(f'feature extractor changes frozen logits: max error {maximum_error}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.savez(stream, graph_ids=np.asarray([str(row['graph_id']) for row in all_rows]), patient_ids=np.asarray([str(row['patient_id']) for row in all_rows]), labels=np.asarray([int(row['label_id']) for row in all_rows], np.int16), partitions=np.asarray([0] * len(train_rows) + [1] * len(validation_rows), np.uint8), features=features.astype(np.float32), logits=logits.astype(np.float32))
    os.replace(temporary, destination)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'fold': fold, 'seed': seed, 'training_rows': len(train_rows), 'validation_rows': len(validation_rows), 'feature_dim': 512, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'npz': str(destination), 'npz_sha256': sha256(destination), 'heldout_logit_max_abs_error': maximum_error, 'heldout_decision_mismatches': decision_mismatches, 'fold_aligned': True, 'validation_only': True, 'official_test_touched': False}
    atomic_json(manifest_path, payload)
    return payload
