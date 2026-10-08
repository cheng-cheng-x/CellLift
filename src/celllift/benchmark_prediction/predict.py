from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
import numpy as np
from celllift.runtime import torch
from torch import nn
from torch.utils.data import DataLoader
from .io import job_dir
from .paths import load_config
from .splits import load_units
from .train_baseline import _MemmapRGB, _RoiTiles, _bracs_bags, _collate_roi, _rgb_rows
from .train_geometry import DualSet, _bags, _load_residual_map, _route_cfg, _stack_bag
from celllift.morphology_interaction.data import SceneCache
EXPERT_ARM = {'E2': 'G2', 'ES': 'G2R', 'ER': 'G2S'}
_SCENE_CACHE: dict[str, SceneCache] = {}

def _get_cache(dataset: str, spatial: bool=False, cfg: Mapping[str, Any] | None=None) -> SceneCache:
    hit = _SCENE_CACHE.get(dataset)
    if hit is not None:
        return hit
    cache = SceneCache(cfg or _route_cfg(dataset), dataset)
    cache.preload(load_spatial_maps=False)
    _SCENE_CACHE[dataset] = cache
    return cache

def _test_ids(dataset: str) -> list[str]:
    units = load_units(dataset)
    if dataset == 'sicapv2':
        return list(map(str, next((u for u in units if u['kind'] == 'sicap_full_train')).get('test') or []))
    if dataset == 'tcga_crc_msi':
        from .crc_score import qualified_patient_ids
        return qualified_patient_ids('test')
    return list(map(str, units[0].get('test') or []))

def _softmax_rows(logits: np.ndarray) -> list[list[float]]:
    z = np.asarray(logits, np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).tolist()

def predict_test(job: Mapping[str, Any], device: str) -> list[dict[str, Any]] | None:
    if job['family'] in {'residual', 'pretrain', 'fusion'}:
        return None
    if job['dataset'] == 'sicapv2' and job.get('unit') != 'full_train':
        return None
    destination = job_dir(job)
    ckpt = destination / 'best.pt'
    dataset = job['dataset']
    test_ids = _test_ids(dataset)
    if job['family'] == 'baseline':
        rows = _predict_baseline(job, device, test_ids)
    elif job['family'] in {'geometry', 'expert'}:
        rows = _predict_geometry(job, device, test_ids) if ckpt.is_file() else None
    elif job['family'] == 'feature':
        rows = _predict_feature(job, device, test_ids) if ckpt.is_file() else None
    elif job['family'] == 'route':
        rows = _predict_route(job, device, test_ids)
    else:
        rows = None
    if dataset == 'tcga_crc_msi' and rows:
        from .crc_score import filter_to_qualified
        rows = filter_to_qualified(rows, 'test')
    return rows

def _val_ids(dataset: str, job: Mapping[str, Any] | None=None) -> list[str]:
    if dataset == 'tcga_crc_msi':
        from .crc_score import qualified_patient_ids
        return qualified_patient_ids('val')
    if job and job.get('official_val'):
        return list(map(str, job['official_val']))
    units = load_units(dataset)
    if dataset == 'sicapv2':
        unit = next((u for u in units if u['kind'] == 'sicap_full_train'), units[0])
        return list(map(str, unit.get('val') or []))
    return list(map(str, units[0].get('val') or []))

def predict_val(job: Mapping[str, Any], device: str) -> list[dict[str, Any]] | None:
    destination = job_dir(job)
    ckpt = destination / 'best.pt'
    dataset = job['dataset']
    val_ids = _val_ids(dataset, job)
    if job['family'] == 'baseline':
        rows = _predict_baseline(job, device, val_ids)
    elif job['family'] in {'geometry', 'expert'}:
        rows = _predict_geometry(job, device, val_ids) if ckpt.is_file() else None
    elif job['family'] == 'feature':
        rows = _predict_feature(job, device, val_ids) if ckpt.is_file() else None
    elif job['family'] == 'route':
        rows = _predict_route(job, device, val_ids)
    else:
        rows = None
    if dataset == 'tcga_crc_msi' and rows:
        from .crc_score import filter_to_qualified
        rows = filter_to_qualified(rows, 'val')
    return rows

def _copy_route_val_if_present(job: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if job.get('family') != 'route':
        return None
    from .train_route import _cfg
    from celllift.morphology_interaction.foundation.loop import prediction_dir
    cfg = _cfg(job['dataset'], job['route'])
    src = prediction_dir(cfg, job['dataset'], job['route'], job['arm'], int(job['seed']), int(job.get('fold') or 0)) / 'predictions.parquet'
    if not src.is_file():
        return None
    import pyarrow.parquet as pq
    rows = pq.read_table(src).to_pylist()
    if job['dataset'] == 'tcga_crc_msi':
        from .crc_score import filter_to_qualified
        rows = filter_to_qualified(rows, 'val')
    return rows

def materialize_val_predictions(job: Mapping[str, Any], device: str) -> dict[str, Any]:
    from celllift.morphology_interaction.io_utils import atomic_parquet
    destination = job_dir(job)
    path = destination / 'predictions.parquet'
    if path.is_file() and (not job.get('refresh_val')):
        return {'status': 'EXISTS', 'path': str(path), 'n': None}
    rows = _copy_route_val_if_present(job)
    source = 'route_prediction_dir' if rows else 'infer'
    if rows is None:
        rows = predict_val(job, device)
    if not rows:
        return {'status': 'SKIP', 'reason': 'no_val_rows'}
    destination.mkdir(parents=True, exist_ok=True)
    atomic_parquet(path, rows)
    return {'status': 'PASS', 'path': str(path), 'n': len(rows), 'source': source}

def _predict_baseline(job, device, test_ids):
    from celllift.geometry_baselines.paper_rgb import _gpu_preprocess
    from .train_geometry import _official_labels
    dataset = job['dataset']
    destination = job_dir(job)
    ckpt = torch.load(destination / 'best.pt', map_location=device, weights_only=False)
    official = _official_labels(dataset)

    def _fill_rgb_labels(rows):
        for row in rows:
            if row.get('label_id') not in (None, ''):
                continue
            for key in ('graph_id', 'patch_id', 'sample_id', 'patient_id'):
                lid = official.get(str(row.get(key) or ''))
                if lid is not None:
                    row['label_id'] = lid
                    break
            else:
                row['label_id'] = 0
        return rows
    if dataset == 'sicapv2':
        from celllift.geometry_baselines.models import PaperFSConv
        cfg = load_config('sicapv2')
        cache_root = cfg['paths']['paper_rgb_cache']
        rows = _fill_rgb_labels(_rgb_rows(cache_root, set(map(str, test_ids))))
        model = PaperFSConv().to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        generator = torch.Generator(device=device).manual_seed(0)
        out = []
        with torch.no_grad():
            loader = DataLoader(_MemmapRGB(rows, cache_root), batch_size=64, shuffle=False, num_workers=0)
            for batch in loader:
                images = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator)
                prob = torch.softmax(model(images)['logits'], 1).cpu().numpy()
                for graph_id, label, p in zip(batch['graph_id'], batch['label'].numpy(), prob):
                    out.append({'sample_id': str(graph_id), 'label_id': int(label), 'probability': p.tolist(), 'arm': 'B'})
        return out
    if dataset == 'tcga_crc_msi':
        from celllift.geometry_baselines.models import build_paper_crc_resnet18
        cfg = load_config('tcga_crc_msi')
        cache_root = cfg['paths']['paper_rgb_cache']
        rows = _fill_rgb_labels(_rgb_rows(cache_root, set(map(str, test_ids)), keys=('graph_id', 'patient_id')))
        model, _ = build_paper_crc_resnet18(imagenet_weights=False)
        model = model.to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        generator = torch.Generator(device=device).manual_seed(0)
        grouped, labels = ({}, {})
        with torch.no_grad():
            loader = DataLoader(_MemmapRGB(rows, cache_root), batch_size=128, shuffle=False, num_workers=0)
            for batch in loader:
                images = _gpu_preprocess(batch['image'].to(device), 'tcga_crc_msi', train=False, generator=generator)
                output = model(images)
                logits = output['logits'] if isinstance(output, dict) else output
                logit_np = logits.float().cpu().numpy()
                for pid, label, logit in zip(batch['patient_id'], batch['label'].numpy(), logit_np):
                    grouped.setdefault(str(pid), []).append(logit)
                    labels[str(pid)] = int(label)
        from .crc_score import hard_vote_fraction
        out = []
        for pid, items in grouped.items():
            score = hard_vote_fraction(np.stack(items, 0))
            out.append({'sample_id': pid, 'label_id': labels[pid], 'probability': [1 - score, score], 'arm': 'B'})
        return out
    from celllift.pretrained import resnet18
    bags = _bracs_bags(test_ids)
    backbone = resnet18(weights=None)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    model = nn.ModuleDict({'backbone': backbone, 'head': nn.Linear(feat_dim, 7)}).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    out = []

    def forward(batch):
        images = batch['image'].to(device).float() / 255.0
        b, n, h, w, c = images.shape
        flat = images.permute(0, 1, 4, 2, 3).reshape(b * n, 3, h, w)
        feat = model['backbone'](flat).reshape(b, n, -1)
        mask = batch['mask'].to(device).unsqueeze(-1)
        pooled = (feat * mask).sum(1) / mask.sum(1).clamp_min(1)
        return model['head'](pooled)
    with torch.no_grad():
        loader = DataLoader(_RoiTiles(bags), batch_size=4, shuffle=False, collate_fn=_collate_roi, num_workers=0)
        for batch in loader:
            logits = forward(batch)
            prob = torch.softmax(logits, 1).cpu().numpy()
            for sample_id, label, p in zip(batch['sample_id'], batch['label'].numpy(), prob):
                out.append({'sample_id': sample_id, 'label_id': int(label), 'probability': p.tolist(), 'arm': 'B_rgb'})
    return out

def _fill_missing(dataset: str, bags: list[dict[str, Any]], test_ids: list[str]) -> list[dict[str, Any]]:
    if dataset != 'bracs':
        return bags
    have = {bag['sample_id'] for bag in bags}
    from .paths import split_root
    from celllift.runtime import json
    labels = {}
    path = split_root() / 'bracs' / 'test_rows.json'
    if path.is_file():
        for row in json.loads(path.read_text(encoding='utf-8')):
            labels[str(row['sample_id'])] = int(row['label_id'])
    for sid in test_ids:
        if sid not in have:
            bags.append({'sample_id': sid, 'graph_ids': [], 'label_id': labels.get(sid, 0)})
    return bags

def _predict_geometry(job, device, test_ids):
    dataset = job['dataset']
    cache = _get_cache(dataset)
    arm = EXPERT_ARM.get(job.get('arm_tag') or job['arm'], job['arm'])
    residual_map = _load_residual_map(job, cache, device) if arm in {'GR', 'G2S', 'G2+residual'} else None
    bags = _fill_missing(dataset, _bags(dataset, cache, list(map(str, test_ids))), test_ids)
    saved = torch.load(job_dir(job) / 'best.pt', map_location=device, weights_only=False)
    dims = saved.get('dims') or [2, 1]
    classes = 4 if dataset == 'sicapv2' else 7 if dataset == 'bracs' else 1
    model = DualSet(int(dims[0]), int(dims[1]), classes, saved.get('kind') or job.get('encoder') or 'deepsets').to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    out = []
    with torch.no_grad():
        for bag in bags:
            if not bag.get('graph_ids'):
                a = np.zeros((1, int(dims[0])), np.float32)
                b = np.zeros((1, int(dims[1])), np.float32)
            else:
                a, b = _stack_bag(cache, bag, arm, residual_map)
            logit = model(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)).detach().cpu().numpy()
            if classes == 1:
                p = float(1 / (1 + np.exp(-float(np.asarray(logit).reshape(-1)[0]))))
                probability = [1 - p, p]
            else:
                probability = _softmax_rows(logit)[0]
            out.append({'sample_id': bag['sample_id'], 'label_id': int(bag['label_id']), 'probability': probability, 'arm': job.get('arm_tag') or arm})
    return out

def _predict_feature(job, device, test_ids):
    from .train_feature import DualSet as _Dual
    dataset = job['dataset']
    cache = _get_cache(dataset)
    arm = 'G2' if job['arm'] in {'H2', 'C1'} else 'G3'
    bags = _fill_missing(dataset, _bags(dataset, cache, list(map(str, test_ids))), test_ids)
    saved = torch.load(job_dir(job) / 'best.pt', map_location=device, weights_only=False)
    weight = saved['geom']['enc_a.0.weight']
    dim_a = int(weight.shape[1])
    dim_b = int(saved['geom']['enc_b.0.weight'].shape[1]) if 'enc_b.0.weight' in saved['geom'] else 1
    classes = 7 if dataset == 'bracs' else 1
    geom = DualSet(dim_a, dim_b, 32, 'deepsets').to(device)
    head = nn.Sequential(nn.Linear(32 + 384, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, classes)).to(device)
    geom.load_state_dict(saved['geom'])
    head.load_state_dict(saved['head'])
    geom.eval()
    head.eval()

    def rgb_of(bag):
        vecs = []
        for graph_id in bag['graph_ids']:
            graph = cache.get(graph_id)
            dino = np.asarray(graph.dino, np.float32)
            include = np.asarray(graph.include, bool)
            if include.any():
                dino = dino[include]
            vecs.append(dino.mean(0) if dino.size else np.zeros(384, np.float32))
        if not vecs:
            return np.zeros(384, np.float32)
        return np.mean(np.stack(vecs, 0), 0)
    out = []
    with torch.no_grad():
        for bag in bags:
            if not bag.get('graph_ids'):
                a = np.zeros((1, dim_a), np.float32)
                b = np.zeros((1, dim_b), np.float32)
            else:
                a, b = _stack_bag(cache, bag, arm, None)
            g = geom(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device))
            rgb = torch.from_numpy(rgb_of(bag)).to(device)
            logit = head(torch.cat([g, rgb], 0)).detach().cpu().numpy()
            if classes == 1:
                p = float(1 / (1 + np.exp(-float(np.asarray(logit).reshape(-1)[0]))))
                probability = [1 - p, p]
            else:
                probability = _softmax_rows(logit)[0]
            out.append({'sample_id': bag['sample_id'], 'label_id': int(bag['label_id']), 'probability': probability, 'arm': job['arm']})
    return out

def _predict_route(job, device, test_ids):
    from celllift.morphology_interaction.data import SceneCache, build_batch
    from celllift.morphology_interaction.foundation.loop import _probabilities, job_dir as route_job_dir, classes_for
    from celllift.morphology_interaction.dataset import load_config as load_route_yaml, ARM_NODE_3D
    from celllift.benchmark_prediction.train_route import _cfg
    from celllift.benchmark_prediction.paths import PACKAGE
    cfg = _cfg(job['dataset'], job['route'])
    route_dest = route_job_dir(cfg, job['dataset'], job['route'], job['arm'], int(job['seed']), int(job.get('fold') or 0))
    ckpt_path = route_dest / 'best.pt'
    if not ckpt_path.is_file():
        return None
    saved = torch.load(ckpt_path, map_location=device, weights_only=False)
    cache = _get_cache(job['dataset'], cfg=cfg)
    from celllift.morphology_interaction.data import Sample
    from celllift.morphology_interaction.foundation.loop import _probabilities, classes_for
    classes = classes_for(job['dataset'])
    dummy = np.full(max(1, classes), 1.0 / max(1, classes), np.float32)
    bags = _bags(job['dataset'], cache, list(map(str, test_ids)))
    if job['dataset'] == 'bracs':
        bags = _fill_missing(job['dataset'], bags, test_ids)
    held = [Sample(str(bag['sample_id']), list(bag['graph_ids']), int(bag['label_id']), 0, str(bag['sample_id']), dummy.copy()) for bag in bags if bag.get('graph_ids')]
    if job['route'] == 'a':
        from celllift.morphology_interaction.route_a.models import RouteA as Model
    elif job['route'] == 'b':
        from celllift.morphology_interaction.route_b.models import RouteB as Model
    elif job['route'] == 'c':
        from celllift.morphology_interaction.route_c.models import RouteC as Model
    else:
        from celllift.morphology_interaction.route_d.models import RouteD as Model
    classes = classes_for(job['dataset'])
    model = Model(classes=classes, arm=job['arm'], bag=job['dataset'] != 'sicapv2').to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    kwargs = dict(mean=saved['mean'], std=saved['std'], fill3d=saved['fill3d'], edge_mean2d=saved.get('edge_mean2d'), edge_std2d=saved.get('edge_std2d'), edge_mean3d=saved.get('edge_mean3d'), edge_std3d=saved.get('edge_std3d'), arm=job['arm'], device=device, with_spatial=job['route'] in {'a', 'c'})
    out = []
    with torch.no_grad():
        for start in range(0, len(held), 8):
            chunk = held[start:start + 8]
            batch = build_batch(cache, chunk, **{k: v for k, v in kwargs.items() if v is not None or k in {'arm', 'device', 'with_spatial'}})
            logits = model(batch)['logits']
            probs = _probabilities(logits, classes).cpu().numpy()
            if classes == 1:
                probs = np.stack([1 - probs.reshape(-1), probs.reshape(-1)], 1)
            for sample, probability in zip(chunk, probs):
                out.append({'sample_id': sample.bag_id, 'label_id': int(sample.label_id), 'probability': np.asarray(probability, float).tolist(), 'arm': job['arm']})
    have = {str(row['sample_id']) for row in out}
    for bag in bags:
        if str(bag['sample_id']) in have:
            continue
        out.append({'sample_id': str(bag['sample_id']), 'label_id': int(bag['label_id']), 'probability': dummy.astype(float).tolist(), 'arm': job['arm']})
    return out
