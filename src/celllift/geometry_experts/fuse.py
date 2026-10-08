from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .dataset import ARM_TO_EXPERT, DATASET_SPEC, EXPERT_TO_FUSED, FUSED_ARMS, EXPERT_ARMS, batch_result_root
from .io_utils import atomic_json, atomic_parquet, atomic_torch, read_json, read_parquet
from .models import FusionHead, PROB_CLIP
from .train_expert import job_dir, classes_for
FUSION_L2 = 0.001
FUSION_STEPS = 400
FUSION_LR = 0.05

def _stack(rows: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(row[key], float) for row in rows])

def expert_logits(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, float), PROB_CLIP, 1.0 - PROB_CLIP)
    if probability.ndim == 1 or probability.shape[-1] == 1:
        value = probability.reshape(-1)
        return np.log(value) - np.log1p(-value)
    return np.log(probability)

def center_logits(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return logits.reshape(-1, 1)
    return logits - logits.mean(axis=1, keepdims=True)

def baseline_logits(probability: np.ndarray) -> np.ndarray:
    return expert_logits(probability)

def _lookup(rows: list[dict]) -> dict[str, dict]:
    return {str(row['sample_id']): row for row in rows}

def load_fold_b(cfg: Mapping[str, Any], dataset: str, outer: int) -> dict[str, dict]:
    path = batch_result_root(cfg) / dataset / 'baseline_fold_scores' / f'fold_{int(outer):02d}.parquet'
    if not path.is_file():
        raise FileNotFoundError(path)
    lookup = {}
    for row in read_parquet(path):
        lookup[str(row['sample_id'])] = {'probability': np.asarray(row['probability'], float).tolist(), 'in_sample': bool(row['in_sample'])}
    return lookup

def apply_fold_b(rows: list[dict], scores: Mapping[str, dict], *, require_in_sample: bool) -> list[dict]:
    updated = []
    missing = []
    wrong_role = []
    for row in rows:
        key = str(row['sample_id'])
        entry = scores.get(key)
        if entry is None:
            missing.append(key)
            continue
        if require_in_sample and (not entry['in_sample']):
            wrong_role.append(key)
        copy = dict(row)
        copy['baseline_probability'] = entry['probability']
        copy['b_in_sample'] = entry['in_sample']
        updated.append(copy)
    if missing:
        raise RuntimeError(f'fold B missing {len(missing)} samples, first={missing[:3]}')
    if wrong_role:
        raise RuntimeError(f'fold B role mismatch for {len(wrong_role)} outer-train samples')
    return updated

def load_inner_tables(cfg: Mapping[str, Any], dataset: str, arm: str, seed: int, outer: int) -> tuple[list[dict], list[dict]]:
    oof, held = ([], [])
    inners = int(DATASET_SPEC[dataset]['inner_folds'])
    for inner in range(inners):
        folder = job_dir(cfg, dataset, arm, seed, outer, inner)
        if not (folder / 'job.json').is_file() or read_json(folder / 'job.json').get('status') != 'PASS':
            raise FileNotFoundError(folder / 'job.json')
        oof.extend(read_parquet(folder / 'oof.parquet'))
        held.append(_lookup(read_parquet(folder / 'heldout.parquet')))
    keys = sorted(held[0])
    for table in held[1:]:
        if set(table) != set(keys):
            raise RuntimeError('heldout identity mismatch across inner experts')
    averaged = []
    for key in keys:
        probabilities = np.mean([np.asarray(table[key]['final_probability'], float) for table in held], 0)
        row = dict(held[0][key])
        row['final_probability'] = probabilities.tolist()
        averaged.append(row)
    return (oof, averaged)

def fit_fusion(cfg: Mapping[str, Any], dataset: str, arm: str, outer: int, seed: int=42, device: str='cpu') -> dict[str, Any]:
    import torch
    expert = arm if arm in EXPERT_ARMS else ARM_TO_EXPERT[arm]
    fused = EXPERT_TO_FUSED[expert]
    result_root = batch_result_root(cfg)
    destination = result_root / dataset / 'predictions' / fused / f'seed_{seed}' / f'fold_{outer:02d}'
    expert_dest = result_root / dataset / 'predictions' / expert / f'seed_{seed}' / f'fold_{outer:02d}'
    if (destination / 'job.json').is_file() and read_json(destination / 'job.json').get('status') == 'PASS':
        return {'status': 'REUSED', **read_json(destination / 'job.json')}
    oof, heldout = load_inner_tables(cfg, dataset, expert, seed, outer)
    fold_b = load_fold_b(cfg, dataset, outer)
    oof = apply_fold_b(oof, fold_b, require_in_sample=True)
    classes = classes_for(dataset)
    s_oof = center_logits(expert_logits(_stack(oof, 'final_probability')))
    b_oof = baseline_logits(_stack(oof, 'baseline_probability'))
    if classes > 1:
        b_oof = np.log(np.clip(_stack(oof, 'baseline_probability'), PROB_CLIP, 1.0))
    elif b_oof.ndim == 1:
        b_oof = b_oof.reshape(-1, 1)
    if s_oof.ndim == 1:
        s_oof = s_oof.reshape(-1, 1)
    labels = np.asarray([row['label_id'] for row in oof], np.int64)
    mu = s_oof.mean(0, keepdims=True)
    evidence = s_oof - mu
    b_tensor = torch.as_tensor(b_oof, dtype=torch.float32, device=device)
    e_tensor = torch.as_tensor(evidence, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    head = FusionHead(classes).to(device)
    optimizer = torch.optim.Adam([head.a], lr=FUSION_LR)
    for _ in range(FUSION_STEPS):
        logits = head(b_tensor, e_tensor)
        if classes == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), y.float())
        else:
            loss = torch.nn.functional.cross_entropy(logits, y)
        loss = loss + FUSION_L2 * head.a.square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite fusion loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        head.project_()
    with torch.no_grad():
        zero = FusionHead(classes).to(device)
        identity = zero(b_tensor, e_tensor)
        if classes == 1:
            left = torch.sigmoid(identity).reshape(-1)
            right = torch.sigmoid(b_tensor).reshape(-1)
        else:
            left = torch.softmax(identity, -1)
            right = torch.softmax(b_tensor, -1)
        match = float((left - right).abs().max())
        if match > 1e-05:
            raise RuntimeError(f'a=0 does not recover B: max abs {match}')
    s_held = center_logits(expert_logits(_stack(heldout, 'final_probability')))
    if s_held.ndim == 1:
        s_held = s_held.reshape(-1, 1)
    e_held = s_held - mu
    if classes == 1:
        b_held = baseline_logits(_stack(heldout, 'baseline_probability'))
        if b_held.ndim == 1:
            b_held = b_held.reshape(-1, 1)
    else:
        b_held = np.log(np.clip(_stack(heldout, 'baseline_probability'), PROB_CLIP, 1.0))
    with torch.no_grad():
        fused_logits = head(torch.as_tensor(b_held, dtype=torch.float32, device=device), torch.as_tensor(e_held, dtype=torch.float32, device=device))
        if classes == 1:
            fused_prob = torch.sigmoid(fused_logits).cpu().numpy().reshape(-1, 1)
        else:
            fused_prob = torch.softmax(fused_logits, -1).cpu().numpy()
    fused_rows = []
    expert_rows = []
    for row, probability in zip(heldout, fused_prob):
        fused_rows.append({**row, 'final_probability': np.asarray(probability, float).tolist(), 'arm': fused})
        expert_rows.append({**row, 'arm': expert})
    destination.mkdir(parents=True, exist_ok=True)
    expert_dest.mkdir(parents=True, exist_ok=True)
    atomic_parquet(destination / 'predictions.parquet', fused_rows)
    atomic_parquet(expert_dest / 'predictions.parquet', expert_rows)
    atomic_torch(destination / 'fusion.pt', {'a': head.a.detach().cpu(), 'mu': mu, 'expert': expert})
    scale = head.scale.detach().cpu().numpy().tolist()
    manifest = {'status': 'PASS', 'dataset': dataset, 'arm': fused, 'expert': expert, 'fold': int(outer), 'seed': int(seed), 'a': scale, 'identity_match': match, 'b_train_source': 'current_outer_fold_checkpoint_5seed_mean', 'b_train_in_sample': True, 'nested_independence': 'fusion training uses current outer-fold B checkpoint in-sample scores; heldout uses the current fold OOF B', 'official_test_touched': False}
    atomic_json(destination / 'job.json', manifest)
    atomic_json(expert_dest / 'job.json', {**manifest, 'arm': expert})
    return manifest

def fuse_dataset(cfg: Mapping[str, Any], dataset: str, seed: int=42, device: str='cpu') -> dict[str, Any]:
    results = []
    for arm in EXPERT_ARMS:
        for outer in range(int(DATASET_SPEC[dataset]['folds'])):
            results.append(fit_fusion(cfg, dataset, arm, outer, seed, device))
    failed = [row for row in results if row.get('status') not in {'PASS', 'REUSED'}]
    return {'status': 'PASS' if not failed else 'FAIL', 'dataset': dataset, 'jobs': len(results), 'failed': len(failed)}
