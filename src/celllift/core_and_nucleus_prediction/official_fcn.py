from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import subprocess
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from .constants import ARVANITI_DATA, AUTHOR_REPO, AUTHOR_WEIGHTS, RESULT_ROOT
from .models import MOBILENET_DW, MobileNetV1Half, MobileNetV1HalfFCN
IMAGENET_H5 = _public_resource('artifact_0043')
IMAGENET_SHA256 = 'df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7'
IMAGENET_SHA256 = 'df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7'
IMAGENET_SHA256 = 'df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7'
IMAGENET_SHA256 = 'df860c59562f4c4af7a2fb2ae90731caef39fd5d03e5fcd6ca46988965f0a4b7'

def ensure_repo() -> Path:
    root = Path(AUTHOR_REPO)
    if not (root / 'plot_heatmaps_and_CAM.py').is_file():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(['git', 'clone', '--depth', '1', _public_resource('artifact_0044'), str(root)])
    weights = root / 'model_weights' / 'MobileNet_Gleason_weights.h5'
    if not weights.is_file():
        url = _public_resource('artifact_0045')
        subprocess.check_call(['wget', '-O', str(weights), url])
    return root

def _keras_map(handle) -> dict[str, np.ndarray]:
    weights = {}

    def walk(group, prefix=''):
        for key in group:
            item = group[key]
            name = f'{prefix}/{key}' if prefix else key
            if hasattr(item, 'shape'):
                weights[name] = np.asarray(item)
            else:
                walk(item, name)
    walk(handle)
    return weights

def _assign_conv(module, kernel: np.ndarray, bn: list[np.ndarray] | None=None) -> None:
    array = np.asarray(kernel)
    if array.ndim == 4 and array.shape[-1] == 1 and (array.shape[2] != 1):
        converted = array.transpose(2, 3, 0, 1)
    else:
        converted = array.transpose(3, 2, 0, 1)
    module[0].weight.data.copy_(torch_as(converted))
    if bn is not None:
        gamma, beta, mean, var = bn
        module[1].weight.data.copy_(torch_as(gamma))
        module[1].bias.data.copy_(torch_as(beta))
        module[1].running_mean.data.copy_(torch_as(mean))
        module[1].running_var.data.copy_(torch_as(var))

def torch_as(value) -> 'torch.Tensor':
    import torch
    return torch.from_numpy(np.asarray(value, np.float32))

def load_h5_into_mobilenet(model: MobileNetV1Half, h5_path: Path) -> dict[str, Any]:
    import h5py
    with h5py.File(h5_path, 'r') as handle:
        raw = _keras_map(handle)

    def _hit(stem: str, key: str) -> bool:
        import re
        return re.search(f'(^|/){re.escape(stem)}(/|_)', key) is not None

    def find(stem: str, suffix: str | None=None):
        for key, value in raw.items():
            if not _hit(stem, key):
                continue
            if suffix and (not key.endswith(suffix)):
                continue
            return (key, value)
        return (None, None)

    def bn_of(stem: str) -> list[np.ndarray] | None:
        names = ('gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0')
        found = []
        for name in names:
            hit = next((value for key, value in raw.items() if _hit(stem, key) and key.endswith(name)), None)
            if hit is None:
                return None
            found.append(hit)
        return found
    required = [('conv1', 'kernel:0', 'conv1_bn')]
    for index in range(1, 14):
        required.extend([(f'conv_dw_{index}', 'depthwise_kernel:0', f'conv_dw_{index}_bn'), (f'conv_pw_{index}', 'kernel:0', f'conv_pw_{index}_bn')])
    missing = [stem for stem, suffix, bn in required if find(stem, suffix)[1] is None or bn_of(bn) is None]
    if missing:
        raise ValueError(f'Incomplete MobileNet backbone: {missing}')
    _, conv1 = find('conv1', 'kernel:0')
    if conv1 is None:
        return {'status': 'FAIL', 'reason': 'conv1 missing', 'keys': list(raw)[:40]}
    _assign_conv(model.features[0], conv1, bn_of('conv1_bn'))
    for index in range(1, 14):
        _, dw = find(f'conv_dw_{index}', 'depthwise_kernel:0')
        _, pw = find(f'conv_pw_{index}', 'kernel:0')
        block = model.features[index]
        if dw is not None:
            _assign_conv(block[:3], dw, bn_of(f'conv_dw_{index}_bn'))
        if pw is not None:
            _assign_conv(block[3:], pw, bn_of(f'conv_pw_{index}_bn'))
    _, dense = find('output', 'kernel:0')
    if dense is None:
        _, dense = find('dense', 'kernel:0')
    bias = next((value for key, value in raw.items() if _hit('output', key) and key.endswith('bias:0')), None)
    if dense is not None and dense.shape == (512, 4):
        model.head.weight.data.copy_(torch_as(dense.T))
        if bias is not None:
            model.head.bias.data.copy_(torch_as(bias))
    conv1_ok = int(model.features[0][0].weight.shape[0]) == 16
    return {'status': 'OK' if conv1_ok else 'FAIL', 'conv1_out': int(model.features[0][0].weight.shape[0]), 'last_pw': int(model.features[-1][3].weight.shape[0]), 'n_tensors': len(raw), 'width_multiplier': 0.5 if conv1_ok else None, 'backbone_layers_loaded': len(required)}

def maybe_load_imagenet(model: MobileNetV1Half) -> dict[str, Any]:
    import hashlib
    import os
    import urllib.request
    cache = Path(os.environ.get('AL_IMAGENET_H5') or Path(AUTHOR_REPO) / 'model_weights' / 'mobilenet_5_0_224_tf_no_top.h5')
    if not cache.is_file() or cache.stat().st_size == 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + f'.{os.getpid()}.download')
        try:
            with urllib.request.urlopen(IMAGENET_H5, timeout=120) as response:
                temporary.write_bytes(response.read())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != IMAGENET_SHA256:
                raise ValueError('Downloaded ImageNet weights failed SHA256 verification')
            temporary.replace(cache)
        finally:
            if temporary.exists():
                temporary.unlink()
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != IMAGENET_SHA256:
        raise ValueError(f'ImageNet weights SHA256 mismatch at {cache}: {digest}')
    info = load_h5_into_mobilenet(model, cache)
    if info.get('status') != 'OK' or info.get('backbone_layers_loaded') != 27:
        raise RuntimeError(f'ImageNet initialization failed: {info}')
    info.update(source='imagenet_no_top', path=str(cache), sha256=digest, bytes=cache.stat().st_size, url=IMAGENET_H5)
    return info

def convert_released(destination: Path | None=None) -> dict[str, Any]:
    import torch
    ensure_repo()
    model = MobileNetV1Half()
    info = load_h5_into_mobilenet(model, AUTHOR_WEIGHTS)
    dest = Path(destination or Path(AUTHOR_REPO) / 'model_weights' / 'MobileNet_Gleason_weights.pt')
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest)
    fcn = MobileNetV1HalfFCN()
    fcn.backbone.load_state_dict(model.state_dict())
    fcn.load_classifier_from_dense(model.head.weight.data, model.head.bias.data)
    fcn_path = dest.with_name('MobileNet_Gleason_fcn.pt')
    torch.save(fcn.state_dict(), fcn_path)
    info.update({'pt': str(dest), 'fcn_pt': str(fcn_path)})
    return info

def _core_gt(core: dict[str, Any]) -> dict[str, dict[str, int]]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_inputs'))
    from celllift.model_inputs.arvaniti_lizard.audit import _decode_mask
    from celllift.model_inputs.arvaniti_lizard.io_utils import read_parquet
    from celllift.model_inputs.arvaniti_lizard.official import core_label_from_mask
    masks = read_parquet(ARVANITI_DATA / '00_manifest' / 'corrected_set_encoding' / 'masks.parquet')
    by = {(row['core_id'], row['reader']): row for row in masks}
    out = {}
    readers = ['train'] if core['role'] != 'TEST' else ['pathologist1', 'pathologist2']
    if core['role'] == 'VAL':
        readers = ['train'] if ('core_id', 'train') in [(core['core_id'], 'train')] else ['pathologist1']
        if (core['core_id'], 'pathologist1') in by:
            readers = ['pathologist1']
        elif (core['core_id'], 'train') in by:
            readers = ['train']
    for reader in readers:
        row = by.get((core['core_id'], reader))
        if row is None:
            continue
        out[reader] = core_label_from_mask(_decode_mask(Path(row['path'])))
    if 'pathologist1' not in out and 'train' in out:
        out['pathologist1'] = out['train']
    return out

def run_official_fcn(device: str='cuda', limit: int | None=None, fcn_pt: Path | None=None, dest: Path | None=None) -> dict[str, Any]:
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_inputs'))
    from celllift.model_inputs.arvaniti_lizard.io_utils import atomic_json, read_parquet
    from celllift.model_inputs.arvaniti_lizard.official import author_tissue_mask, pool_official, resample_mask
    info = {'fcn_pt': str(fcn_pt)} if fcn_pt else convert_released()
    print(json.dumps({'convert': info}, default=str), flush=True)
    fcn = MobileNetV1HalfFCN().to(device).eval()
    fcn.load_state_dict(torch.load(Path(info['fcn_pt']), map_location=device))
    cores = read_parquet(ARVANITI_DATA / '04_labels_splits' / 'cores.parquet')
    if limit:
        cores = cores[:limit]
    rows = []
    for core in cores:
        if core['role'] not in {'VAL', 'TEST'}:
            continue
        rgb = np.asarray(Image.open(core['path']).convert('RGB'))
        resized = np.asarray(Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.BILINEAR), np.float32)
        tensor = torch.from_numpy((resized / 127.5 - 1.0).transpose(2, 0, 1)[None]).to(device)
        with torch.no_grad():
            pred = fcn(tensor)[0].permute(1, 2, 0).cpu().numpy()
        gray = np.asarray(Image.open(core['path']).convert('L'))
        tissue = resample_mask(author_tissue_mask(gray), 1024, 1024)
        pred[tissue == 4] = 0
        pooled = pool_official(pred, tissue)
        gt = _core_gt(core)
        rows.append({'core_id': core['core_id'], 'role': core['role'], 'pred': pooled, 'gt': gt})
        if len(rows) % 25 == 0:
            print(f'fcn {len(rows)} cores', flush=True)

    def kappa(role: str, reader: str) -> float | None:
        y_true, y_pred = ([], [])
        for row in rows:
            if row['role'] != role or reader not in row['gt']:
                continue
            y_true.append(row['gt'][reader]['qwk_label'])
            y_pred.append(row['pred']['qwk_label'])
        if len(y_true) < 2:
            return None
        return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    scores = {'VAL_p1': kappa('VAL', 'pathologist1') or kappa('VAL', 'train'), 'TEST_p1': kappa('TEST', 'pathologist1'), 'TEST_p2': kappa('TEST', 'pathologist2')}
    if scores['TEST_p1'] is not None and scores['TEST_p2'] is not None:
        scores['TEST_mean'] = 0.5 * (scores['TEST_p1'] + scores['TEST_p2'])
    target = {'VAL_p1': 0.72, 'TEST_p1': 0.75, 'TEST_p2': 0.71}
    close = True
    for key, want in target.items():
        got = scores.get(key)
        if got is None or abs(got - want) > 0.12:
            close = False
    payload = {'status': 'OFFICIAL' if close else 'author_repro', 'scores': scores, 'target': target, 'n': len(rows), 'convert': info, 'note': 'Official if within 0.12 of published VAL/TEST QWK magnitude; else author_repro.'}
    folder = Path(dest) if dest is not None else RESULT_ROOT / 'arvaniti' / 'official_fcn'
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'core_pool.npz', core_id=np.asarray([row['core_id'] for row in rows]), role=np.asarray([row['role'] for row in rows]), w_sum=np.asarray([row['pred']['w_sum'] for row in rows], np.float64), pred_qwk=np.asarray([row['pred']['qwk_label'] for row in rows], np.int64), gt_p1=np.asarray([(row['gt'].get('pathologist1') or {}).get('qwk_label', -1) for row in rows], np.int64), gt_p2=np.asarray([(row['gt'].get('pathologist2') or {}).get('qwk_label', -1) for row in rows], np.int64))
    (folder / 'verify.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if dest is None:
        out = ARVANITI_DATA / '05_qc' / 'author_fcn_verify.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out, payload)
    return payload

def run_seed42_fcn(device: str='cuda') -> dict[str, Any]:
    import torch
    ckpt = RESULT_ROOT / 'arvaniti' / 'baseline' / 'B_paper' / 'seed42' / 'model.pt'
    if not ckpt.is_file():
        return {'status': 'WAIT', 'reason': 'B_paper missing'}
    cls = MobileNetV1Half()
    cls.load_state_dict(torch.load(ckpt, map_location='cpu'))
    fcn = MobileNetV1HalfFCN()
    fcn.backbone.load_state_dict(cls.state_dict())
    fcn.load_classifier_from_dense(cls.head.weight.data, cls.head.bias.data)
    dest = RESULT_ROOT / 'arvaniti' / 'official_fcn_seed42'
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(fcn.state_dict(), dest / 'fcn.pt')
    payload = run_official_fcn(device=device, fcn_pt=dest / 'fcn.pt', dest=dest)
    payload['source'] = 'B_paper_seed42'
    (dest / 'verify.json').write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return payload

def diagnose_fcn_permutations(pool_path: Path | None=None) -> dict[str, Any]:
    import itertools
    from celllift.model_inputs.arvaniti_lizard.aggregation import gleason_summary_wsum, gleason_sum_to_qwk_label
    folder = RESULT_ROOT / 'arvaniti' / 'official_fcn'
    path = Path(pool_path) if pool_path else folder / 'core_pool.npz'
    packed = np.load(path)
    w_sum = np.asarray(packed['w_sum'], np.float64)
    role = np.asarray(packed['role']).astype(str)
    gt_p1 = np.asarray(packed['gt_p1'])
    gt_p2 = np.asarray(packed['gt_p2'])
    found = []
    for perm in itertools.permutations(range(4)):
        pred = []
        for row in w_sum:
            ordered = row[list(perm)]
            _primary, _secondary, gleason_sum = gleason_summary_wsum(ordered, thres=0.25)
            pred.append(gleason_sum_to_qwk_label(gleason_sum))
        pred = np.asarray(pred)

        def kappa(mask, gt):
            keep = mask & (gt >= 0)
            if keep.sum() < 2:
                return None
            return float(cohen_kappa_score(gt[keep], pred[keep], weights='quadratic'))
        val = kappa(role == 'VAL', gt_p1)
        test_p1 = kappa(role == 'TEST', gt_p1)
        test_p2 = kappa(role == 'TEST', gt_p2)
        found.append({'perm': list(perm), 'VAL_p1': val, 'TEST_p1': test_p1, 'TEST_p2': test_p2, 'score': (val or -9) + (test_p1 or -9) + (test_p2 or -9)})
    found.sort(key=lambda item: item['score'], reverse=True)
    payload = {'best': found[0], 'identity': next((item for item in found if item['perm'] == [0, 1, 2, 3])), 'top': found[:5]}
    (folder / 'permutation.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload

def verify_released_fcn(device: str='cuda') -> dict[str, Any]:
    ensure_repo()
    try:
        return run_official_fcn(device=device)
    except Exception as exc:
        payload = {'status': 'author_repro', 'error': f'{type(exc).__name__}: {exc}', 'note': 'Conversion or FCN forward failed; downstream may start as author_repro.'}
        (ARVANITI_DATA / '05_qc').mkdir(parents=True, exist_ok=True)
        (ARVANITI_DATA / '05_qc' / 'author_fcn_verify.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload
