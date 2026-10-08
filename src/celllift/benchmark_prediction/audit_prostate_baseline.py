from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
import numpy as np
from celllift.runtime import torch
from torch.utils.data import DataLoader
from .io import job_dir
from .paths import load_config, comparison_result_root
from .splits import load_units
from .train_baseline import _MemmapRGB, _rgb_rows
from .train_geometry import _official_labels

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _fill(rows):
    official = _official_labels('sicapv2')
    out = []
    for row in rows:
        item = dict(row)
        if item.get('label_id') in (None, ''):
            for key in ('graph_id', 'patch_id', 'sample_id'):
                lid = official.get(str(item.get(key) or ''))
                if lid is not None:
                    item['label_id'] = lid
                    break
        if item.get('label_id') not in (None, ''):
            out.append(item)
    return out

def run(device: str='cuda') -> dict:
    from PIL import Image
    from celllift.geometry_baselines.paper_rgb import PaperFSConv, _gpu_preprocess
    cfg = load_config('sicapv2')
    cache_root = Path(cfg['paths']['paper_rgb_cache'])
    test_ids = set(map(str, next((u for u in load_units('sicapv2') if u['kind'] == 'sicap_full_train')).get('test') or []))
    rows = _fill(_rgb_rows(cache_root, test_ids, keys=('graph_id',)))
    ckpt = job_dir({'dataset': 'sicapv2', 'family': 'baseline', 'arm': 'B', 'unit': 'full_train', 'seed': 42, 'fold': 0}) / 'best.pt'
    report = {'ckpt': str(ckpt), 'ckpt_sha256': _sha256(ckpt) if ckpt.is_file() else None, 'n_test_rows': len(rows), 'source_path_field': sum((1 for row in rows if row.get('source_path'))), 'shared_cache_bug': False}
    by_class = defaultdict(list)
    for row in rows:
        by_class[int(row['label_id'])].append(row)
    rng = np.random.default_rng(42)
    sample = []
    for cls in range(4):
        pool = by_class.get(cls) or []
        if pool:
            take = min(8, len(pool))
            sample.extend((pool[int(i)] for i in rng.choice(len(pool), size=take, replace=False)))
    diffs, missing, logit_delta = ([], 0, [])
    ds = _MemmapRGB(sample, cache_root)
    cached = []
    for i, row in enumerate(sample):
        cached.append(ds[i]['image'].numpy())
        src = row.get('source_path') or row.get('path') or row.get('image_path')
        if not src or not Path(str(src)).is_file():
            missing += 1
            continue
        with Image.open(src) as image:
            pil = np.transpose(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), np.uint8), (2, 0, 1))
        diffs.append(float(np.mean(np.abs(cached[-1].astype(np.int16) - pil.astype(np.int16)))))
    report['pixel'] = {'n': len(sample), 'n_missing_source': missing, 'mean_abs': None if not diffs else float(np.mean(diffs)), 'max_abs': None if not diffs else float(np.max(diffs)), 'n_mismatch_gt1': int(sum((d > 1.0 for d in diffs)))}
    if ckpt.is_file() and sample:
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model = PaperFSConv(4).to(device)
        model.load_state_dict(saved['model'])
        model.eval()
        generator = torch.Generator(device=device).manual_seed(42)
        loader = DataLoader(_MemmapRGB(sample, cache_root), batch_size=len(sample), shuffle=False, num_workers=0)
        batch = next(iter(loader))
        with torch.no_grad():
            cache_logits = model(_gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator))
            cache_logits = cache_logits['logits'] if isinstance(cache_logits, dict) else cache_logits
        pil_tensors = []
        for row in sample:
            src = row.get('source_path') or row.get('path')
            if not src or not Path(str(src)).is_file():
                pil_tensors.append(batch['image'][len(pil_tensors)])
                continue
            with Image.open(src) as image:
                arr = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), np.uint8)
            pil_tensors.append(torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy()))
        pil_batch = torch.stack(pil_tensors, 0)
        with torch.no_grad():
            pil_logits = model(_gpu_preprocess(pil_batch.to(device), 'sicapv2', train=False, generator=generator))
            pil_logits = pil_logits['logits'] if isinstance(pil_logits, dict) else pil_logits
            logit_delta = (cache_logits - pil_logits).abs().mean().item()
            same_pred = int((cache_logits.argmax(1) == pil_logits.argmax(1)).sum().item())
        report['logits'] = {'mean_abs_delta': float(logit_delta), 'same_argmax': same_pred, 'n': len(sample)}
    if report['pixel'].get('mean_abs') not in (None, 0.0) and (report['pixel'].get('mean_abs') or 0) > 1.0:
        report['shared_cache_bug'] = True
        report['disposition'] = 're-infer_all_memmap_readers'
    else:
        report['disposition'] = 'old_test_script_only_keep_geometry_and_A'
    out = comparison_result_root() / 'analysis' / 'sicap_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    report['written'] = str(out)
    return report
if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
