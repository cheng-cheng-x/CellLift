from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from collections import Counter, defaultdict
from celllift.runtime import ResourcePath as Path
import numpy as np
from celllift.runtime import torch
from torch.utils.data import DataLoader
from .io import job_dir
from .paths import load_config, baseline_result_root
from .splits import load_units
from .train_baseline import _MemmapRGB, _rgb_rows
from .train_geometry import _official_labels

def _full_job() -> dict:
    return {'dataset': 'sicapv2', 'family': 'baseline', 'arm': 'B', 'unit': 'full_train', 'seed': 42, 'fold': 0}

def _index_audit(cache_root: Path) -> dict:
    from celllift.geometry_baselines.paper_rgb import read_rows
    report = {}
    for name in ('index.parquet', 'official_test/index.parquet'):
        path = cache_root / name
        if not path.is_file():
            report[name] = {'exists': False}
            continue
        rows = read_rows(str(path))
        splits = Counter((str(row.get('official_split') or '<missing>') for row in rows))
        shards = Counter((int(row['shard']) for row in rows))
        missing_split = sum((1 for row in rows if not row.get('official_split')))
        report[name] = {'exists': True, 'n': len(rows), 'official_split': dict(splits), 'shards': dict(sorted(shards.items())), 'missing_official_split': missing_split, 'would_read_train_shard': bool(name.startswith('official_test') and missing_split)}
    return report

def _epoch_lock() -> dict:
    epochs = []
    for fold in range(4):
        path = job_dir({'dataset': 'sicapv2', 'family': 'baseline', 'arm': 'B', 'unit': f'fold_{fold:02d}', 'seed': 42, 'fold': fold}) / 'job.json'
        if not path.is_file():
            epochs.append({'fold': fold, 'missing': True})
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        selected = int(payload.get('selected_epoch', payload.get('epochs', 1)))
        epochs.append({'fold': fold, 'selected_epoch': selected, 'locked_epochs': selected + 1, 'best': payload.get('best')})
    present = [row['locked_epochs'] for row in epochs if 'locked_epochs' in row]
    return {'folds': epochs, 'mean_locked': None if not present else max(1, int(round(sum(present) / len(present)))), 'note': 'full_train uses mean(selected_epoch+1); fold_02 historically contributed 8 → 50'}

def _confusion(labels: np.ndarray, pred: np.ndarray, n_class: int=4) -> list[list[int]]:
    mat = np.zeros((n_class, n_class), np.int64)
    for y, p in zip(labels, pred):
        if 0 <= int(y) < n_class and 0 <= int(p) < n_class:
            mat[int(y), int(p)] += 1
    return mat.tolist()

def _fill_labels(rows: list[dict]) -> list[dict]:
    official = _official_labels('sicapv2')
    filled = []
    for row in rows:
        item = dict(row)
        if item.get('label_id') in (None, ''):
            for key in ('graph_id', 'patch_id', 'sample_id', 'patient_id'):
                lid = official.get(str(item.get(key) or ''))
                if lid is not None:
                    item['label_id'] = lid
                    break
        if item.get('label_id') in (None, ''):
            continue
        filled.append(item)
    return filled

def _sample_compare(cache_root: Path, rows: list[dict], device: str, n: int=16) -> dict:
    from PIL import Image
    from celllift.geometry_baselines.paper_rgb import PaperFSConv, _gpu_preprocess
    dest = job_dir(_full_job())
    ckpt = dest / 'best.pt'
    if not ckpt.is_file():
        return {'status': 'SKIP', 'reason': 'no_full_train_ckpt'}
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model = PaperFSConv(4).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    rng = np.random.default_rng(42)
    by_class = defaultdict(list)
    for row in rows:
        by_class[int(row['label_id'])].append(row)
    sample = []
    for cls in range(4):
        pool = by_class.get(cls) or []
        if not pool:
            continue
        take = min(max(1, n // 4), len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sample.extend((pool[int(i)] for i in idx))
    loader = DataLoader(_MemmapRGB(sample, cache_root), batch_size=len(sample), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    cache_uint = batch['image'].cpu().numpy()
    pil_uint = []
    diffs = []
    for row, cached in zip(sample, cache_uint):
        path = row.get('path') or row.get('image_path') or row.get('rgb_path')
        if not path or not Path(str(path)).is_file():
            pil_uint.append(None)
            continue
        with Image.open(path) as image:
            arr = np.asarray(image.convert('RGB').resize((224, 224)), np.uint8)
        chw = np.transpose(arr, (2, 0, 1))
        pil_uint.append(chw)
        diffs.append(float(np.mean(np.abs(cached.astype(np.int16) - chw.astype(np.int16)))))
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        proc = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator)
        logits = model(proc)
        logits = logits['logits'] if isinstance(logits, dict) else logits
        pred = logits.argmax(1).cpu().numpy()
    return {'n': len(sample), 'classes': sorted({int(row['label_id']) for row in sample}), 'shards': sorted({int(row['shard']) for row in sample}), 'official_split': sorted({str(row.get('official_split') or '') for row in sample}), 'mean_abs_pil_vs_cache': None if not diffs else float(np.mean(diffs)), 'max_abs_pil_vs_cache': None if not diffs else float(np.max(diffs)), 'n_with_pil_path': sum((v is not None for v in pil_uint)), 'pred_hist': dict(Counter((int(v) for v in pred))), 'true_hist': dict(Counter((int(row['label_id']) for row in sample))), 'preprocess_finite': bool(torch.isfinite(proc).all().item())}

def _score_split(cache_root: Path, wanted: set[str], device: str, keys=('graph_id',)) -> dict:
    from celllift.geometry_baselines.paper_rgb import PaperFSConv, _gpu_preprocess
    from .metrics import qwk
    dest = job_dir(_full_job())
    ckpt = dest / 'best.pt'
    if not ckpt.is_file():
        return {'status': 'SKIP', 'reason': 'no_ckpt'}
    rows = _fill_labels(_rgb_rows(cache_root, wanted, keys=keys))
    if not rows:
        return {'status': 'SKIP', 'reason': 'no_rows', 'wanted': len(wanted)}
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    model = PaperFSConv(4).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    preds, labels = ([], [])
    loader = DataLoader(_MemmapRGB(rows, cache_root), batch_size=256, shuffle=False, num_workers=2)
    generator = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for batch in loader:
            images = _gpu_preprocess(batch['image'].to(device), 'sicapv2', train=False, generator=generator)
            logits = model(images)
            logits = logits['logits'] if isinstance(logits, dict) else logits
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(batch['label'].numpy())
    pred = np.concatenate(preds, 0)
    y = np.concatenate(labels, 0)
    return {'n': int(y.size), 'qwk': qwk(y, pred), 'true_hist': dict(Counter((int(v) for v in y))), 'pred_hist': dict(Counter((int(v) for v in pred))), 'confusion': _confusion(y, pred), 'missing_official_split': sum((1 for row in rows if not row.get('official_split'))), 'test_tagged': sum((1 for row in rows if str(row.get('official_split') or '').lower() == 'test'))}

def run(device: str='cuda') -> dict:
    cfg = load_config('sicapv2')
    cache_root = Path(cfg['paths']['paper_rgb_cache'])
    units = load_units('sicapv2')
    full = next((unit for unit in units if unit['kind'] == 'sicap_full_train'))
    train_ids = set(map(str, full.get('fit') or []))
    test_ids = set(map(str, full.get('test') or []))
    dest = job_dir(_full_job())
    manifest = {}
    if (dest / 'job.json').is_file():
        manifest = json.loads((dest / 'job.json').read_text(encoding='utf-8'))
    report = {'cache_root': str(cache_root), 'ckpt': str(dest / 'best.pt'), 'ckpt_exists': (dest / 'best.pt').is_file(), 'full_train_manifest': {'epochs': manifest.get('epochs'), 'selected_epoch': manifest.get('selected_epoch'), 'best': manifest.get('best'), 'fixed_epochs': manifest.get('fixed_epochs')}, 'index': _index_audit(cache_root), 'epoch_lock': _epoch_lock(), 'n_train': len(train_ids), 'n_test': len(test_ids)}
    report['sample_compare'] = _sample_compare(cache_root, _fill_labels(_rgb_rows(cache_root, test_ids, keys=('graph_id',))), device)
    report['train_confusion'] = _score_split(cache_root, train_ids, device)
    report['test_confusion'] = _score_split(cache_root, test_ids, device)
    test_index = report['index'].get('official_test/index.parquet') or {}
    cache_bug = bool(test_index.get('would_read_train_shard'))
    train_qwk = (report['train_confusion'] or {}).get('qwk')
    test_qwk = (report['test_confusion'] or {}).get('qwk')
    impl_bug = cache_bug
    if report['sample_compare'].get('mean_abs_pil_vs_cache') not in (None, 0.0) and (report['sample_compare'].get('mean_abs_pil_vs_cache') or 0) > 5:
        impl_bug = True
        report['pil_mismatch'] = True
    report['verdict'] = {'cache_would_read_train_shard': cache_bug, 'implementation_error': impl_bug, 'disposition': 're-infer_2122' if cache_bug else 'keep_anomalous_B_plus_paper_200' if not impl_bug else 'investigate', 'train_qwk': train_qwk, 'test_qwk': test_qwk, 'note': 'A2/AS do not read B and are not retrained'}
    out = baseline_result_root() / 'analysis' / 's0_sicap_b.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    report['written'] = str(out)
    return report
if __name__ == '__main__':
    print(json.dumps(run(), indent=2, default=str))
