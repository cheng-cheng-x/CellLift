from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import sys
from celllift.runtime import ResourcePath as Path
import numpy as np
from PIL import Image
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from celllift.model_inputs.common.segment import _eval, _load_models, canonical_relabel, stain_inputs
from celllift.model_inputs.common.utils import atomic_json, read_config, read_parquet_rows

def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrate exact-equivalent Cellpose nuclei batch size')
    parser.add_argument('--dataset', choices=('sicapv2', 'crc_msi'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--selected-only', action='store_true')
    parser.add_argument('--individual-images', action='store_true')
    parser.add_argument('--vectorized-masks', action='store_true')
    parser.add_argument('--all-pilot', action='store_true')
    parser.add_argument('--sorted-percentile', action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config)
    data_root = Path(cfg['paths']['data_root'])
    rows = read_parquet_rows(data_root / '00_manifest/pilot_manifest.parquet')
    if not args.all_pilot:
        rows = rows[:8]
    images: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    for row in rows:
        with Image.open(row['rgb_path']) as image:
            images.append(stain_inputs(np.asarray(image.convert('RGB')), sorted_percentile=args.sorted_percentile)[0])
        baseline.append(canonical_relabel(np.load(row['nucleus_mask_path'])))
    candidates = [8, 4, 2, 1] if args.dataset == 'sicapv2' else [32, 16, 8, 4, 2, 1]
    if args.selected_only:
        candidates = [int(cfg['cellpose']['batch_size'])]
    model, _ = _load_models(nucleus_only=True, vectorized_masks=args.vectorized_masks)
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    device = str(getattr(model, 'device', 'unknown'))
    if not device.startswith('cuda'):
        report = {'status': 'FAIL', 'dataset': args.dataset, 'sample_count': len(rows), 'selected_batch_size': None, 'model_device': device, 'cuda_available': bool(torch is not None and torch.cuda.is_available()), 'failure_reason': 'Cellpose did not select CUDA; CPU masks are not an accepted calibration.', 'results': []}
        atomic_json(data_root / '05_qc/nucleus_batch_calibration.json', report)
        print(report)
        raise SystemExit(1)
    results = []
    selected = None
    for candidate in candidates:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        masks = [_eval(model, [item], diameter=17.0, channels=[0, 0], batch_size=candidate)[0] for item in images] if args.individual_images else _eval(model, images, diameter=17.0, channels=[0, 0], batch_size=candidate)
        exact = all((np.array_equal(canonical_relabel(mask), reference) for mask, reference in zip(masks, baseline)))
        peak = int(torch.cuda.max_memory_allocated()) if torch is not None and torch.cuda.is_available() else None
        results.append({'batch_size': candidate, 'canonical_masks_exact': exact, 'peak_cuda_bytes': peak})
        if selected is None and exact:
            selected = candidate
    report = {'status': 'PASS' if selected is not None else 'FAIL', 'dataset': args.dataset, 'sample_count': len(rows), 'model_device': device, 'cuda_available': bool(torch is not None and torch.cuda.is_available()), 'image_chunk_mode': 'individual' if args.individual_images else 'list', 'mask_implementation': 'vectorized_exact' if args.vectorized_masks else 'cellpose_original', 'percentile_implementation': 'sorted_linear' if args.sorted_percentile else 'numpy_percentile', 'baseline': 'saved pilot canonical nucleus masks; batch_size=1', 'selected_batch_size': selected, 'results': results}
    atomic_json(data_root / '05_qc/nucleus_batch_calibration.json', report)
    print(report)
    if selected is None:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
