from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import sys
import time
from celllift.runtime import ResourcePath as Path
import numpy as np
from PIL import Image
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from celllift.model_inputs.common.segment import _eval, _load_models, canonical_relabel, instance_payload, stain_inputs
from celllift.model_inputs.common.utils import atomic_json, read_config, read_parquet_rows

def main() -> None:
    parser = argparse.ArgumentParser(description='Profile one not-yet-written formal nucleus patch')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--patch-id')
    args = parser.parse_args()
    cfg = read_config(args.config)
    data_root = Path(cfg['paths']['data_root'])
    rows = read_parquet_rows(data_root / '00_manifest/patch_manifest.parquet')
    row = next((item for item in rows if Path(item['rgb_path']).is_file() and (str(item['patch_id']) == args.patch_id if args.patch_id else not Path(item['nucleus_mask_path']).is_file())))
    timings = {}
    started = time.perf_counter()
    model, _ = _load_models(nucleus_only=True, require_cuda=bool(cfg['cellpose'].get('require_cuda', False)), vectorized_masks=bool(cfg['cellpose'].get('vectorized_get_masks', False)))
    timings['load_model_s'] = time.perf_counter() - started
    print({'stage': 'load_model', 'seconds': timings['load_model_s'], 'device': str(model.device)}, flush=True)
    started = time.perf_counter()
    with Image.open(row['rgb_path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    nucleus = stain_inputs(rgb, sorted_percentile=bool(cfg.get('cellpose', {}).get('sorted_percentile', False)))[0]
    timings['read_and_stain_s'] = time.perf_counter() - started
    print({'stage': 'read_and_stain', 'seconds': timings['read_and_stain_s']}, flush=True)
    started = time.perf_counter()
    mask = _eval(model, [nucleus], diameter=17.0, channels=[0, 0], batch_size=int(cfg['cellpose']['batch_size']))[0]
    timings['cellpose_eval_s'] = time.perf_counter() - started
    print({'stage': 'cellpose_eval', 'seconds': timings['cellpose_eval_s']}, flush=True)
    started = time.perf_counter()
    mask = canonical_relabel(mask)
    payload = instance_payload(mask)
    timings['canonical_and_payload_s'] = time.perf_counter() - started
    print({'stage': 'canonical_and_payload', 'seconds': timings['canonical_and_payload_s']}, flush=True)
    report = {'status': 'PASS', 'patch_id': row['patch_id'], 'nucleus_count': len(payload['instances']), 'shape': list(mask.shape), 'timings': timings}
    atomic_json(data_root / '05_qc/nucleus_single_patch_profile.json', report)
    print(report)
if __name__ == '__main__':
    main()
