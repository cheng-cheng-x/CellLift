from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.conditional_geometry.protocol import OFFICIAL_TEST_ALLOWED, PROTOCOL_ID

def read_config(path: Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if cfg['split'].get('official_test_frozen') is not True or OFFICIAL_TEST_ALLOWED:
        raise RuntimeError('conditional_geometry must remain validation-only')
    if 'probe' in cfg and int(cfg['probe']['inner_folds']) < 2:
        raise ValueError('probe requires at least two inner folds')
    fusion_protocols = {'complete_geometry_probability_late_fusion_late_geometry_fusion', 'complete_geometry_rank_late_fusion_rank_fusion_crc', 'protected_calibrated_logit_fusion_calibrated_fusion', 'protected_raw_scalar_logit_fusion_scalar_fusion_sicap', 'matched_mask_full3d_vs_residual3d_raw_residual_comparison'}
    if 'probe' not in cfg and cfg.get('protocol_id') not in fusion_protocols:
        raise ValueError('non-fusion protocols require a probe configuration')
    return cfg

def preflight(cfg: dict[str, Any], fold: int, graph_limit: int | None) -> dict[str, Any]:
    from celllift.conditional_geometry.common.probe_data import FoldProbeData
    data = FoldProbeData(cfg, fold, graph_limit=graph_limit)
    return {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': fold, 'graphs': len(data.graph_ids), 'training_graphs': int((data.roles == 'train').sum()), 'validation_graphs': int((data.roles == 'validation').sum()), 'official_test_touched': False}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('preflight', 'fit_probe', 'build_residuals', 'downstream_smoke', 'downstream', 'evaluate', 'mask_preflight', 'fit_mask_probe', 'build_mask_residuals', 'mask_downstream_smoke', 'mask_downstream', 'mask_evaluate', 'rgb_mask_preflight', 'fit_rgb_mask_probe', 'build_rgb_mask_residuals', 'rgb_mask_downstream_smoke', 'rgb_mask_downstream', 'rgb_mask_evaluate', 'late_fusion_evaluate', 'calibrated_fusion_evaluate', 'raw_scalar_fusion_evaluate', 'full3d_downstream', 'full3d_fusion_evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--graph-limit', type=int)
    parser.add_argument('--sample-limit', type=int)
    parser.add_argument('--encoder', action='append', choices=('meanpool', 'deepsets'))
    parser.add_argument('--seed', action='append', type=int)
    parser.add_argument('--arm', action='append', choices=('conditional_geometry_2D', 'conditional_geometry_RES3D', 'conditional_geometry_SHUF_RES3D', 'M3_MASK2D', 'M3_MASK_RES3D', 'M3_MASK_SHUF_RES3D', 'M5_MASK_FULL3D', 'M5_MASK_SHUF_FULL3D'))
    parser.add_argument('--official-test', action='store_true')
    args = parser.parse_args(argv)
    if args.official_test:
        raise RuntimeError('official TEST is structurally unavailable in SetEncoder conditional_geometry')
    cfg = read_config(args.config)
    if args.stage not in {'evaluate', 'mask_evaluate', 'rgb_mask_evaluate', 'late_fusion_evaluate', 'calibrated_fusion_evaluate', 'raw_scalar_fusion_evaluate', 'full3d_fusion_evaluate'} and args.fold is None:
        raise ValueError('--fold is required for this stage')
    if args.fold is not None and (not 0 <= args.fold < int(cfg['split']['validation_folds'])):
        raise ValueError('fold is outside the registered validation range')
    if args.stage == 'full3d_fusion_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_full3d_fusion import evaluate_full3d_fusion
        result = evaluate_full3d_fusion(cfg)
    elif args.stage == 'full3d_downstream':
        if args.graph_limit is not None or args.sample_limit is not None:
            raise RuntimeError('partial full-3D caches cannot enter downstream training')
        from celllift.conditional_geometry.scripts.run_mask_downstream import run_mask_residual_downstream
        result = run_mask_residual_downstream(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=args.seed, arms=args.arm)
    elif args.stage == 'raw_scalar_fusion_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_raw_scalar_fusion import evaluate_raw_scalar_fusion
        result = evaluate_raw_scalar_fusion(cfg)
    elif args.stage == 'calibrated_fusion_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_calibrated_fusion import evaluate_calibrated_fusion
        result = evaluate_calibrated_fusion(cfg)
    elif args.stage == 'late_fusion_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_late_fusion import evaluate_complete_geometry_late_fusion
        result = evaluate_complete_geometry_late_fusion(cfg)
    elif args.stage == 'rgb_mask_preflight':
        from celllift.conditional_geometry.common.rgb_mask_probe_data import RGBMaskFoldProbeData
        from celllift.conditional_geometry.rgb_mask_protocol import RGB_MASK_PROTOCOL_ID
        data = RGBMaskFoldProbeData(cfg, args.fold, graph_limit=args.graph_limit)
        result = {'status': 'PASS', 'protocol_id': RGB_MASK_PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': args.fold, 'graphs': len(data.graph_ids), 'training_graphs': int((data.roles == 'train').sum()), 'validation_graphs': int((data.roles == 'validation').sum()), 'conditioning': 'frozen RGB512 + nucleus-mask 36 rays', 'official_test_touched': False}
    elif args.stage == 'fit_rgb_mask_probe':
        from celllift.conditional_geometry.scripts.fit_rgb_mask_probe import fit_rgb_mask_fold_probe
        result = fit_rgb_mask_fold_probe(cfg, args.fold, device=args.device, graph_limit=args.graph_limit, sample_limit=args.sample_limit)
    elif args.stage == 'build_rgb_mask_residuals':
        from celllift.conditional_geometry.scripts.build_rgb_mask_residuals import build_rgb_mask_fold_residuals
        result = build_rgb_mask_fold_residuals(cfg, args.fold, device=args.device, graph_limit=args.graph_limit)
    elif args.stage == 'rgb_mask_downstream_smoke':
        from celllift.conditional_geometry.scripts.mask_downstream_smoke import mask_downstream_store_smoke
        result = mask_downstream_store_smoke(cfg, args.fold, graph_limit=int(args.graph_limit or 64), seed=int((args.seed or [42])[0]))
    elif args.stage == 'rgb_mask_downstream':
        if args.graph_limit is not None or args.sample_limit is not None:
            raise RuntimeError('partial RGB+mask caches cannot enter downstream training')
        from celllift.conditional_geometry.scripts.run_mask_downstream import run_mask_residual_downstream
        result = run_mask_residual_downstream(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=args.seed, arms=args.arm)
    elif args.stage == 'rgb_mask_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_mask import evaluate_mask_validation
        result = evaluate_mask_validation(cfg)
    elif args.stage == 'mask_preflight':
        from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData
        from celllift.conditional_geometry.mask_protocol import MASK_PROTOCOL_ID
        data = MaskFoldProbeData(cfg, args.fold, graph_limit=args.graph_limit)
        result = {'status': 'PASS', 'protocol_id': MASK_PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': args.fold, 'graphs': len(data.graph_ids), 'training_graphs': int((data.roles == 'train').sum()), 'validation_graphs': int((data.roles == 'validation').sum()), 'conditioning': 'nucleus-mask 36 rays only', 'official_test_touched': False}
    elif args.stage == 'fit_mask_probe':
        from celllift.conditional_geometry.scripts.fit_mask_probe import fit_mask_fold_probe
        result = fit_mask_fold_probe(cfg, args.fold, device=args.device, graph_limit=args.graph_limit, sample_limit=args.sample_limit)
    elif args.stage == 'build_mask_residuals':
        from celllift.conditional_geometry.scripts.build_mask_residuals import build_mask_fold_residuals
        result = build_mask_fold_residuals(cfg, args.fold, device=args.device, graph_limit=args.graph_limit)
    elif args.stage == 'mask_downstream_smoke':
        from celllift.conditional_geometry.scripts.mask_downstream_smoke import mask_downstream_store_smoke
        result = mask_downstream_store_smoke(cfg, args.fold, graph_limit=int(args.graph_limit or 64), seed=int((args.seed or [42])[0]))
    elif args.stage == 'mask_downstream':
        if args.graph_limit is not None or args.sample_limit is not None:
            raise RuntimeError('partial mask caches cannot enter downstream training')
        from celllift.conditional_geometry.scripts.run_mask_downstream import run_mask_residual_downstream
        result = run_mask_residual_downstream(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=args.seed, arms=args.arm)
    elif args.stage == 'mask_evaluate':
        from celllift.conditional_geometry.scripts.evaluate_mask import evaluate_mask_validation
        result = evaluate_mask_validation(cfg)
    elif args.stage == 'preflight':
        result = preflight(cfg, args.fold, args.graph_limit)
    elif args.stage == 'fit_probe':
        from celllift.conditional_geometry.scripts.fit_probe import fit_fold_probe
        result = fit_fold_probe(cfg, args.fold, device=args.device, graph_limit=args.graph_limit, sample_limit=args.sample_limit)
    elif args.stage == 'build_residuals':
        from celllift.conditional_geometry.scripts.build_residuals import build_fold_residuals
        result = build_fold_residuals(cfg, args.fold, device=args.device, graph_limit=args.graph_limit)
    elif args.stage == 'downstream_smoke':
        from celllift.conditional_geometry.scripts.downstream_smoke import downstream_store_smoke
        result = downstream_store_smoke(cfg, args.fold, graph_limit=int(args.graph_limit or 64), seed=int((args.seed or [42])[0]))
    elif args.stage == 'downstream':
        if args.graph_limit is not None or args.sample_limit is not None:
            raise RuntimeError('downstream smoke uses its dedicated store test; partial caches cannot enter training')
        from celllift.conditional_geometry.scripts.run_downstream import run_residual_downstream
        result = run_residual_downstream(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=args.seed, arms=args.arm)
    else:
        from celllift.conditional_geometry.scripts.evaluate import evaluate_validation
        result = evaluate_validation(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
