from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from celllift.breast_roi_baseline.pipeline import finalize_graphs, load_config, run_build_direct3d, run_build_graphs, run_build_residual3d, run_build_shuffles, run_build_masks, run_evaluate, run_infer_geometry, run_preflight, run_prepare_tiles, run_probe, run_report, run_train_expert

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('preflight', 'prepare_tiles', 'build_masks', 'build_graphs', 'finalize_graphs', 'infer_geometry', 'build_direct3d', 'fit_probe', 'build_residual3d', 'build_shuffles', 'train_expert', 'train_experts', 'fit_fusion', 'evaluate', 'report'))
    value.add_argument('--config', type=Path, required=True)
    value.add_argument('--shard-id', type=int, default=0)
    value.add_argument('--num-shards', type=int, default=1)
    value.add_argument('--phase', choices=('development', 'final_test'), default='development')
    value.add_argument('--fold', type=int)
    value.add_argument('--seed', type=int)
    value.add_argument('--task', choices=('t7', 't3'))
    value.add_argument('--encoder', choices=('meanpool', 'deepsets'))
    value.add_argument('--expert', choices=('E0_RGB_MASK2D', 'E1_MASK2D', 'E2_MASK_DIRECT3D', 'E3_MASK_SHUF_DIRECT3D', 'E4_MASK_RESIDUAL3D', 'E5_MASK_SHUF_RESIDUAL3D'))
    value.add_argument('--device', default='cuda')
    value.add_argument('--tile-budget', type=int)
    value.add_argument('--object-budget', type=int)
    return value

def main() -> int:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.stage == 'preflight':
        result = run_preflight(cfg)
    elif args.stage == 'prepare_tiles':
        result = run_prepare_tiles(cfg)
    elif args.stage == 'build_masks':
        result = run_build_masks(cfg, args.shard_id, args.num_shards)
    elif args.stage == 'build_graphs':
        result = run_build_graphs(cfg, args.shard_id, args.num_shards)
    elif args.stage == 'finalize_graphs':
        result = finalize_graphs(cfg)
    elif args.stage == 'infer_geometry':
        result = run_infer_geometry(cfg, args.device)
    elif args.stage == 'build_direct3d':
        result = run_build_direct3d(cfg)
    elif args.stage == 'fit_probe':
        if args.fold is None:
            raise ValueError('fit_probe requires --fold')
        result = run_probe(cfg, args.phase, args.fold, args.device)
    elif args.stage == 'build_residual3d':
        result = run_build_residual3d(cfg, args.phase)
    elif args.stage == 'build_shuffles':
        if args.fold is None or args.seed is None:
            raise ValueError('build_shuffles requires fold/seed')
        result = run_build_shuffles(cfg, args.phase, args.fold, args.seed)
    elif args.stage in {'train_expert', 'train_experts'}:
        if None in (args.fold, args.seed, args.task, args.encoder, args.expert):
            raise ValueError('train_expert requires fold/seed/task/encoder/expert')
        result = run_train_expert(cfg, phase=args.phase, fold=args.fold, seed=args.seed, task=args.task, encoder=args.encoder, expert=args.expert, device=args.device, tile_budget=args.tile_budget, object_budget=args.object_budget)
    elif args.stage in {'fit_fusion', 'evaluate'}:
        if args.task is None or args.encoder is None:
            raise ValueError('evaluate requires task/encoder')
        result = run_evaluate(cfg, phase=args.phase, task=args.task, encoder=args.encoder)
    else:
        if args.task is None or args.encoder is None:
            raise ValueError('report requires task/encoder')
        result = run_report(cfg, phase=args.phase, task=args.task, encoder=args.encoder)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
