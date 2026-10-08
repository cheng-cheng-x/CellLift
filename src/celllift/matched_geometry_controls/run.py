from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
import sys
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from celllift.matched_geometry_controls.pipeline import finalize_modal, finalize_shards, load_config, run_cache_inputs, run_freeze, run_infer, run_lock, run_meanpool, run_meanpool_all, run_modal, run_residual, run_shuffle, run_train
from celllift.matched_geometry_controls.protocol import ARMS, require_frozen

def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('lock', 'cache-inputs', 'finalize-inputs', 'infer', 'build-modal', 'finalize-modal', 'fit-residual', 'build-shuffle', 'build-meanpool', 'build-meanpool-all', 'train-expert', 'evaluate-oof', 'inventory', 'freeze-protocol', 'official-fit-residual', 'official-build-shuffle', 'official-build-meanpool', 'official-train-geometry', 'official-train-b0', 'official-fuse', 'official-evaluate'))
    value.add_argument('--config', type=Path)
    value.add_argument('--config-dir', type=Path)
    value.add_argument('--fold', type=int)
    value.add_argument('--arm', choices=tuple(ARMS))
    value.add_argument('--encoder', choices=('meanpool', 'deepsets'))
    value.add_argument('--seed', type=int)
    value.add_argument('--shard-id', type=int, default=0)
    value.add_argument('--num-shards', type=int, default=1)
    value.add_argument('--device', default='cuda')
    value.add_argument('--official-test', action='store_true')
    value.add_argument('--global-result-root', type=Path)
    return value

def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage == 'freeze-protocol':
        if args.config_dir is None or args.global_result_root is None:
            raise ValueError('freeze-protocol needs --config-dir and --global-result-root')
        configs = {name: args.config_dir / f'{name}.yaml' for name in ('sicapv2', 'tcga_crc_msi', 'bracs')}
        result = run_freeze(configs, args.global_result_root)
    else:
        if args.config is None:
            raise ValueError('stage requires --config')
        cfg = load_config(args.config)
        if args.official_test:
            require_frozen(cfg['paths']['global_result_root'])
        if args.stage == 'lock':
            result = run_lock(cfg)
        elif args.stage == 'cache-inputs':
            result = run_cache_inputs(cfg, args.shard_id, args.num_shards, args.device, args.official_test)
        elif args.stage == 'finalize-inputs':
            root = Path(cfg['paths']['data_root']) / ('09_official_test/01_projection_scene_inputs' if args.official_test else '01_projection_scene_inputs')
            result = finalize_shards(root, 'manifest_shard_*.json', root / 'manifest.json')
        elif args.stage == 'infer':
            result = run_infer(cfg, args.shard_id, args.num_shards, args.device, args.official_test)
        elif args.stage == 'build-modal':
            result = run_modal(cfg, args.shard_id, args.num_shards, args.official_test)
        elif args.stage == 'finalize-modal':
            result = finalize_modal(cfg, args.official_test)
        elif args.stage == 'fit-residual':
            if args.fold is None:
                raise ValueError('fit-residual requires --fold')
            result = run_residual(cfg, args.fold, args.device)
        elif args.stage == 'build-shuffle':
            if args.fold is None:
                raise ValueError('build-shuffle requires --fold')
            result = run_shuffle(cfg, args.fold)
        elif args.stage == 'build-meanpool':
            if args.fold is None or args.arm not in {'D', 'DS', 'R', 'RS'}:
                raise ValueError('build-meanpool requires fold and geometry arm')
            result = run_meanpool(cfg, args.fold, args.arm)
        elif args.stage == 'build-meanpool-all':
            if args.fold is None:
                raise ValueError('build-meanpool-all requires --fold')
            result = run_meanpool_all(cfg, args.fold)
        elif args.stage == 'train-expert':
            if None in (args.fold, args.arm, args.encoder, args.seed):
                raise ValueError('train-expert requires fold/arm/encoder/seed')
            result = run_train(cfg, args.fold, args.arm, args.encoder, args.seed, args.device)
        elif args.stage == 'evaluate-oof':
            if args.encoder is None:
                raise ValueError('evaluate-oof requires --encoder')
            from celllift.matched_geometry_controls.evaluation import evaluate_oof
            result = evaluate_oof(cfg, args.encoder)
        elif args.stage == 'inventory':
            from celllift.matched_geometry_controls.inventory import verify_oof_inventory
            result = verify_oof_inventory(cfg)
        else:
            from celllift.matched_geometry_controls import official
            if args.stage == 'official-fit-residual':
                result = official.run_official_residual(cfg, args.fold, args.device)
            elif args.stage == 'official-build-shuffle':
                result = official.run_official_shuffle(cfg, args.fold)
            elif args.stage == 'official-build-meanpool':
                result = official.run_official_meanpool_all(cfg, args.fold)
            elif args.stage == 'official-train-geometry':
                if None in (args.arm, args.encoder, args.seed):
                    raise ValueError('official-train-geometry requires arm/encoder/seed')
                result = official.run_official_geometry(cfg, args.fold, args.arm, args.encoder, args.seed, args.device)
            elif args.stage == 'official-train-b0':
                if cfg['dataset'] != 'bracs' or None in (args.fold, args.encoder, args.seed):
                    raise ValueError('official-train-b0 is BRACS-only and requires fold/encoder/seed')
                result = official.run_official_bracs_b0(cfg, args.fold, args.encoder, args.seed, args.device)
            elif args.stage == 'official-fuse':
                if args.encoder is None:
                    raise ValueError('official-fuse requires encoder')
                result = official.run_official_fusion(cfg, args.encoder)
            else:
                result = official.run_official_evaluate(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
