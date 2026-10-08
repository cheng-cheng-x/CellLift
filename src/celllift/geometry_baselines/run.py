from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
import sys
from typing import Any
ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
from celllift.geometry_baselines.io_utils import atomic_json
from celllift.geometry_baselines.protocol import ARMS, PROTOCOL_ID, require_test_gate
STAGES = ('preflight', 'cache_paper_rgb', 'train_paper_rgb', 'predict_paper_rgb', 'build_modal_tokens', 'build_shuffle_maps', 'train_patient_geometry', 'train_tile_geometry', 'predict_geometry', 'evaluate_validation', 'freeze_selection', 'predict_test', 'evaluate_test', 'report')

def read_config(path: Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    if cfg.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('geometry_baselines config protocol mismatch')
    return cfg

def _cache_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg['paths']['data_root']) / '01_rgb224_cache'

def _not_test(stage: str, args: argparse.Namespace) -> None:
    if args.official_test and stage not in {'cache_paper_rgb', 'train_paper_rgb', 'build_modal_tokens', 'predict_test', 'evaluate_test'}:
        raise RuntimeError(f'--official-test is not valid for {stage}')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--encoder', action='append', choices=('meanpool', 'deepsets'))
    parser.add_argument('--geometry-id', action='append', choices=('G1', 'G1S', 'G2', 'G2S', 'G3', 'G3S', 'G4', 'G4S', 'G5', 'G5S'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--official-test', action='store_true')
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    _not_test(args.stage, args)
    if args.stage == 'preflight':
        from celllift.geometry_baselines.preflight import full_preflight
        result = full_preflight(cfg)
        atomic_json(Path(cfg['paths']['data_root']) / '00_manifest' / 'preflight.json', result)
    elif args.stage == 'cache_paper_rgb':
        from celllift.geometry_baselines.paper_rgb import cache_paper_rgb
        labels = Path(cfg['paths']['model_input_root']) / '04_labels_splits' / 'labels_splits.parquet'
        destination = _cache_root(cfg) / 'official_test' if args.official_test else _cache_root(cfg)
        result = cache_paper_rgb(labels, destination, include_test=args.official_test, result_root=cfg['paths']['result_root'], shards=16)
    elif args.stage == 'train_paper_rgb':
        from celllift.geometry_baselines.paper_rgb import train_paper_rgb
        if args.seed is None or (not args.official_test and args.fold is None):
            raise ValueError('train_paper_rgb requires --seed and validation requires --fold')
        result = train_paper_rgb(dataset=cfg['dataset'], fold=args.fold, seed=args.seed, cache_root=_cache_root(cfg), result_root=cfg['paths']['result_root'], workers=args.workers, official_test=args.official_test)
    elif args.stage in {'build_modal_tokens', 'build_shuffle_maps'}:
        if args.fold is None:
            raise ValueError(f'{args.stage} requires --fold')
        if args.official_test:
            if args.stage != 'build_modal_tokens':
                raise ValueError('official shuffle maps are generated with each fold/seed prediction job')
            from celllift.geometry_baselines.official_tokens import build_official_fold_tokens
            result = build_official_fold_tokens(cfg, args.fold, device=args.device)
        else:
            from celllift.geometry_baselines.geometry_training import audit_geometry_store
            result = audit_geometry_store(cfg, args.fold, seed=int(args.seed or 42))
    elif args.stage in {'train_patient_geometry', 'predict_geometry'}:
        from celllift.geometry_baselines.geometry_training import train_patient_geometry
        if args.fold is None:
            raise ValueError(f'{args.stage} requires --fold')
        result = train_patient_geometry(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=None if args.seed is None else [args.seed], geometry_ids=args.geometry_id)
    elif args.stage == 'freeze_selection':
        from celllift.geometry_baselines.selection import freeze_selection
        result = freeze_selection(cfg)
    elif args.stage == 'evaluate_validation':
        from celllift.geometry_baselines.validation import evaluate_validation
        result = evaluate_validation(cfg)
    elif args.stage == 'train_tile_geometry':
        from celllift.geometry_baselines.tile_training import train_tile_geometry
        if args.fold is None:
            raise ValueError('train_tile_geometry requires --fold')
        result = train_tile_geometry(cfg, args.fold, device=args.device, encoders=args.encoder, seeds=None if args.seed is None else [args.seed], geometry_ids=args.geometry_id)
    elif args.stage in {'predict_paper_rgb', 'predict_test', 'evaluate_test', 'report'}:
        if args.stage in {'predict_test', 'evaluate_test'}:
            require_test_gate(cfg['paths']['result_root'])
        from celllift.geometry_baselines.workflow import run_stage
        result = run_stage(args.stage, cfg, args)
    else:
        raise AssertionError(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
