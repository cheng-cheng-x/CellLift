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
from celllift.feature_fusion.protocol import ARMS, PRIMARY_ARMS, PROTOCOL_ID, RAW_ARMS, validate_config

def read_config(path: Path):
    from celllift.runtime import yaml
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    validate_config(cfg)
    return cfg

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'cache_rgb_features', 'screen', 'evaluate', 'statistics', 'report'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', action='append', choices=tuple(ARMS))
    parser.add_argument('--family', choices=('residual', 'raw'), default='residual')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    arms = tuple(args.arm or PRIMARY_ARMS)
    if args.stage == 'preflight':
        from celllift.feature_fusion.models import paired_parameter_count
        result = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'paired_parameters': paired_parameter_count(cfg['dataset']), 'validation_only': True, 'official_test_touched': False}
    elif args.stage == 'cache_rgb_features':
        if args.fold is None:
            raise ValueError('cache_rgb_features requires --fold')
        from celllift.feature_fusion.rgb_features import cache_fold_features
        result = cache_fold_features(cfg, fold=args.fold, seed=args.seed, workers=args.workers)
    elif args.stage == 'screen':
        if args.fold is None:
            raise ValueError('screen requires --fold')
        from celllift.feature_fusion.training import run_grid
        result = run_grid(cfg, fold=args.fold, seed=args.seed, arms=arms, device=args.device)
    elif args.stage == 'evaluate':
        from celllift.feature_fusion.evaluation import evaluate
        result = evaluate(cfg, seed=args.seed, arms=arms)
    elif args.stage == 'statistics':
        from celllift.feature_fusion.statistics import run_statistics
        comparisons = ('C3-R0', 'C3-C0', 'C3-C1', 'C3-C3S', 'C2-C0', 'C2-C2S', 'C1-C0', 'C1-C1S') if args.family == 'residual' else ('C5-R0', 'C5-C0', 'C5-C1', 'C5-C5S', 'C4-C0', 'C4-C4S')
        result = run_statistics(cfg, seed=args.seed, comparisons=comparisons, family=args.family)
    else:
        from celllift.feature_fusion.reporting import build_report
        result = build_report(cfg, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
