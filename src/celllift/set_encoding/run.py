from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.experiment import CONFIRM_SEEDS, ENCODERS, SCREENING_PROTOCOL_ID, confirmation_arms, screening_arms
STAGES = ('preflight', 'infer_dual_geometry', 'build_nucleus_tokens', 'build_cell_tokens', 'build_ncr', 'extract_rgb', 'screen', 'confirm', 'evaluate', 'report')

def read_config(path: Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError('configuration root must be a mapping')
    required = {'dataset', 'paths', 'upstream', 'data', 'split', 'training', 'runtime'}
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f'configuration is missing {missing}')
    if value['dataset'] not in {'sicapv2', 'tcga_crc_msi'}:
        raise ValueError('unsupported dataset')
    if value['upstream'].get('inference_dtype') != 'float32':
        raise ValueError('frozen reconstruction inference must remain float32')
    if value['upstream'].get('pose_evidence') != 'empty':
        raise ValueError('public inference must use empty pose evidence')
    if tuple(value['runtime'].get('confirm_seeds', ())) != CONFIRM_SEEDS:
        raise ValueError('confirmation seed list changed')
    return value

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _paths(cfg: dict[str, Any]) -> dict[str, Path]:
    model_input = Path(cfg['paths']['model_input_root'])
    data_root = Path(cfg['paths']['data_root'])
    return {'graph_index': model_input / '03_graph_cache' / 'graph_index.parquet', 'graph_root': model_input / '03_graph_cache', 'patch_manifest': model_input / '00_manifest' / 'patch_manifest.parquet', 'data_root': data_root, 'geometry': data_root / '01_dual_geometry', 'pilot_geometry': data_root / '06_qc' / 'pilot_dual_geometry', 'rgb': data_root / '05_rgb_features', 'qc': data_root / '06_qc', 'result': Path(cfg['paths']['result_root'])}

def run_preflight(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from celllift.set_encoding.scripts.preflight import preflight_compatibility_score_public
    paths = _paths(cfg)
    paths['qc'].mkdir(parents=True, exist_ok=True)
    return preflight_compatibility_score_public(dataset_name=cfg['dataset'], graph_index=paths['graph_index'], graph_root=paths['graph_root'], training_code_root=cfg['paths']['upstream_code_root'], checkpoint=cfg['paths']['checkpoint'], graph_shards=int(cfg['data']['graph_shards']), smoke_count=int(args.limit or cfg['runtime']['pilot_graphs']), device=args.device, expected_checkpoint_sha256=cfg['upstream']['checkpoint_sha256'], output=paths['qc'] / f"preflight_{args.device.replace(':', '_')}.json")

def run_geometry(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from celllift.set_encoding.common.dual_geometry import infer_dual_geometry
    paths = _paths(cfg)
    output = paths['pilot_geometry'] if args.limit else paths['geometry']
    return infer_dual_geometry(dataset_name=cfg['dataset'], graph_index=paths['graph_index'], graph_root=paths['graph_root'], training_code_root=cfg['paths']['upstream_code_root'], checkpoint=cfg['paths']['checkpoint'], output_dir=output, graph_shards=int(cfg['data']['graph_shards']), limit=args.limit, workers=args.workers, device=args.device, expected_checkpoint_sha256=cfg['upstream']['checkpoint_sha256'])

def run_features(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from celllift.set_encoding.scripts.build_feature_cache import build_feature_cache
    paths = _paths(cfg)
    geometry = paths['pilot_geometry'] if args.pilot else paths['geometry']
    destination = paths['data_root'] / 'pilot' if args.pilot else paths['data_root']
    return build_feature_cache(geometry, destination, patch_width_um=float(cfg['data']['patch_width_um']), patch_height_um=float(cfg['data']['patch_height_um']), section_thickness_um=float(cfg['data']['section_thickness_um']), workers=args.workers)

def run_rgb(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from celllift.set_encoding.scripts.extract_rgb import extract_rgb
    if args.fold is None:
        raise ValueError('extract_rgb requires --fold')
    return extract_rgb(cfg, fold=args.fold, seed=42 if args.seed is None else args.seed, include_test=args.official_test, train_baseline=not args.official_test, device=args.device, num_workers=args.workers)

def _training_encoders(cfg: dict[str, Any], args: argparse.Namespace, *, confirmation: bool) -> tuple[str, ...]:
    if not confirmation:
        return (args.set_encoder,) if args.set_encoder else ENCODERS
    selection_path = Path(cfg['paths']['result_root']) / 'screening_selection.json'
    if not selection_path.is_file():
        raise RuntimeError('confirmation is locked until screening_selection.json exists')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selection_metric_aggregation') != 'arithmetic_mean_of_fold_metrics_set_encoding':
        raise RuntimeError('screening selection does not use registered fold-mean aggregation')
    retained = tuple(selection.get('retained_encoders', ()))
    if selection.get('status') != 'PASS' or len(retained) != 2:
        raise RuntimeError('screening selection is not a valid two-encoder PASS gate')
    if selection.get('protocol_id') != SCREENING_PROTOCOL_ID:
        raise RuntimeError('screening selection uses an incompatible downstream protocol')
    if len(set(retained)) != 2 or any((encoder not in ENCODERS for encoder in retained)):
        raise RuntimeError('screening selection contains invalid retained encoders')
    if args.set_encoder is not None:
        if args.set_encoder not in retained:
            raise ValueError(f'encoder {args.set_encoder!r} was not retained by validation screening')
        return (args.set_encoder,)
    return retained

def _write_confirmation_fold_marker(cfg: dict[str, Any], *, fold: int, encoders: tuple[str, ...], seeds: tuple[int, ...], arm_count: int, result: dict[str, Any]) -> Path:
    expected_jobs = len(encoders) * len(seeds) * int(arm_count)
    if result.get('status') != 'PASS' or int(result.get('jobs', -1)) != expected_jobs:
        raise RuntimeError('confirmation fold result is incomplete; refusing completion marker')
    path = Path(cfg['paths']['result_root']) / 'runtime' / f'confirmation_fold_{fold}_complete.json'
    _atomic_json(path, {'status': 'PASS', 'dataset': cfg['dataset'], 'fold': int(fold), 'retained_encoders': list(encoders), 'seeds': list(seeds), 'arm_count': int(arm_count), 'jobs': expected_jobs, 'protocol_id': SCREENING_PROTOCOL_ID})
    return path

def _write_official_test_fold_marker(cfg: dict[str, Any], *, fold: int, encoders: tuple[str, ...], seeds: tuple[int, ...], arm_count: int, result: dict[str, Any]) -> Path:
    expected_jobs = len(encoders) * len(seeds) * int(arm_count)
    if result.get('status') != 'PASS' or int(result.get('jobs', -1)) != expected_jobs:
        raise RuntimeError('official TEST fold result is incomplete; refusing completion marker')
    path = Path(cfg['paths']['result_root']) / 'runtime' / f'official_test_fold_{fold}_complete.json'
    _atomic_json(path, {'status': 'PASS', 'dataset': cfg['dataset'], 'fold': int(fold), 'retained_encoders': list(encoders), 'seeds': list(seeds), 'arm_count': int(arm_count), 'jobs': expected_jobs, 'protocol_id': SCREENING_PROTOCOL_ID, 'selection_gate': str(Path(cfg['paths']['result_root']) / 'selection_frozen.json')})
    return path

def run_training(cfg: dict[str, Any], args: argparse.Namespace, *, confirmation: bool) -> dict[str, Any]:
    from celllift.set_encoding.scripts.run_downstream import run_grid, run_official_test
    arms = confirmation_arms() if confirmation else screening_arms()
    seeds = CONFIRM_SEEDS if confirmation else (42,)
    selected = [arm for arm in arms if args.arm is None or arm.arm_id == args.arm]
    encoders = _training_encoders(cfg, args, confirmation=confirmation)
    if not selected:
        raise ValueError(f'unknown arm {args.arm!r}')
    if args.official_test:
        if not confirmation:
            raise ValueError('official TEST inference is only available through confirm')
        official_seeds = seeds if args.seed is None else (args.seed,)
        result = run_official_test(cfg, selected, encoders=encoders, seeds=official_seeds, fold=args.fold, device=args.device)
        if args.fold is not None:
            _write_official_test_fold_marker(cfg, fold=args.fold, encoders=tuple(encoders), seeds=tuple(official_seeds), arm_count=len(selected), result=result)
        return result
    result = run_grid(cfg, selected, encoders=encoders, seeds=seeds if args.seed is None else (args.seed,), fold=args.fold, validation_only=not confirmation, device=args.device)
    if confirmation and args.fold is not None:
        _write_confirmation_fold_marker(cfg, fold=args.fold, encoders=tuple(encoders), seeds=tuple(seeds if args.seed is None else (args.seed,)), arm_count=len(selected), result=result)
    return result

def run_evaluation(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from celllift.set_encoding.scripts.evaluate_results import evaluate_results
    return evaluate_results(cfg, official_test=args.official_test)

def run_report(cfg: dict[str, Any]) -> dict[str, Any]:
    from celllift.set_encoding.scripts.evaluate_results import write_report
    return write_report(cfg)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', choices=('sicapv2', 'tcga_crc_msi'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm')
    parser.add_argument('--set-encoder', choices=ENCODERS)
    parser.add_argument('--official-test', action='store_true')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = read_config(args.config)
    if args.dataset is not None and args.dataset != cfg['dataset']:
        raise ValueError('--dataset does not match configuration')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    if args.stage == 'preflight':
        result = run_preflight(cfg, args)
    elif args.stage == 'infer_dual_geometry':
        result = run_geometry(cfg, args)
    elif args.stage in {'build_nucleus_tokens', 'build_cell_tokens', 'build_ncr'}:
        result = run_features(cfg, args)
    elif args.stage == 'extract_rgb':
        result = run_rgb(cfg, args)
    elif args.stage == 'screen':
        result = run_training(cfg, args, confirmation=False)
    elif args.stage == 'confirm':
        result = run_training(cfg, args, confirmation=True)
    elif args.stage == 'evaluate':
        result = run_evaluation(cfg, args)
    else:
        result = run_report(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
