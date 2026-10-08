from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS

def run_official_stage(stage: str, cfg: Mapping[str, Any], args: Any, gate: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(cfg['paths']['result_root'])
    if gate.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('official TEST gate protocol mismatch')
    if stage == 'predict_test':
        if args.fold is not None:
            if args.seed is None:
                raise ValueError('fold-wise official geometry prediction requires --seed')
            from .official_tokens import build_official_fold_tokens
            from .official_geometry import predict_official_geometry_job
            build_official_fold_tokens(cfg, int(args.fold), device=args.device)
            encoders = tuple(args.encoder or ENCODERS)
            geometry_ids = tuple(args.geometry_id or ('G1', 'G1S', 'G2', 'G2S', 'G3', 'G3S', 'G4', 'G4S', 'G5', 'G5S'))
            jobs = []
            for encoder in encoders:
                for geometry_id in geometry_ids:
                    jobs.append(predict_official_geometry_job(cfg, int(args.fold), int(args.seed), encoder, geometry_id, device=args.device, tile_route=False))
                    if cfg['dataset'] == 'tcga_crc_msi':
                        jobs.append(predict_official_geometry_job(cfg, int(args.fold), int(args.seed), encoder, geometry_id, device=args.device, tile_route=True))
            return {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'phase': 'official_geometry_staging', 'jobs': len(jobs)}
        if args.seed is not None:
            from .paper_rgb import train_paper_rgb
            return train_paper_rgb(dataset=str(cfg['dataset']), fold=None, seed=int(args.seed), cache_root=Path(cfg['paths']['data_root']) / '01_rgb224_cache', result_root=root, workers=int(args.workers), official_test=True)
        from .official_evaluation import verify_official_staging
        return verify_official_staging(cfg)
    if stage == 'evaluate_test':
        transaction = root / 'official_test' / 'transaction_ready.json'
        if not transaction.is_file():
            raise RuntimeError('official TEST evaluation is locked until the complete label-blind transaction is ready')
        from .official_evaluation import evaluate_official_test
        return evaluate_official_test(cfg)
    raise ValueError(stage)
