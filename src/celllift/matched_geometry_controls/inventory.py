from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
from .protocol import DEEPSETS_SEEDS, MEANPOOL_SEEDS

def _passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('status') == 'PASS'
    except Exception:
        return False

def _baseline_dir(cfg: Mapping[str, Any], encoder: str, fold: int, seed: int) -> Path:
    root = Path(cfg['paths']['baseline_predictions'])
    if cfg['dataset'] == 'bracs':
        return root / f'experts/development/t7/{encoder}/E0_RGB_MASK2D/fold_{fold:02d}/seed_{seed}'
    return root / f'paper_baseline/fold_{fold:02d}/seed_{seed}'

def verify_oof_inventory(cfg: Mapping[str, Any]) -> dict[str, Any]:
    dataset = cfg['dataset']
    folds = int(cfg['split']['folds'])
    data_root = Path(cfg['paths']['data_root'])
    result_root = Path(cfg['paths']['result_root'])
    missing: list[str] = []
    modal_manifest = data_root / '03_modal_features/manifest.json'
    if not _passed(modal_manifest):
        missing.append(str(modal_manifest))
    for fold in range(folds):
        residual = data_root / f'03_modal_features/residual/fold_{fold:02d}/all_residuals.json'
        shuffle = data_root / f'04_shuffle_maps/fold_{fold:02d}/mapping.json'
        if not _passed(residual):
            missing.append(str(residual))
        if not residual.with_suffix('.pt').is_file():
            missing.append(str(residual.with_suffix('.pt')))
        if not _passed(shuffle):
            missing.append(str(shuffle))
        for arm in ('D', 'DS', 'R', 'RS'):
            meanpool = data_root / f'05_meanpool_cache/fold_{fold:02d}/{arm}/manifest.json'
            if not _passed(meanpool):
                missing.append(str(meanpool))
            for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
                for seed in seeds:
                    root = result_root / f'06_experts/{encoder}/{arm}/fold_{fold:02d}/seed_{seed}'
                    for name in ('best.pt', 'predictions.parquet'):
                        if not (root / name).is_file():
                            missing.append(str(root / name))
                    if not _passed(root / 'manifest.json'):
                        missing.append(str(root / 'manifest.json'))
        for encoder, seeds in (('meanpool', MEANPOOL_SEEDS), ('deepsets', DEEPSETS_SEEDS)):
            for seed in seeds:
                baseline = _baseline_dir(cfg, encoder, fold, seed)
                prediction = baseline / ('predictions.parquet' if dataset == 'bracs' else 'validation_predictions.parquet')
                if not prediction.is_file():
                    missing.append(str(prediction))
                if not (baseline / 'best.pt').is_file():
                    missing.append(str(baseline / 'best.pt'))
    for encoder in ('meanpool', 'deepsets'):
        summary = result_root / f'08_metrics/{encoder}/summary.json'
        predictions = result_root / f'08_metrics/{encoder}/oof_predictions.parquet'
        if not _passed(summary):
            missing.append(str(summary))
        if not predictions.is_file():
            missing.append(str(predictions))
    return {'status': 'PASS' if not missing else 'INCOMPLETE', 'dataset': dataset, 'missing_count': len(missing), 'missing': missing}

def require_all_oof(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    inventories = {name: verify_oof_inventory(cfg) for name, cfg in configs.items()}
    incomplete = {name: value for name, value in inventories.items() if value['status'] != 'PASS'}
    if incomplete:
        preview = {name: value['missing'][:8] for name, value in incomplete.items()}
        raise RuntimeError(f'protocol freeze refused: OOF inventory incomplete: {preview}')
    return {'status': 'PASS', 'datasets': inventories}
