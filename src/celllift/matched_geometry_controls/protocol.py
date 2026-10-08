from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
from celllift.runtime import json
from .io_utils import atomic_json, sha256_file
PROTOCOL_ID = 'matched_geometry_controls'
PROTOCOL_DATE = '2026-09-09'
SHUFFLE_SEED = 20260909
MEANPOOL_SEEDS = (17, 42, 73, 101, 137)
DEEPSETS_SEEDS = (42,)
ENCODERS = ('meanpool', 'deepsets')
DATASETS = ('sicapv2', 'tcga_crc_msi', 'bracs')
STAGE_DIRS = ('00_manifest', '01_projection_scene_inputs', '02_projection_scene_selected_scene', '03_modal_features', '04_shuffle_maps', '05_meanpool_cache', '06_experts', '07_fusion', '08_metrics', '09_official_test', 'runtime', 'logs')

@dataclass(frozen=True)
class Arm:
    arm_id: str
    geometry_mode: str
    shuffled: bool = False

    @property
    def has_geometry(self) -> bool:
        return self.geometry_mode != 'none'

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
ARMS: dict[str, Arm] = {'B0': Arm('B0', 'none'), 'D': Arm('D', 'direct'), 'DS': Arm('DS', 'direct', True), 'R': Arm('R', 'residual'), 'RS': Arm('RS', 'residual', True)}
DATASET_CONTRACTS: dict[str, dict[str, Any]] = {'sicapv2': {'folds': 4, 'group': 'patient_id', 'classes': 4, 'metric': 'patch_qwk', 'bootstrap': 'patient_cluster', 'expected_graphs': 12081}, 'tcga_crc_msi': {'folds': 5, 'group': 'patient_id', 'classes': 2, 'metric': 'mean_fold_patient_auroc', 'bootstrap': 'fold_stratified_patient', 'minimum_tiles': 10, 'expected_graphs': 51915, 'expected_zero_nuclei_exclusions': 3}, 'bracs': {'folds': 5, 'group': 'wsi_id', 'classes': 7, 'metric': 'roi_macro_f1', 'bootstrap': 'wsi_cluster', 'expected_graphs': 10186}}

def prepare_tree(root: str | Path) -> Path:
    root = Path(root)
    for name in STAGE_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root

def validate_job(dataset: str, arm: str, encoder: str, seed: int) -> None:
    if dataset not in DATASETS:
        raise ValueError(f'unregistered dataset: {dataset}')
    if arm not in ARMS:
        raise ValueError(f'unregistered arm: {arm}')
    if encoder not in ENCODERS:
        raise ValueError(f'unregistered encoder: {encoder}')
    allowed = MEANPOOL_SEEDS if encoder == 'meanpool' else DEEPSETS_SEEDS
    if int(seed) not in allowed:
        raise ValueError(f'unregistered {encoder} seed: {seed}')

def freeze_protocol(result_root: str | Path, *, hashes: Mapping[str, str], manifests: Mapping[str, str], config_paths: Mapping[str, str], inventory: Mapping[str, Any] | None=None) -> dict[str, Any]:
    root = prepare_tree(result_root)
    missing = [name for name, value in hashes.items() if not value]
    if missing:
        raise ValueError(f'empty locked hashes: {missing}')
    payload: dict[str, Any] = {'status': 'FROZEN', 'protocol_id': PROTOCOL_ID, 'protocol_date': PROTOCOL_DATE, 'arms': {key: value.as_dict() for key, value in ARMS.items()}, 'meanpool_seeds': list(MEANPOOL_SEEDS), 'deepsets_seeds': list(DEEPSETS_SEEDS), 'shuffle_seed': SHUFFLE_SEED, 'datasets': DATASET_CONTRACTS, 'hashes': dict(hashes), 'manifests': dict(manifests), 'configs': dict(config_paths), 'oof_inventory': None if inventory is None else dict(inventory), 'fusion': {'sicapv2': 'centered logits: z_B0 + alpha*z_G, alpha in [0,4], cross-fit by outer fold', 'bracs': 'centered logits: z_B0 + alpha*z_G, alpha in [0,4], cross-fit by outer fold', 'tcga_crc_msi': 'z_B0/T_B + alpha*z_G/T_G; temperatures in [0.25,4], alpha in [0,4]'}, 'test_outputs': {dataset: {'scopes': [f'fold_{fold:02d}' for fold in range(DATASET_CONTRACTS[dataset]['folds'])] if dataset == 'bracs' else ['final'], 'geometry_arms': ['D', 'DS', 'R', 'RS'], 'encoders': {'meanpool': {'seeds': list(MEANPOOL_SEEDS)}, 'deepsets': {'seeds': list(DEEPSETS_SEEDS)}}, 'baseline': 'train_val_final_fold_ensemble' if dataset == 'bracs' else 'reuse_locked_geometry_baselines_official_test', 'fused_predictions': ['B0', 'D', 'DS', 'R', 'RS'], 'metrics': ['meanpool', 'deepsets']} for dataset in DATASETS}, 'test_label_policy': {'source_read': 'once_after_both_encoder_prediction_inventories_are_complete', 'retry_source': 'compact_test_labels.parquet_only', 'artifact': '09_official_test/test_labels.parquet'}}
    destination = root / 'protocol_frozen.json'
    atomic_json(destination, payload)
    payload['sha256'] = sha256_file(destination)
    return payload

def require_frozen(result_root: str | Path) -> dict[str, Any]:
    path = Path(result_root) / 'protocol_frozen.json'
    if not path.is_file():
        raise RuntimeError('official TEST is disabled until protocol_frozen.json exists')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('status') != 'FROZEN' or payload.get('protocol_id') != PROTOCOL_ID:
        raise RuntimeError('official TEST freeze is absent or belongs to another protocol')
    return payload
