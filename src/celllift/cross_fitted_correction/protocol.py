from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
PROTOCOL_ID = 'paper_rgb_oof_geometry_correction_cross_fitted_correction'
TOKEN_DIM = 41
SEEDS = (17, 42, 73, 101, 137)
POOLERS = ('deepsets', 'multistat', 'balanced_multistat', 'gated_attention')
FUSIONS = ('correction', 'correction_gate')

@dataclass(frozen=True)
class CorrectionArm:
    arm_id: str
    geometry_id: str
    description: str
ARMS = {'B1': CorrectionArm('B1', 'G1', 'RGB + MASK2D'), 'B1S': CorrectionArm('B1S', 'G1S', 'RGB + shuffled MASK2D'), 'B2': CorrectionArm('B2', 'G4', 'RGB + residual3D'), 'B2S': CorrectionArm('B2S', 'G4S', 'RGB + shuffled residual3D'), 'B3': CorrectionArm('B3', 'G5', 'RGB + MASK2D + residual3D'), 'B3S': CorrectionArm('B3S', 'G5S', 'RGB + MASK2D + shuffled residual3D'), 'B4': CorrectionArm('B4', 'G2', 'RGB + raw3D'), 'B4S': CorrectionArm('B4S', 'G2S', 'RGB + shuffled raw3D'), 'B5': CorrectionArm('B5', 'G3', 'RGB + MASK2D + raw3D'), 'B5S': CorrectionArm('B5S', 'G3S', 'RGB + MASK2D + shuffled raw3D')}
PRIMARY_ARMS = ('B1', 'B1S', 'B2', 'B2S', 'B3', 'B3S')
RAW_SECONDARY_ARMS = ('B4', 'B4S', 'B5', 'B5S')

def assert_validation_only_path(path: str | Path) -> Path:
    value = Path(path)
    lowered = {part.lower() for part in value.parts}
    if 'official_test' in lowered or 'test' in lowered:
        raise RuntimeError(f'cross_fitted_correction is validation-only and refuses TEST paths: {value}')
    return value

def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('cross_fitted_correction protocol mismatch')
    if cfg.get('dataset') not in {'sicapv2', 'tcga_crc_msi'}:
        raise ValueError('unsupported dataset')
    assert_validation_only_path(cfg['paths']['result_root'])
    if 'official_test' in cfg or 'test_gate' in cfg:
        raise ValueError('cross_fitted_correction config must not contain an official TEST switch')
    training = cfg['training']
    if float(training['max_delta_logit']) <= 0 or float(training['delta_l2']) < 0:
        raise ValueError('invalid correction regularization')
