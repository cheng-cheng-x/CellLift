from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
PROTOCOL_ID = 'paper_rgb_feature_geometry_fusion_feature_fusion'
SEEDS = (17, 42, 73, 101, 137)
POOLER = 'multistat'
FUSION = 'residual_add'

@dataclass(frozen=True)
class FusionArm:
    arm_id: str
    geometry_id: str | None
    description: str
ARMS = {'C0': FusionArm('C0', None, 'frozen paper-RGB feature refit'), 'C1': FusionArm('C1', 'G1', 'RGB feature + MASK2D'), 'C1S': FusionArm('C1S', 'G1S', 'RGB feature + shuffled MASK2D'), 'C2': FusionArm('C2', 'G4', 'RGB feature + residual3D'), 'C2S': FusionArm('C2S', 'G4S', 'RGB feature + shuffled residual3D'), 'C3': FusionArm('C3', 'G5', 'RGB feature + MASK2D + residual3D'), 'C3S': FusionArm('C3S', 'G5S', 'RGB feature + MASK2D + shuffled residual3D'), 'C4': FusionArm('C4', 'G2', 'RGB feature + raw3D'), 'C4S': FusionArm('C4S', 'G2S', 'RGB feature + shuffled raw3D'), 'C5': FusionArm('C5', 'G3', 'RGB feature + MASK2D + raw3D'), 'C5S': FusionArm('C5S', 'G3S', 'RGB feature + MASK2D + shuffled raw3D')}
PRIMARY_ARMS = ('C0', 'C1', 'C1S', 'C2', 'C2S', 'C3', 'C3S')
RAW_ARMS = ('C4', 'C4S', 'C5', 'C5S')

def assert_validation_only_path(path: str | Path) -> Path:
    value = Path(path)
    if any((part.lower() in {'test', 'official_test'} for part in value.parts)):
        raise RuntimeError(f'feature_fusion is validation-only and refuses TEST paths: {value}')
    return value

def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get('protocol_id') != PROTOCOL_ID:
        raise ValueError('feature_fusion protocol mismatch')
    if cfg.get('dataset') not in {'sicapv2', 'tcga_crc_msi'}:
        raise ValueError('unsupported dataset')
    assert_validation_only_path(cfg['paths']['result_root'])
    if 'official_test' in cfg or 'test_gate' in cfg:
        raise ValueError('feature_fusion has no TEST switch')
    if int(cfg['training']['rgb_feature_dim']) != 512:
        raise ValueError('paper-RGB feature width must remain 512')
