from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from celllift.conditional_geometry.mask_protocol import MASK_PROTOCOL_ID
SCREENING_PROTOCOL_ID = MASK_PROTOCOL_ID

@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    object_mode: str = 'both'
    geometry_mode: str = 'mask2d'
    ncr_mode: str = 'none'
    use_rgb: bool = False
    control_mode: str = 'none'

    def to_dict(self) -> dict:
        return asdict(self)
ARMS = {'M3_MASK2D': ExperimentArm('M3_MASK2D'), 'M3_MASK_RES3D': ExperimentArm('M3_MASK_RES3D', geometry_mode='mask_residual3d'), 'M3_MASK_SHUF_RES3D': ExperimentArm('M3_MASK_SHUF_RES3D', geometry_mode='mask_residual3d', control_mode='shuffled_residual')}
