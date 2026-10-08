from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from celllift.conditional_geometry.full3d_protocol import PROTOCOL_ID
SCREENING_PROTOCOL_ID = PROTOCOL_ID

@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    object_mode: str = 'both'
    geometry_mode: str = 'mask_full3d'
    ncr_mode: str = '3d'
    use_rgb: bool = False
    control_mode: str = 'none'

    def to_dict(self) -> dict:
        return asdict(self)
ARMS = {'M5_MASK_FULL3D': ExperimentArm('M5_MASK_FULL3D'), 'M5_MASK_SHUF_FULL3D': ExperimentArm('M5_MASK_SHUF_FULL3D', control_mode='shuffled_full3d')}
