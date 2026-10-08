from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from celllift.conditional_geometry.protocol import PROTOCOL_ID
SCREENING_PROTOCOL_ID = PROTOCOL_ID

@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    object_mode: str = 'both'
    geometry_mode: str = '2d'
    ncr_mode: str = '2d'
    use_rgb: bool = True
    control_mode: str = 'none'

    def to_dict(self) -> dict:
        return asdict(self)
ARMS = {'conditional_geometry_2D': ExperimentArm('conditional_geometry_2D'), 'conditional_geometry_RES3D': ExperimentArm('conditional_geometry_RES3D', geometry_mode='residual3d'), 'conditional_geometry_SHUF_RES3D': ExperimentArm('conditional_geometry_SHUF_RES3D', geometry_mode='residual3d', control_mode='shuffled_residual')}

def residual_arms() -> tuple[ExperimentArm, ...]:
    return tuple((ARMS[name] for name in ('conditional_geometry_2D', 'conditional_geometry_RES3D', 'conditional_geometry_SHUF_RES3D')))
