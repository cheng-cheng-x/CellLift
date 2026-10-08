from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
from typing import Literal
DatasetName = Literal['sicapv2', 'tcga_crc_msi']
EncoderName = Literal['meanpool', 'deepsets', 'set_transformer']
ObjectMode = Literal['rgb', 'nucleus', 'cell', 'both']
GeometryMode = Literal['2d', '3d']
NCRMode = Literal['none', '2d', '3d']
ControlMode = Literal['none', 'nucleus_duplicate', 'cell_duplicate', 'shuffled_cell_original_ncr', 'shuffled_cell_recomputed_ncr', 'shuffled_ncr']
SCREEN_SEED = 42
CONFIRM_SEEDS = (17, 42, 73, 101, 137)
ENCODERS: tuple[EncoderName, ...] = ('meanpool', 'deepsets')
RETIRED_ENCODERS: tuple[EncoderName, ...] = ('set_transformer',)
SCREENING_PROTOCOL_ID = 'paired_job_seed_count_bucketed_conditional_geometry'

@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    use_rgb: bool
    object_mode: ObjectMode
    geometry_mode: GeometryMode
    ncr_mode: NCRMode = 'none'
    control_mode: ControlMode = 'none'
    description: str = ''

    def __post_init__(self) -> None:
        if self.ncr_mode != 'none' and self.object_mode != 'both':
            raise ValueError('NCR is only legal for the nucleus+cell combined arm')
        if self.object_mode == 'rgb' and (not self.use_rgb):
            raise ValueError('object_mode=rgb requires use_rgb=true')
        if self.control_mode != 'none' and self.object_mode != 'both':
            raise ValueError('two-branch controls require object_mode=both')
        if self.control_mode != 'none' and self.ncr_mode != '3d':
            raise ValueError('pre-registered controls use NCR3D')

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def screening_arms() -> tuple[ExperimentArm, ...]:
    return (ExperimentArm('S0', True, 'rgb', '2d', description='RGB only'), ExperimentArm('S1', True, 'nucleus', '2d', description='RGB+nucleus2D'), ExperimentArm('S2', True, 'nucleus', '3d', description='RGB+nucleus3D'), ExperimentArm('S3', True, 'cell', '2d', description='RGB+cell2D'), ExperimentArm('S4', True, 'cell', '3d', description='RGB+cell3D'), ExperimentArm('S5', True, 'both', '2d', 'none', description='RGB+nucleus2D+cell2D'), ExperimentArm('S6', True, 'both', '3d', 'none', description='RGB+nucleus3D+cell3D'), ExperimentArm('S7', True, 'both', '2d', '2d', description='RGB+nucleus2D+cell2D+NCR2D'), ExperimentArm('S8', True, 'both', '3d', '3d', description='RGB+nucleus3D+cell3D+NCR3D'), ExperimentArm('S9', False, 'both', '3d', '3d', description='nucleus3D+cell3D+NCR3D'))

def confirmation_arms() -> tuple[ExperimentArm, ...]:
    base = list(screening_arms()[:9])
    base.extend((ExperimentArm('C_N3', False, 'nucleus', '3d', description='nucleus3D without RGB'), ExperimentArm('C_C3', False, 'cell', '3d', description='cell3D without RGB'), ExperimentArm('C_B3_NCR3', False, 'both', '3d', '3d', description='both3D+NCR3D without RGB'), ExperimentArm('C_B2_NCR3', True, 'both', '2d', '3d', description='2D morphology with NCR3D'), ExperimentArm('C_B3_NCR2', True, 'both', '3d', '2d', description='3D morphology with NCR2D'), ExperimentArm('C_NN_NCR3', True, 'both', '3d', '3d', 'nucleus_duplicate', 'capacity-matched nucleus duplicate'), ExperimentArm('C_CC_NCR3', True, 'both', '3d', '3d', 'cell_duplicate', 'capacity-matched cell duplicate'), ExperimentArm('C_SHUF_CELL_ORIG', True, 'both', '3d', '3d', 'shuffled_cell_original_ncr', 'donor cell with target NCR'), ExperimentArm('C_SHUF_CELL_RECOMP', True, 'both', '3d', '3d', 'shuffled_cell_recomputed_ncr', 'donor cell with recomputed NCR'), ExperimentArm('C_SHUF_NCR', True, 'both', '3d', '3d', 'shuffled_ncr', 'sample-specific NCR control')))
    ids = [arm.arm_id for arm in base]
    if len(ids) != len(set(ids)):
        raise RuntimeError('duplicate confirmation arm id')
    return tuple(base)
PRIMARY_CONTRAST = ('S8', 'S7')
SECONDARY_CONTRASTS = (('C_B3_NCR2', 'S7', '3D morphology at fixed NCR2D'), ('S8', 'C_B3_NCR2', 'NCR3D at fixed 3D morphology'), ('S2', 'S1', 'nucleus 3D increment'), ('S4', 'S3', 'cell 3D increment'), ('S8', 'S6', 'NCR contribution in combined 3D'), ('S8', 'S2', 'full versus nucleus only'), ('S8', 'S4', 'full versus cell only'), ('S8', 'C_NN_NCR3', 'full versus nucleus duplicate'), ('S8', 'C_CC_NCR3', 'full versus cell duplicate'), ('S8', 'C_SHUF_CELL_RECOMP', 'full versus shuffled cell'), ('S8', 'C_SHUF_NCR', 'full versus shuffled NCR'))
