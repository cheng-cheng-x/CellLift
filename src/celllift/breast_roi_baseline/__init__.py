from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'bracs_current_compatibility_score_direct_residual_breast_roi_scalar_fusion'
SEEDS = (17, 42, 73, 101, 137)
ENCODERS = ('meanpool', 'deepsets')
TASKS = ('t7', 't3')
EXPERTS = ('E0_RGB_MASK2D', 'E1_MASK2D', 'E2_MASK_DIRECT3D', 'E3_MASK_SHUF_DIRECT3D', 'E4_MASK_RESIDUAL3D', 'E5_MASK_SHUF_RESIDUAL3D')
