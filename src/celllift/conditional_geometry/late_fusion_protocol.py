from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'complete_geometry_probability_late_fusion_late_geometry_fusion'
RANK_PROTOCOL_ID = 'complete_geometry_rank_late_fusion_rank_fusion_crc'
RGB_MASK_PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_rgb_mask_conditioning_rgb_nucleus_mask36'
GEOMETRY_PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_mask_conditioning_nucleus_mask36_only'
BASE_ARM = 'M3_MASK2D'
GEOMETRY_ARMS = ('M3_MASK2D', 'M3_MASK_RES3D', 'M3_MASK_SHUF_RES3D')
PRIMARY_WEIGHT = 0.25
SENSITIVITY_WEIGHTS = tuple((index / 20 for index in range(11)))
OFFICIAL_TEST_ALLOWED = False
