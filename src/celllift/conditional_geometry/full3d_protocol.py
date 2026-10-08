from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'matched_mask_full3d_vs_residual3d_raw_residual_comparison'
OFFICIAL_TEST_ALLOWED = False
SOURCE_PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_mask_conditioning_nucleus_mask36_only'
FULL3D_ARMS = ('M5_MASK_FULL3D', 'M5_MASK_SHUF_FULL3D')
ENCODERS = ('meanpool', 'deepsets')
SEEDS = (17, 42, 73, 101, 137)
ALPHA_BOUNDS = (0.0, 4.0)
TEMPERATURE_BOUNDS = (0.25, 4.0)
BASE_ARM = 'M3_MASK2D'
FUSION_ARMS = {'set_encoding_2D': ('residual', 'M3_MASK2D'), 'conditional_geometry_FULL3D': ('full', 'M5_MASK_FULL3D'), 'mask_conditioning_SHUF_FULL3D': ('full', 'M5_MASK_SHUF_FULL3D'), 'calibrated_fusion_RESIDUAL3D': ('residual', 'M3_MASK_RES3D'), 'raw_residual_comparison_SHUF_RESIDUAL3D': ('residual', 'M3_MASK_SHUF_RES3D')}
PRIMARY_COMPARISONS = ('full3d_vs_2d', 'full3d_vs_shuffled_full3d', 'full3d_vs_rgb_mask_base')
