from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'protected_raw_scalar_logit_fusion_scalar_fusion_sicap'
OFFICIAL_TEST_ALLOWED = False
DATASET = 'sicapv2'
PRIMARY_ENCODER = 'meanpool'
SECONDARY_ENCODERS = ('deepsets',)
BASE_ARM = 'M3_MASK2D'
GEOMETRY_ARMS = {'R1_2D': 'M3_MASK2D', 'R2_REAL3D': 'M3_MASK_RES3D', 'R3_SHUF3D': 'M3_MASK_SHUF_RES3D'}
ALPHA_BOUNDS = (0.0, 4.0)
OPTIMIZER_TOLERANCE = 1e-10
OPTIMIZER_MAX_ITERATIONS = 1000
PRIMARY_COMPARISONS = ('real3d_vs_2d', 'real3d_vs_shuffled3d', 'real3d_vs_rgb_mask_base')
