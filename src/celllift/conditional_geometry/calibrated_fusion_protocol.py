from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'protected_calibrated_logit_fusion_calibrated_fusion'
OFFICIAL_TEST_ALLOWED = False
PRIMARY_ENCODER = 'meanpool'
SECONDARY_ENCODERS = ('deepsets',)
BASE_ARM = 'M3_MASK2D'
GEOMETRY_ARMS = {'F1_2D': 'M3_MASK2D', 'F2_REAL3D': 'M3_MASK_RES3D', 'F3_SHUF3D': 'M3_MASK_SHUF_RES3D'}
ABLATION_ARM = 'F4_REAL3D_UNCALIBRATED_GATE'
TEMPERATURE_BOUNDS = (0.25, 4.0)
ALPHA_BOUNDS = (0.0, 4.0)
EPSILON = 1e-07
OPTIMIZER_TOLERANCE = 1e-10
OPTIMIZER_MAX_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260830
PRIMARY_COMPARISONS = ('real3d_vs_2d', 'real3d_vs_shuffled3d', 'real3d_vs_rgb_mask_base')
