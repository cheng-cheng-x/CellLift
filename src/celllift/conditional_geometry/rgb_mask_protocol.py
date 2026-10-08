from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.conditional_geometry.mask_protocol import CONFIRM_SEEDS, ENCODERS, INNER_FOLDS, MASK_CONTEXT_DIM, MASK_RESIDUAL_TOKEN_DIM, RAY_DIM, TARGET_COLUMNS, TARGET_DIM, stable_patient_fold
RGB_MASK_PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_rgb_mask_conditioning_rgb_nucleus_mask36'
RGB_DIM = 512
RGB_MASK_CONTEXT_DIM = RGB_DIM + MASK_CONTEXT_DIM
RGB_MASK_ARMS = ('M3_MASK2D', 'M3_MASK_RES3D', 'M3_MASK_SHUF_RES3D')
OFFICIAL_TEST_ALLOWED = False
PROBE_SEED = 20260823
