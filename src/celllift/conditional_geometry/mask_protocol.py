from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
MASK_PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_mask_conditioning_nucleus_mask36_only'
OFFICIAL_TEST_ALLOWED = False
PROBE_SEED = 20260823
INNER_FOLDS = 3
RAY_DIM = 36
MASK_CONTEXT_DIM = 73
TARGET_COLUMNS = ('nucleus_log_volume', 'nucleus_log_a_b', 'nucleus_log_b_c', 'nucleus_long_axis_z_squared', 'cell_log_volume', 'cell_log_a_b', 'cell_log_b_c', 'cell_long_axis_z_squared', 'log_ncr_3d')
TARGET_DIM = len(TARGET_COLUMNS)
MASK_RESIDUAL_TOKEN_DIM = 41
MASK_ARMS = ('M3_MASK2D', 'M3_MASK_RES3D', 'M3_MASK_SHUF_RES3D')
ENCODERS = ('meanpool', 'deepsets')
CONFIRM_SEEDS = (17, 42, 73, 101, 137)

def stable_patient_fold(patient_id: str, outer_fold: int, folds: int=INNER_FOLDS) -> int:
    payload = f'{MASK_PROTOCOL_ID}|outer={outer_fold}|patient={patient_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % int(folds)
