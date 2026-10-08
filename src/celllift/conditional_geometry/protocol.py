from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
PROTOCOL_ID = 'conditional_3d_residual_patient_crossfit_conditional_geometry_valid_ncr_masked'
OFFICIAL_TEST_ALLOWED = False
PROBE_SEED = 20260822
INNER_FOLDS = 3
TARGET_COLUMNS = ('nucleus_log_volume', 'nucleus_log_a_b', 'nucleus_log_b_c', 'nucleus_long_axis_z_squared', 'cell_log_volume', 'cell_log_a_b', 'cell_log_b_c', 'cell_long_axis_z_squared', 'log_ncr_3d')
TARGET_DIM = len(TARGET_COLUMNS)
ANCHOR_2D_DIM = 20
CONTEXT_DIM = 38
RGB_DIM = 512
RESIDUAL_TOKEN_DIM = 16
RESIDUAL_ARMS = ('conditional_geometry_2D', 'conditional_geometry_RES3D', 'conditional_geometry_SHUF_RES3D')
ENCODERS = ('meanpool', 'deepsets')
CONFIRM_SEEDS = (17, 42, 73, 101, 137)

def stable_patient_fold(patient_id: str, outer_fold: int, folds: int=INNER_FOLDS) -> int:
    import hashlib
    payload = f'{PROTOCOL_ID}|outer={outer_fold}|patient={patient_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % folds
