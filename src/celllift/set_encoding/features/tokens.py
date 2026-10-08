from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from .geometry import Morphology2D, Morphology3D
from .ncr import NCRFeatures
TOKEN_COLUMNS = ('log_area', 'log_equivalent_radius', 'log_major_minor', 'circularity', 'cos_2theta', 'sin_2theta', 'x_normalized', 'y_normalized', 'border_flag', 'log_volume', 'log_a_b', 'log_b_c', 'long_axis_z_squared', 'standardized_log_ncr', 'invalid_ncr_flag')
TOKEN_DIM = len(TOKEN_COLUMNS)
MORPHOLOGY_3D_SLICE = slice(9, 13)
NCR_SLICE = slice(13, 15)

def build_object_tokens(morphology_2d: Morphology2D, positions_xy: np.ndarray, border_flag: np.ndarray, morphology_3d: Morphology3D, *, geometry_mode: str) -> np.ndarray:
    if geometry_mode not in {'2d', '3d'}:
        raise ValueError("geometry_mode must be '2d' or '3d'")
    positions = np.asarray(positions_xy, dtype=np.float32)
    border = np.asarray(border_flag, dtype=np.float32)
    count = len(morphology_2d.valid)
    if positions.shape != (count, 2) or border.shape != (count,):
        raise ValueError('positions_xy/border_flag must have shapes [N,2] and [N]')
    if morphology_3d.channels().shape != (count, 4):
        raise ValueError('2-D and 3-D morphology row counts differ')
    if not np.isfinite(positions).all() or not np.isfinite(border).all():
        raise ValueError('position channels must be finite')
    tokens = np.zeros((count, TOKEN_DIM), dtype=np.float32)
    tokens[:, :6] = morphology_2d.channels()
    tokens[:, 6:8] = positions
    tokens[:, 8] = border
    if geometry_mode == '3d':
        tokens[:, MORPHOLOGY_3D_SLICE] = morphology_3d.channels()
    if not np.isfinite(tokens).all():
        raise ValueError('tokens contain NaN or infinity')
    return tokens

def apply_paired_ncr_channels(nucleus_tokens: np.ndarray, cell_tokens: np.ndarray, ncr: NCRFeatures | None, *, object_mode: str, ncr_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if object_mode not in {'nucleus', 'cell', 'both', 'rgb'}:
        raise ValueError('unsupported object_mode')
    if ncr_mode not in {'none', '2d', '3d'}:
        raise ValueError('ncr_mode must be none, 2d or 3d')
    if object_mode != 'both' and ncr_mode != 'none':
        raise ValueError("NCR can be enabled only when object_mode='both'")
    nucleus = np.asarray(nucleus_tokens, dtype=np.float32).copy()
    cell = np.asarray(cell_tokens, dtype=np.float32).copy()
    if nucleus.ndim != 2 or cell.ndim != 2 or nucleus.shape[1:] != (TOKEN_DIM,) or (cell.shape[1:] != (TOKEN_DIM,)):
        raise ValueError('nucleus/cell tokens must have shape [N,15]')
    nucleus[:, NCR_SLICE] = 0.0
    cell[:, NCR_SLICE] = 0.0
    if ncr_mode == 'none':
        return (nucleus, cell)
    if ncr is None:
        raise ValueError('active ncr_mode requires fitted NCR features')
    if len(nucleus) != len(cell):
        raise ValueError('paired NCR requires equal anchor counts in the two object sets')
    channels = ncr.channels()
    if channels.shape != (len(nucleus), 2) or not np.isfinite(channels).all():
        raise ValueError('NCR channels must be finite with shape [N,2]')
    nucleus[:, NCR_SLICE] = channels
    cell[:, NCR_SLICE] = channels
    return (nucleus, cell)

def assert_token_schema(tokens: np.ndarray, *, geometry_mode: str, ncr_enabled: bool) -> None:
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM or (not np.isfinite(tokens).all()):
        raise ValueError('token cache must be finite with shape [N,15]')
    if geometry_mode == '2d' and np.any(tokens[:, MORPHOLOGY_3D_SLICE] != 0):
        raise ValueError('2-D arms must zero all four 3-D channels')
    if not ncr_enabled and np.any(tokens[:, NCR_SLICE] != 0):
        raise ValueError('NCR-off arms must zero both NCR channels')
