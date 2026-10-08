from celllift.runtime import resource_path as _public_resource
from pathlib import Path
import os, json, hashlib
import numpy as np
ROOT = Path(_public_resource('artifact_0048'))
PUBLIC = ROOT / _public_resource('artifact_0049')
RESULT = ROOT / _public_resource('artifact_0050')
DATA = Path(_public_resource('artifact_0051'))
OUT = Path(os.environ.get('DOWNSTREAM_ANALYSIS_OUTPUT', RESULT / 'downstream_3d_contribution_v1'))
NODE = ['nucleus_log_volume', 'nucleus_log_axis12', 'nucleus_log_axis23', 'nucleus_z_orientation', 'cell_log_volume', 'cell_log_axis12', 'cell_log_axis23', 'cell_z_orientation', 'log_nucleus_cytoplasm_ratio', 'offset_xy_relative', 'offset_z_relative', 'offset_xyz_relative']
EDGE = ['relative_axial_gap', 'nucleus_direction_spacing', 'cell_direction_spacing', 'spacing_3d_minus_2d', 'neighbor_abs_nucleus_log_volume', 'neighbor_abs_cell_log_volume', 'neighbor_abs_nucleus_axis', 'neighbor_abs_cell_axis']
DIST = ['nucleus_distance_3d_um', 'nucleus_distance_xy_um', 'nucleus_distance_excess_um', 'cell_distance_3d_um', 'cell_distance_xy_um', 'cell_distance_excess_um']
TWO = ['log_area', 'log_axis_ratio', 'xy_distance', 'xy_direction_spacing', 'neighbor_abs_log_area', 'neighbor_abs_log_axis']

def columns(names):
    return [n + '__' + s for n in names for s in ('median', 'iqr')]
PRIMARY = columns(NODE + EDGE)
DISTANCE = columns(DIST)
COVARIATES = ['log_node_count'] + columns(TWO[:4]) + ['valid3d_fraction']
TASKS = {'sicapv2': ['grading'], 'bracs': ['roi7'], 'tcga_crc_msi': ['msi'], 'tcga_brca': ['subtype4', 'luma_lumb', 'idc_ilc', 'er_ihc'], 'arvaniti': ['local_p1', 'local_p2', 'core_p1', 'core_p2'], 'lizard': ['nucleus6']}
LABELS = {'grading': ['NC', 'G3', 'G4', 'G5'], 'roi7': ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC'], 'msi': ['MSS', 'MSI'], 'subtype4': ['LumA', 'LumB', 'Her2-enriched', 'Basal-like'], 'luma_lumb': ['LumA', 'LumB'], 'idc_ilc': ['IDC', 'ILC'], 'er_ihc': ['Negative', 'Positive'], 'local_p1': ['benign', 'G3', 'G4', 'G5'], 'local_p2': ['benign', 'G3', 'G4', 'G5'], 'core_p1': ['benign', 'Gleason6', 'Gleason7', 'Gleason8', 'Gleason9', 'Gleason10'], 'core_p2': ['benign', 'Gleason6', 'Gleason7', 'Gleason8', 'Gleason9', 'Gleason10'], 'nucleus6': ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)

    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    tmp = p.with_suffix(p.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv))
    tmp.replace(p)

def stable(x):
    return int(hashlib.sha256(str(x).encode()).hexdigest()[:12], 16)

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ix = np.flatnonzero(np.isfinite(p))
    ix = ix[np.argsort(p[ix])]
    if len(ix):
        out[ix] = np.minimum(1, np.minimum.accumulate((p[ix] * len(ix) / np.arange(1, len(ix) + 1))[::-1])[::-1])
    return out
