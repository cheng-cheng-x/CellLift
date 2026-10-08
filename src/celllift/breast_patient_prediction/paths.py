from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from .protocol import PROTOCOL_ID
REMOTE_DATA_ROOT = Path(_resource_path('artifact_0021'))
REMOTE_RESULT_ROOT = Path(_resource_path('artifact_0022'))
ROUTES_RESULT_ROOT = Path(_resource_path('artifact_0023'))
ROUTES_conditional_geometry_RESULT_ROOT = Path(_resource_path('artifact_0024'))
CELLPOSE_PYTHON = Path(_resource_path('artifact_0025'))
STAGE1_SLIDES = Path(_resource_path('artifact_0026'))
CLINICAL_PATIENT = Path(_resource_path('artifact_0027'))
BRCA_conditional_geometry_MANIFEST = Path(_resource_path('artifact_0028'))
BRACS_CANDIDATES = (Path(_resource_path('artifact_0014')), Path(_resource_path('artifact_0005')), Path(_resource_path('artifact_0029')))
PRETRAIN_PYTHON = Path(_resource_path('artifact_0008'))

def data_root(override: str | Path | None=None) -> Path:
    raw = override or os.environ.get('TCGA_BRCA_DATA_ROOT') or REMOTE_DATA_ROOT
    return Path(raw)

def source_dir(root: Path | None=None) -> Path:
    return data_root(root) / '00_sources'

def inventory_dir(root: Path | None=None) -> Path:
    return data_root(root) / '01_inventory'

def label_dir(root: Path | None=None) -> Path:
    return data_root(root) / '04_labels'

def split_dir(root: Path | None=None) -> Path:
    return data_root(root) / '05_splits'

def tile_dir(root: Path | None=None) -> Path:
    return data_root(root) / '06_tiles'

def audit_dir(root: Path | None=None) -> Path:
    return data_root(root) / '07_audit'

def result_root(override: str | Path | None=None) -> Path:
    raw = override or os.environ.get('TCGA_BRCA_RESULT_ROOT') or REMOTE_RESULT_ROOT
    return Path(raw)

def routes_result_root(override: str | Path | None=None) -> Path:
    raw = override or os.environ.get('TCGA_BRCA_ROUTES_RESULT_ROOT') or ROUTES_RESULT_ROOT
    return Path(raw)

def routes_set_encoding_result_root() -> Path:
    return ROUTES_RESULT_ROOT

def model_input_dir(root: Path | None=None) -> Path:
    return data_root(root) / '08_model_inputs' / 'set_encoding'

def ensure_tree(root: Path | None=None) -> Path:
    base = data_root(root)
    for folder in (source_dir(base), inventory_dir(base), label_dir(base), split_dir(base), tile_dir(base) / 'shards', audit_dir(base), model_input_dir(base)):
        folder.mkdir(parents=True, exist_ok=True)
    (base / 'protocol_id.txt').write_text(PROTOCOL_ID + '\n', encoding='utf-8')
    return base
