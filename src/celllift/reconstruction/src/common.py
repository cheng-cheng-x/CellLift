from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from functools import lru_cache
from celllift.runtime import json
import hashlib
import os
import socket
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(_resource_path('artifact_0062'))
FAST = 'prostate_serial_roi1024_native_cellpose3_dual_cross_layer_set_encoding'
FAMILY = PROJECT / _public_resource('artifact_0063') / FAST / 'serial_training'
from celllift.runtime import output_root
RESULT = output_root() / 'reconstruction'

def paths():
    return {k: Path(v) for k, v in json.loads((ROOT / 'configs/data.json').read_text()).items()}

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)

def atomic_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)

@lru_cache(maxsize=1)
def digest():
    return hashlib.sha256(b''.join((str(p.relative_to(ROOT)).encode() + p.read_bytes() for p in sorted(ROOT.rglob('*')) if p.suffix in ('.py', '.json') and '__pycache__' not in str(p)))).hexdigest()

def record(**extra):
    value = dict(time=time.time(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=digest())
    value.update(extra)
    return value

def layout():
    for name in ('01_geometry', '02_inputs', '03_training', '04_predictions', '05_evaluation', '06_figures', 'logs', 'runtime'):
        (RESULT / name).mkdir(parents=True, exist_ok=True)
