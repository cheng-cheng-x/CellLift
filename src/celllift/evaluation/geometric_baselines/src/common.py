from celllift.runtime import resource_path as _public_resource
from pathlib import Path
import os, time, socket, hashlib
from celllift.runtime import json
from functools import lru_cache
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(_public_resource('artifact_0048'))
from celllift.runtime import output_root
RESULT = output_root() / 'geometric_baselines'
CONFIG = json.loads((ROOT / 'configs/protocol.json').read_text())
PATHS = {k: Path(v) for k, v in json.loads((ROOT / 'configs/data.json').read_text()).items()}

def write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False))
    tmp.replace(path)

def save(path, x):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(x, tmp)
    tmp.replace(path)

@lru_cache(None)
def digest():
    return hashlib.sha256(b''.join((p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in sorted(ROOT.rglob('*')) if p.suffix in ('.py', '.json') and '__pycache__' not in str(p)))).hexdigest()

def record(**kw):
    out = dict(time=time.time(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=digest())
    out.update(kw)
    return out

def layout():
    for s in ['01_validation', '02_inputs', '03_calibration', '04_predictions', '05_evaluation', '06_figures', 'logs', 'runtime']:
        (RESULT / s).mkdir(parents=True, exist_ok=True)
