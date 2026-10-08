from celllift.runtime import resource_path as _public_resource
from pathlib import Path
import json, os, time, socket, hashlib
from functools import lru_cache
ROOT = Path(__file__).resolve().parent
from celllift.runtime import output_root
RESULT = output_root() / 'scene_metrics'
BASE = output_root() / 'geometric_baselines'
MODEL = output_root() / 'reconstruction'
MODEL_CODE = ROOT.parents[1] / 'reconstruction'
SCALES = (1.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.9)
CONFIG = dict(replicates=4, samples_per_body_per_replicate=512, seed=42, splits={'test': 1101}, volume_baseline_modes=['main'], model_modes=['main', 'soft'], scales=SCALES, intervention='Translate each nucleus and its cell together until nucleus center z=2.5; keep transforms fixed', estimator='Sum_i V_i/S * mean_{x uniform C_i}(1-1/multiplicity(x)); randomized Sobol, independent scrambles')

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

@lru_cache(None)
def digest():
    return hashlib.sha256(b''.join((p.relative_to(ROOT).as_posix().encode() + p.read_bytes() for p in sorted(ROOT.rglob('*')) if p.suffix in ('.py', '.json') and '__pycache__' not in str(p)))).hexdigest()

def record(**kwargs):
    return dict(time=time.time(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=digest(), **kwargs)

def source_protocol():
    x = dict(config=CONFIG, source_sha256=digest(), dependencies={})
    for name in ['src/geometry.py', 'src/projection_geometry.py', 'src/candidate_tables.py']:
        x['dependencies'][str(MODEL_CODE / name)] = hashlib.sha256((MODEL_CODE / name).read_bytes()).hexdigest()
    return x
