from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
import socket
import subprocess
import time
from celllift.runtime import ResourcePath as Path
from .constants import CODE_ROOT, RECONSTRUCT_PYTHON, RESULT_ROOT
from .train import ARMS
NEED_SCENE = {'G3', 'GR', 'G23', 'G2R', 'H23', 'AS', 'C3', 'N3', 'NR', 'N23', 'N2R', 'JS', 'JR', 'LH23', 'LA3'}
NEED_RESIDUAL = {'GR', 'G2R', 'NR', 'N2R'}
NEED_B = {'H2', 'H23', 'LH2', 'LH23'}
NEED_SPATIAL = {'C2', 'C3'}
ORDER = ['B_paper', 'G2', 'N2', 'B_dino', 'B_crop', 'G3', 'G23', 'G2R', 'GR', 'N3', 'N23', 'N2R', 'NR', 'J2', 'JS', 'JR', 'H2', 'H23', 'A2', 'AS', 'C2', 'C3', 'LH2', 'LH23', 'LA2', 'LA3']

def _arm_dir(arm: str) -> Path:
    return RESULT_ROOT / _route(arm) / arm / 'seed42'

def _metrics_ok(arm: str) -> bool:
    path = _arm_dir(arm) / 'metrics.json'
    if not path.is_file():
        return (_arm_dir(arm) / 'model.pt').is_file()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return False
    if payload.get('status') in {'NO_RGB', 'WAIT'}:
        return False
    history = payload.get('history') or []
    if history:
        loss = history[0].get('loss')
        if loss != loss or loss is None:
            return False
    return True

def _done(arm: str) -> bool:
    return (_arm_dir(arm) / 'model.pt').is_file() and (not _busy(arm)) and _metrics_ok(arm)

def _busy(arm: str) -> bool:
    claim = _arm_dir(arm) / 'claim.json'
    if not claim.is_file():
        return False
    try:
        payload = json.loads(claim.read_text(encoding='utf-8'))
        pid = int(payload.get('pid') or 0)
        host = str(payload.get('host') or '')
        local = socket.gethostname()
        root = _arm_dir(arm)
        metrics = root / 'metrics.json'
        if metrics.is_file():
            try:
                if isinstance(json.loads(metrics.read_text(encoding='utf-8')).get('epochs'), int):
                    return False
            except Exception:
                pass
        local_alive = bool(pid and Path(f'/proc/{pid}').exists())
        if host == local:
            return local_alive
        if not host and local_alive:
            return True
        mtime = claim.stat().st_mtime
        for path in root.glob('epoch_*.pt'):
            mtime = max(mtime, path.stat().st_mtime)
        for path in (metrics, root / 'model.pt'):
            if path.is_file():
                mtime = max(mtime, path.stat().st_mtime)
        return time.time() - mtime < 10800
    except Exception:
        return True
    return False

def _claim(arm: str, gpu: int, pid: int) -> None:
    dest = _arm_dir(arm)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'claim.json').write_text(json.dumps({'pid': pid, 'gpu': gpu, 'arm': arm, 'host': socket.gethostname()}), encoding='utf-8')

def _route(arm: str) -> str:
    dataset = 'lizard' if arm.startswith(('N', 'J', 'B_c', 'B_d', 'LH', 'LA')) or arm in {'B_crop', 'B_dino'} else 'arvaniti'
    if arm in {'B_paper', 'B_crop', 'B_dino'}:
        route = 'baseline'
    elif arm.startswith(('G', 'N')):
        route = 'geom'
    elif arm.startswith('J'):
        route = 'relation'
    elif arm in {'H2', 'H23', 'LH2', 'LH23'}:
        route = 'fusion'
    elif arm in {'A2', 'AS', 'LA2', 'LA3'}:
        route = 'interact'
    else:
        route = 'spatial'
    return f'{dataset}/{route}'

def _data_root() -> Path:
    return Path(_resource_path('artifact_0047'))

def _scenes_ready(arm: str) -> bool:
    if arm not in NEED_SCENE:
        return True
    root = _data_root()
    if arm.startswith(('N', 'J', 'LH', 'LA')) or arm in {'B_crop', 'B_dino'}:
        folder = root / 'lizard' / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
        return folder.is_dir() and sum((1 for _ in folder.glob('*.pt'))) >= 1000
    folder = root / 'arvaniti' / '04_projection_scene_inputs_conditional_geometry' / 'selected_scene'
    return folder.is_dir() and sum((1 for _ in folder.glob('*.pt'))) >= 20000

def _residual_ready(arm: str) -> bool:
    if arm not in NEED_RESIDUAL:
        return True
    root = _data_root()
    folder = root / ('lizard' if arm.startswith('N') else 'arvaniti') / '04_projection_scene_inputs_conditional_geometry' / 'residual'
    if not folder.is_dir() or not (folder / 'probe.json').is_file():
        return False
    n = sum((1 for _ in folder.glob('*.npz')))
    return n >= (1000 if arm.startswith('N') else 40000)

def _spatial_ready(arm: str) -> bool:
    if arm not in NEED_SPATIAL:
        return True
    folder = _data_root() / 'arvaniti' / '04_projection_scene_inputs_conditional_geometry' / 'dino_spatial'
    return folder.is_dir() and sum((1 for _ in folder.glob('*.npy'))) >= 1000

def _b_ready(arm: str) -> bool:
    if arm not in NEED_B:
        return True
    if arm.startswith('LH'):
        return (RESULT_ROOT / 'lizard/baseline/B_crop/seed42/model.pt').is_file()
    return (RESULT_ROOT / 'arvaniti/baseline/B_paper/seed42/model.pt').is_file()

def free_gpus(host_smi: str | None=None) -> list[int]:
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free = []
    for line in raw.strip().splitlines():
        idx, mem, util = [part.strip() for part in line.split(',')]
        if float(mem) < 20000 and float(util) < 50:
            free.append(int(idx))
    return free

def launch_ready(reserved: list[int] | None=None) -> dict:
    log_root = RESULT_ROOT / 'logs'
    log_root.mkdir(parents=True, exist_ok=True)
    launched = []
    pending = [a for a in ORDER if not _done(a) and (not _busy(a)) and _scenes_ready(a) and _residual_ready(a) and _b_ready(a) and _spatial_ready(a)]
    reserved = set(reserved or [])
    free = [gpu for gpu in free_gpus() if gpu not in reserved]
    for gpu, arm in zip(free, pending):
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': f'{CODE_ROOT}:{CODE_ROOT}/model_inputs', 'PYTHONUNBUFFERED': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1'})
        log = log_root / f'train_{arm}.log'
        handle = log.open('w', encoding='utf-8')
        proc = subprocess.Popen([RECONSTRUCT_PYTHON, '-m', 'core_and_nucleus_prediction.run', 'train', '--arm', arm, '--device', 'cuda'], cwd=str(CODE_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        _claim(arm, gpu, proc.pid)
        launched.append({'arm': arm, 'gpu': gpu, 'pid': proc.pid})
    return {'launched': launched, 'done': [arm for arm in ORDER if _done(arm)]}

def loop(interval: int=90) -> None:
    while True:
        payload = launch_ready()
        (RESULT_ROOT / 'status' / 'worker.json').parent.mkdir(parents=True, exist_ok=True)
        (RESULT_ROOT / 'status' / 'worker.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        if len(payload['done']) == len(ORDER):
            return
        time.sleep(interval)
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'loop':
        loop()
    else:
        print(json.dumps(launch_ready(), indent=2))
