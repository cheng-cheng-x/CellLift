from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
import sys
from celllift.runtime import json
import random
import time
import subprocess
from celllift.reconstruction.src.common import ROOT, RESULT, paths, layout, atomic_json, record

def main():
    layout()
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    workers = []
    start = time.time()
    for rank, gpu in enumerate(gpus):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        log = (RESULT / 'logs' / ('prepare_%d.log' % rank)).open('a')
        p = subprocess.Popen([sys.executable, '-m', 'celllift.reconstruction.scripts.prepare_full_part', str(rank), str(len(gpus))], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        workers.append((p, log))
    errors = []
    for p, f in workers:
        code = p.wait()
        f.close()
        if code:
            errors.append(dict(pid=p.pid, exit_code=code))
    if errors:
        raise RuntimeError(str(errors))
    items = sum([json.loads((RESULT / '02_inputs' / ('part_%d.json' % rank)).read_text()) for rank in range(len(gpus))], [])
    items.sort(key=lambda x: (x['split'], x['uid']))
    counts = {s: sum((x['split'] == s for x in items)) for s in ['train', 'val']}
    assert counts == dict(train=8813, val=1102), counts
    for split in counts:
        for index, item in enumerate((x for x in items if x['split'] == split)):
            item['index'] = index
    cfg = json.loads((ROOT / 'configs/experiment.json').read_text())
    monitor = [x['uid'] for x in items if x['split'] == 'val']

    def stats(rows):
        a = sum((x['moment'][0] for x in rows))
        b = sum((x['moment'][1] for x in rows))
        n = sum((x['moment'][2] for x in rows))
        return dict(ray_mean_um=a / n, ray_std_um=(b / n - (a / n) ** 2) ** 0.5)
    train = [x for x in items if x['split'] == 'train']
    info = dict(protocol='train_val_test', items=items, monitor=monitor, counts=counts, fit_stats=stats([x for x in train if x['uid'] not in set(monitor)]), full_stats=stats(train), config=cfg)
    atomic_json(paths()['cache'] / 'manifest.json', dict(items=items, **info['full_stats']))
    atomic_json(RESULT / '02_inputs/manifest.json', info)
    atomic_json(RESULT / 'runtime/prepare_done.json', record(counts=counts, monitor=len(monitor), seconds=time.time() - start))
if __name__ == '__main__':
    main()
