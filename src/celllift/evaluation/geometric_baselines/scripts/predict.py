from celllift.runtime import resource_path as _public_resource
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
import torch
from celllift.evaluation.geometric_baselines.src.common import RESULT, CONFIG, layout, write, save, record
from celllift.evaluation.geometric_baselines.src.data import rows, load
from celllift.evaluation.geometric_baselines.src.geometry import coefficients, decode, stats

@torch.no_grad()
def main(method):
    torch.set_num_threads(4)
    start = time.time()
    best = json.loads((RESULT / '03_calibration' / method / 'complete.json').read_text())['best']['parameters']
    done = 0
    for split in ['test']:
        rr = rows(split)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(load, rr):
                dest = RESULT / '04_predictions' / method / split / (item['uid'] + '.pt')
                if dest.exists():
                    done += 1
                    continue
                cf = coefficients(item['root'], method)
                bodies = {}
                for label, mult in [('short', 0.75), ('main', 1.0), ('long', 1.5)]:
                    params = [best[0] * mult, *best[1:]]
                    geo = decode(cf, params, method)
                    bodies[label] = geo
                    for kind, cap, ratio in [('nucleus', 800, 3), ('cell', 4000, 6)]:
                        st = stats(cf, geo, method, kind)
                        v = geo['valid']
                        assert (st['volume'][v] <= cap * (1 + 1e-08)).all() and (st['ratio'][v] <= ratio * (1 + 1e-08)).all()
                save(dest, dict(ids=item['ids'], root=item['root'], center=item['center'], metadata=item['metadata'], bodies=bodies, method=method, parameters=best))
                done += 1
                if done % 128 == 0:
                    write(RESULT / 'runtime' / ('predict_' + method + '.json'), record(done=done, total=1101, seconds=time.time() - start))
    write(RESULT / 'runtime' / ('predict_' + method + '_done.json'), record(done=done, seconds=time.time() - start))
if __name__ == '__main__':
    main(sys.argv[1])
