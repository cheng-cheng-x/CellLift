from celllift.runtime import resource_path as _public_resource
import argparse, sys, traceback, socket
from .common import *
from . import model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--tasks', nargs='*')
    p.add_argument('--stages', nargs='+', default=['predict', 'perturb', 'mask', 'ig'])
    a = p.parse_args()
    provenance = dict(gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source_hashes={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')})
    for task in a.tasks or TASKS[a.dataset]:
        for stage in a.stages:
            splits = ['test', 'fit' if a.dataset == 'tcga_brca' else 'train'] if stage == 'predict' else ['fit' if a.dataset == 'tcga_brca' else 'train', 'test'] if stage in ('mask', 'ig') else ['test']
            if a.dataset in ('arvaniti', 'lizard'):
                splits = ['test', 'val'] if stage == 'predict' else ['val', 'test'] if stage in ('mask', 'ig') else ['test']
            for split in splits:
                status = OUT / a.dataset / task / 'model/worker' / f'{stage}_{split}.json'
                try:
                    write(status, dict(status='running', pid=os.getpid(), stage=stage, split=split, **provenance))
                    sys.argv = ['model', stage, '--dataset', a.dataset, '--split', split]
                    if stage == 'followup':
                        sys.argv = ['followup', '--dataset', a.dataset]
                        if a.dataset in ('arvaniti', 'lizard'):
                            from .al_followup import main as run
                        else:
                            from .ig_followup import main as run
                            sys.argv.extend(['--task', task])
                    elif a.dataset in ('arvaniti', 'lizard'):
                        from . import al_model, al_explain
                        run = (al_explain if stage in ('ig', 'mask') else al_model).main
                    else:
                        sys.argv.extend(['--task', task])
                        run = model.main
                    import torch, gc
                    for attempt in range(6):
                        try:
                            run()
                            break
                        except torch.cuda.OutOfMemoryError:
                            if attempt == 5:
                                raise
                            from .matching import _MATCHERS
                            _MATCHERS.clear()
                            model.adapter.cache_clear()
                            gc.collect()
                            torch.cuda.empty_cache()
                            os.environ['DOWNSTREAM_GRAPH_CHUNK'] = str(max(1, 16 // 2 ** attempt))
                            os.environ['DOWNSTREAM_FORCE_CHECKPOINT'] = '1'
                            print('OOM: resume unfinished units with graph chunk', os.environ['DOWNSTREAM_GRAPH_CHUNK'], flush=True)
                    write(status, dict(status='finished', pid=os.getpid(), stage=stage, split=split, **provenance))
                except Exception as exc:
                    traceback.print_exc()
                    write(status, dict(status='failed', pid=os.getpid(), stage=stage, split=split, error=str(exc), **provenance))
if __name__ == '__main__':
    main()
