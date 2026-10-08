from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import time
import sys
from celllift.runtime import torch
from torch.utils.data import DataLoader
from celllift.reconstruction.src.common import RESULT, ROOT, paths, atomic_json, atomic_save, record
from celllift.reconstruction.src.data import InputDataset, identity_collate, move
from celllift.reconstruction.src.model import ConditionalSceneNetwork, selected
from celllift.reconstruction.src.candidate_tables import select

@torch.no_grad()
def main(shard, shards):
    torch.set_num_threads(4)
    info = json.loads((RESULT / '02_inputs/manifest.json').read_text())
    config = info['config']
    state = torch.load(RESULT / '03_training/final.pt', map_location='cuda', weights_only=False)
    model = ConditionalSceneNetwork(info['full_stats'], config).cuda().eval()
    model.attach_scorer(state['scorer_normalization'])
    model.load_state_dict(state['model'])
    done = 0
    started = time.time()
    for split in config['evaluation_splits']:
        data = InputDataset(paths(), split, False)
        loader = DataLoader(data, batch_size=1, sampler=list(range(shard, len(data), shards)), num_workers=4, pin_memory=True, persistent_workers=True, collate_fn=identity_collate)
        for items in loader:
            g, meta = items[0]
            uid = g.graph_uids[0]
            dest = RESULT / '04_predictions' / split / (uid + '.pt')
            if dest.exists():
                done += 1
                continue
            g = move(g, 'cuda')
            p = model.propose(g, scoring=True)
            cost = -p.scores.log_softmax(-1)
            soft = select(p.tables, (cost + p.tables.prior) * p.tables.valid[:, None], sweeps=3)
            atomic_save(dest, move(dict(ids=g.nucleus_id, geometry=p.geometry, tables=p.tables, score=cost, labels=dict(main=p.labels, soft=soft), search=p.search, metadata=meta), device='cpu'))
            done += 1
            if done % 8 == 0:
                atomic_json(RESULT / 'runtime' / ('predict_%d_status.json' % shard), record(done=done, total=(len(data) + shards - 1 - shard) // shards, seconds=time.time() - started))
                print('PREDICT', shard, done, flush=True)
    atomic_json(RESULT / 'runtime' / ('predict_%d_done.json' % shard), record(done=done, seconds=time.time() - started))
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
