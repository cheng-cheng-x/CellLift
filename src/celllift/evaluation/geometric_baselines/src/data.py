from celllift.runtime import resource_path as _public_resource
import json, torch
from .common import PATHS

def rows(split):
    return [x for x in json.loads((PATHS['cache'] / 'manifest.json').read_text())['items'] if x['split'] == split]

def load(row, supervision=False):
    p = torch.load(PATHS['cache'] / row['graph_file'], map_location='cpu', weights_only=False)
    g = p['graph']
    item = dict(root=g['fitted_root_xy'].double(), center=g['fitted_center_xy'].double(), ids=g['nucleus_id'], metadata=p['metadata'], uid=row['graph_id'])
    if supervision:
        if row['split'] != 'train':
            raise ValueError('Only TRAIN can supply calibration targets')
        s = torch.load(PATHS['cache'] / row['supervision_file'], map_location='cpu', weights_only=False)
        w = torch.load(PATHS['result'] / '02_inputs' / row['supervision_file'], map_location='cpu', weights_only=True)
        assert torch.equal(w['nucleus_id'], item['ids'])
        for kind in ['nucleus', 'cell']:
            s[kind]['weight'] = w[kind + '_weight']
        item['supervision'] = s
    return item
