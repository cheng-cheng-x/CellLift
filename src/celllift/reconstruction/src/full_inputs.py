from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import fields
from celllift.runtime import ResourcePath as Path
import hashlib
from celllift.runtime import json
import numpy as np
import pandas as pd
from celllift.runtime import torch
from .common import RESULT, paths, atomic_save
from .upstream.cache_io import ShardedLMDBReader
from .upstream.schemas import ObservationRecord, TargetRecord
from .upstream.compatibility_schemas import GraphRecord
from .upstream.moments import graph_from_record, observations_from_record
from .geometry import coefficients
from .reference import reference

class RawSource:

    def __init__(self):
        p = paths()
        self.graph = ShardedLMDBReader(p['graph_cache'], 'graph_cache', 64)
        self.positive = ShardedLMDBReader(p['observation_cache'], 'positive', 64)
        self.target = ShardedLMDBReader(p['observation_cache'], 'target_cache', 64)
        self.fits = {}

    def read_target(self, uid):
        return TargetRecord.from_bytes(self.target.get(uid))

    def graph_record(self, uid):
        return GraphRecord.from_bytes(self.graph.get(uid))

    def observations(self, uid, g, index):
        if len(self.fits) > 100000:
            self.fits.clear()
        record = ObservationRecord.from_bytes(self.positive.get(uid))
        out = {}
        aligned = index.reindex(g.nucleus_id.numpy())
        for kind, typ in [('nucleus', 0), ('cell', 1)]:
            obs = observations_from_record(record, g, typ, self.read_target, self.fits, 0.46)
            for plane, side in [(0, 'lower'), (2, 'upper')]:
                keep = obs.plane_index == plane
                quality = aligned[side + '_link_class'].to_numpy()[obs.anchor_node_index[keep].numpy()]
                if not np.isin(quality, [1, 2]).all():
                    raise ValueError('Positive observation has no Gold/Silver identity')
                obs.weight[keep] = torch.tensor(np.where(quality == 1, 1.0, 0.25), dtype=obs.weight.dtype)
            obs.weight[obs.plane_index == 1] = 1.0
            out[kind] = {f.name: getattr(obs, f.name) for f in fields(obs)}
        out['confirmed_empty_mask'] = torch.zeros(len(g.nucleus_id), 2)
        return out

def confidence_index(split):
    columns = ['anchor_layer_idx', 'anchor_nucleus_id', 'lower_link_class', 'upper_link_class']
    x = pd.read_parquet(paths()['prepared'] / 'nucleus_training_index.parquet', columns=columns, filters=[('split', '==', split)])
    for side in ['lower', 'upper']:
        x[side + '_link_class'] = x[side + '_link_class'].map({'gold': 1, 'silver_nucleus': 2}).fillna(0).astype('uint8')
    return (x, x.groupby('anchor_layer_idx', sort=False).indices)

def sample_one(field, xy):
    grid = (2 * (xy.float() + 6) / 1036 - 1)[None, :, None, :]
    return torch.nn.functional.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, :, 0].T
