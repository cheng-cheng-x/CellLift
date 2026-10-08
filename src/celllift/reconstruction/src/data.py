from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass, fields
from celllift.runtime import ResourcePath as Path
from celllift.runtime import json
from celllift.runtime import torch
from torch.utils.data import Dataset
from .frozen.batch_types import NucleusGraphBatch as BaseGraph, EllipseObservationBatch
from .frozen.collate import collate_graphs, _cat_observations
from .geometry import coefficients, Coefficients

@dataclass(frozen=True)
class Graph(BaseGraph):
    dino_features: torch.Tensor | None = None
    coeff: Coefficients | None = None
    graph_uids: tuple = ()
    query: torch.Tensor | None = None
    initial_raw: torch.Tensor | None = None

    def validate(self):
        super().validate()
        if self.dino_features.shape != (len(self.nucleus_id), 384):
            raise ValueError('DINO feature shape')

    def pin_memory(self):
        return move(self, pin=True)

@dataclass(frozen=True)
class Supervision:
    nucleus: EllipseObservationBatch
    cell: EllipseObservationBatch
    compatibility_confirmed_empty: torch.Tensor

    def pin_memory(self):
        return move(self, pin=True)

def move(value, device=None, pin=False):
    if isinstance(value, torch.Tensor):
        return value.pin_memory() if pin else value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [move(x, device, pin) for x in value]
    if isinstance(value, tuple):
        return tuple((move(x, device, pin) for x in value))
    if isinstance(value, dict):
        return {k: move(v, device, pin) for k, v in value.items()}
    if hasattr(value, '__dataclass_fields__'):
        return type(value)(**{f.name: move(getattr(value, f.name), device, pin) for f in fields(value)})
    return value

def aligned_indices(source, wanted):
    order = torch.argsort(source)
    pos = torch.searchsorted(source[order], wanted)
    if len(torch.unique(source)) != len(source) or (pos >= len(source)).any():
        raise ValueError('invalid UID set')
    if not torch.equal(source[order[pos]], wanted):
        raise ValueError('missing UID')
    return order[pos]

class InputDataset(Dataset):

    def __init__(self, p, split='train', targets=False, initialize_missing=True):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        if split == 'test' and targets:
            raise ValueError('TEST supervision is evaluation-only')
        self.p = {k: Path(v) for k, v in p.items()}
        self.targets = targets
        self.split = split
        self.initialize_missing = initialize_missing
        self.manifest = json.loads((self.p['cache'] / 'manifest.json').read_text())
        self.rows = [r for r in self.manifest['items'] if r['split'] == split]
        self._coeff = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = torch.load(self.p['cache'] / row['graph_file'], weights_only=False, map_location='cpu')
        base = BaseGraph.from_mapping(payload['graph'])
        side = payload['input']
        if not torch.equal(base.nucleus_id, side['ids']):
            raise ValueError('Input UID mismatch')
        cf = Coefficients(**side['coeff'])
        query = side['query']
        raw = torch.zeros((len(base.nucleus_id), 9, 10))
        raw[:, :, 0] = query
        g = Graph(**{f.name: getattr(base, f.name) for f in fields(base)}, dino_features=side['dino_features'], coeff=cf, graph_uids=(row['graph_id'],), query=query, initial_raw=raw)
        if not self.targets:
            return (g, payload['metadata'])
        sup = torch.load(self.p['cache'] / row['supervision_file'], weights_only=False, map_location='cpu')
        return (g, Supervision(EllipseObservationBatch(**sup['nucleus']), EllipseObservationBatch(**sup['cell']), sup['confirmed_empty_mask']))

def collate(items):
    graphs, other = zip(*items)
    base, offsets = collate_graphs(graphs)
    cf = Coefficients(**{f.name: torch.cat([getattr(g.coeff, f.name) for g in graphs]) for f in fields(Coefficients)})
    graph = Graph(**{f.name: getattr(base, f.name) for f in fields(base)}, dino_features=torch.cat([g.dino_features for g in graphs]), coeff=cf, graph_uids=tuple((g.graph_uids[0] for g in graphs)), query=torch.cat([g.query for g in graphs]), initial_raw=torch.cat([g.initial_raw for g in graphs]))
    if not isinstance(other[0], Supervision):
        return (graph, list(other))
    return (graph, Supervision(_cat_observations([s.nucleus for s in other], offsets), _cat_observations([s.cell for s in other], offsets), torch.cat([s.compatibility_confirmed_empty for s in other])))

def identity_collate(items):
    return items
