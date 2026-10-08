from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np
from celllift.runtime import torch
from scipy import ndimage

def prepare_targets(obs, graph_ids, index, layer, track_lookup, kind, device):
    owners = obs['anchor_node_index'].numpy()
    planes = obs['plane_index'].numpy()
    arrays = {}
    boxes = {}
    xy_all = []
    which = []
    counts = []
    centroids = []
    for oi, (owner, plane) in enumerate(zip(owners, planes)):
        row = index.loc[int(graph_ids[owner])]
        if plane == 1:
            section = int(layer.section_id)
            entity = int(graph_ids[owner]) if kind == 'nucleus' else row.anchor_cell_id
        else:
            side = 'lower' if plane == 0 else 'upper'
            section = row[side + '_section_id']
            entity = row[side + '_' + kind + '_id']
        if not np.isfinite(section) or not np.isfinite(entity):
            raise ValueError('positive target identity missing')
        section = int(section)
        entity = int(entity)
        if section not in arrays:
            target_layer = track_lookup.loc[section]
            arrays[section] = np.load(target_layer[kind + '_mask_path'])
            boxes[section] = ndimage.find_objects(arrays[section])
        labels = arrays[section]
        box = boxes[section][entity - 1]
        if box is None:
            raise ValueError('positive mask UID disappeared')
        y, x = np.nonzero(labels[box] == entity)
        xy = (np.stack((x + box[1].start, y + box[0].start), -1) + 0.5) * 0.46
        xy_all.append(xy)
        which.append(np.full(len(xy), oi))
        counts.append(len(xy))
        centroids.append(xy.mean(0))

    def tensor(x, dtype):
        return torch.as_tensor(x, dtype=dtype, device=device)
    return dict(obs_owner=tensor(owners, torch.long), plane=tensor(planes, torch.long), observation=tensor(np.concatenate(which) if which else [], torch.long), xy=tensor(np.concatenate(xy_all) if xy_all else np.empty((0, 2)), torch.float32), target_count=tensor(counts, torch.long), target_centroid=tensor(np.array(centroids).reshape(-1, 2), torch.float32))

def distribution(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(count=0)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()), quantiles={str(q): float(np.quantile(x, q)) for q in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)})
