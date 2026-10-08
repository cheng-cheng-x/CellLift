from celllift.runtime import resource_path as _public_resource
from .common import *
from .special_extract import al_load
from .extract import arrays
import pandas as pd
_CACHE = None

def extract_core(item):
    global _CACHE
    if _CACHE is None:
        import torch
        torch.set_num_threads(2)
        root, get = al_load('arvaniti')
        supervised = pd.read_parquet(root / '04_labels_splits_v2/supervised_windows.parquet').set_index('graph_id')
        _CACHE = (get, supervised)
    get, supervised = _CACHE
    core, rows = item
    dest = OUT / 'arvaniti/shared'
    path = dest / 'shards_edge_ownership_v2' / f'{core}.parquet'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return str(path)
    graphs = {}
    node_owner = {}
    edge_owner = {}
    for r in rows.to_dict('records'):
        gid = r['graph_id']
        g, record = get(gid)
        graphs[gid] = (g, r)
        xy = np.asarray(record.xy_px)
        center = float(r['size_target']) / 2
        scores = np.sum((xy - center) ** 2, axis=1)
        for j, nid in enumerate(g['nucleus_id']):
            score = (float(scores[j]), gid)
            if int(nid) not in node_owner or score < node_owner[int(nid)]:
                node_owner[int(nid)] = score
        a, b = g['edge_index']
        scores = np.sum(((xy[a] + xy[b]) / 2 - center) ** 2, axis=1)
        for j in np.flatnonzero(a != b):
            key = tuple(sorted((int(g['nucleus_id'][a[j]]), int(g['nucleus_id'][b[j]]))))
            score = (float(scores[j]), gid)
            if key not in edge_owner or score < edge_owner[key]:
                edge_owner[key] = score
    out = []
    counted_edges = 0
    for gid, (g, r) in graphs.items():
        nodes = np.array([node_owner[int(n)][1] == gid for n in g['nucleus_id']], dtype=bool)
        a, b = g['edge_index']
        edges = np.array([u != v and edge_owner[tuple(sorted((int(g['nucleus_id'][u]), int(g['nucleus_id'][v]))))][1] == gid for u, v in zip(a, b)], dtype=bool)
        v = arrays(g, node_mask=nodes, edge_mask=edges)
        v.update(graph_id=gid, core_id=core, cluster_id=core, cluster_kind='core_not_patient', split=str(r['official_split']), rgb_path=r['rgb_path'], label_p1=-1, label_p2=-1)
        counted_edges += v['edge_count']
        if gid in supervised.index:
            s = supervised.loc[gid]
            v['label_p1'] = int(s.label_id_p1 if r['official_split'] == 'test' else s.label_id)
            v['label_p2'] = int(s.label_id_p2 if r['official_split'] == 'test' else s.label_id)
        out.append(v)
    expected = 0
    lookups = {gid: {int(n): j for j, n in enumerate(g['nucleus_id'])} for gid, (g, _) in graphs.items()}
    for pair, (_, owner) in edge_owner.items():
        g, _ = graphs[owner]
        lookup = lookups[owner]
        expected += bool(g['include'][lookup[pair[0]]] and g['include'][lookup[pair[1]]])
    if counted_edges != expected:
        raise ValueError(f'Window edge ownership mismatch: {core} {counted_edges} != {expected}')
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    pd.DataFrame(out).to_parquet(temporary, index=False)
    temporary.replace(path)
    return str(path)

def run():
    root = DATA / 'model_input_v1/arvaniti'
    dest = OUT / 'arvaniti/shared'
    dest.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(root / '04_labels_splits_v2/inference_windows.parquet')
    available = set(pd.read_parquet(root / '03_graph_cache_v2/graph_index.parquet').graph_id)
    meta = meta[meta.graph_id.isin(available)]
    supervised = pd.read_parquet(root / '04_labels_splits_v2/supervised_windows.parquet').set_index('graph_id')
    from celllift.core_and_nucleus_prediction.evaluate import _core_gt_map
    gt = _core_gt_map()
    files = []
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, path in enumerate(pool.map(extract_core, meta.groupby('core_id', sort=True))):
            files.append(path)
            if i % 20 == 0:
                print('arvaniti cores', i + 1, meta.core_id.nunique(), flush=True)
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    f.to_parquet(dest / 'graphs.parquet', index=False)
    units = []
    numeric = [c for c in f if pd.api.types.is_numeric_dtype(f[c]) and (not c.startswith('label_'))]
    for reader in (1, 2):
        for r in f[f[f'label_p{reader}'] >= 0].to_dict('records'):
            r.update(dataset='arvaniti', task=f'local_p{reader}', sample_id=r['graph_id'], label_id=r[f'label_p{reader}'])
            units.append(r)
        for core, g in f.groupby('core_id'):
            ref = gt.get(core, {}).get(f'pathologist{reader}')
            if ref is None and str(g.iloc[0].split) != 'test':
                ref = gt.get(core, {}).get('pathologist1', gt.get(core, {}).get('train'))
            if ref is None:
                continue
            r = g[numeric].median().to_dict()
            r.update(dataset='arvaniti', task=f'core_p{reader}', sample_id=core, cluster_id=core, cluster_kind='core_not_patient', split=g.iloc[0].split, label_id=int(ref['qwk_label']))
            for c in PRIMARY:
                r['between_graph_' + c] = g[c].quantile(0.75) - g[c].quantile(0.25)
            units.append(r)
    pd.DataFrame(units).to_parquet(dest / 'units.parquet', index=False)
    write(dest / 'extract_status.json', dict(status='complete', edge_ownership_version=2, graphs=len(f), cores=f.core_id.nunique(), units=len(units), development_reader2='same training annotation as reader1; test readers separate'))
