from celllift.runtime import resource_path as _public_resource
from .common import *
from .extract import arrays, summarize
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor

def brca_slide(item):
    path, rows = item
    dest = OUT / 'tcga_brca/shared/shards' / f'{Path(path).stem}.parquet'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return str(dest)
    base = DATA / 'tcga_brca_downstream_v1/08_model_inputs/v1'
    scenes = torch.load(base / 'selected_scenes' / (Path(path).stem + '.pt'), map_location='cpu', weights_only=False)['scenes']
    keys = ['node2d', 'node3d', 'include', 'valid3d', 'edge2d', 'edge3d', 'edge_index', 'nucleus_id']
    with np.load(path, allow_pickle=False) as z:
        ids = z['graph_ids'].astype(str)
        data = {k: z[k] for k in keys}
        ptr = {k: z[k + '_ptr'] for k in keys}
    lookup = {r['graph_id']: r for r in rows}
    out = []
    for i, gid in enumerate(ids):
        g = {k: data[k][:, ptr[k][i]:ptr[k][i + 1]] if k == 'edge_index' else data[k][ptr[k][i]:ptr[k][i + 1]] for k in keys}
        if gid in scenes:
            s = scenes[gid]
            if not np.array_equal(g['nucleus_id'], np.asarray(s['nucleus_id'])):
                raise ValueError('BRCA nucleus alignment')
            for k in ('nucleus_center', 'cell_center'):
                g[k] = np.asarray(s[k])
        r = lookup[gid]
        v = arrays(g)
        v.update({k: r[k] for k in ('graph_id', 'patient_id', 'slide_id', 'split', 'rgb_path')})
        out.append(v)
    pd.DataFrame(out).to_parquet(dest, index=False)
    return str(dest)

def brca():
    base = DATA / 'tcga_brca_downstream_v1/08_model_inputs/v1'
    dest = OUT / 'tcga_brca/shared'
    dest.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(base / 'indices/tiles.parquet', columns=['graph_id', 'patient_id', 'slide_id', 'split', 'rgb_path'])
    from celllift.breast_patient_prediction.model_input.layout import scene_graph_path
    jobs = [(str(scene_graph_path(base, slide)), g.to_dict('records')) for slide, g in t.groupby('slide_id', sort=True)]
    files = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, p in enumerate(pool.map(brca_slide, jobs)):
            files.append(p)
            if i % 20 == 0:
                print('brca slides', i + 1, len(jobs), flush=True)
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    f.to_parquet(dest / 'graphs.parquet', index=False)
    cols = [c for c in f if pd.api.types.is_numeric_dtype(f[c])]
    s = f.groupby(['patient_id', 'slide_id'])[cols].median()
    u = s.groupby('patient_id').median()
    for c in PRIMARY:
        u['between_graph_' + c] = f.groupby('patient_id')[c].quantile(0.75) - f.groupby('patient_id')[c].quantile(0.25)
    patients = pd.read_parquet(base / 'indices/patients.parquet').set_index('patient_id')
    out = []
    for task, col in [('subtype4', 'subtype4_id'), ('luma_lumb', 'luma_lumb_id'), ('idc_ilc', 'idc_ilc_id'), ('er_ihc', 'er_id')]:
        eligible = patients[patients.evaluable & patients[col].notna()]
        for pid, r in eligible.iterrows():
            if pid not in u.index:
                continue
            v = u.loc[pid].to_dict()
            v.update(dataset='tcga_brca', task=task, sample_id=pid, cluster_id=pid, cluster_kind='patient', split={'fit': 'train'}.get(r['split'], r['split']), label_id=int(r[col]))
            out.append(v)
    pd.DataFrame(out).to_parquet(dest / 'units.parquet', index=False)
    write(dest / 'extract_status.json', dict(status='complete', graphs=len(f), patients=len(u), task_units=len(out)))

def canonical(record, scene):
    from celllift.morphology_interaction.geometry import ray_contour_stats, body_features_torch, direction_spacing, equivalent_radius, DISTANCE_SCALE_UM
    ids = np.asarray(record.nucleus_id)
    sid = np.asarray(scene['nucleus_id'])
    lookup = {int(x): j for j, x in enumerate(sid)}
    present = np.array([int(x) in lookup for x in ids])
    order = np.array([lookup.get(int(x), 0) for x in ids])
    s = {}
    for k in ('nucleus_center', 'cell_center', 'nucleus_transform', 'cell_transform', 'valid'):
        original = np.asarray(scene[k])
        v = np.empty((len(ids),) + original.shape[1:], dtype=original.dtype)
        if present.any():
            v[present] = original[order[present]]
        if k == 'valid':
            v[~present] = False
        elif 'transform' in k:
            v[~present] = np.eye(3)
        else:
            v[~present] = np.nan
        s[k] = v
    rays = np.asarray(record.rho_um)
    xy = np.asarray(record.xy_um)
    n2, cov2, _ = ray_contour_stats(rays)
    d, v, nvol, _, nc, cc, na, _ = body_features_torch(torch.tensor(s['nucleus_transform'], dtype=torch.float64), torch.tensor(s['cell_transform'], dtype=torch.float64))
    d = d.numpy()
    d[((na[:, 0] - na[:, 1]) / na[:, 0].clamp_min(1e-12)).numpy() < 0.05, 3] = 1 / 3
    rad = equivalent_radius(nvol).numpy()
    offset = s['cell_center'] - s['nucleus_center']
    den = np.maximum(rad, 1e-06)
    n3 = np.column_stack((d, np.linalg.norm(offset[:, :2], axis=1) / den, abs(offset[:, 2]) / den, np.linalg.norm(offset, axis=1) / den)).astype(np.float32)
    inc = np.isfinite(rays).all(1) & np.isfinite(xy).all(1)
    valid = inc & s['valid'].astype(bool) & v.numpy().astype(bool)
    n3[~valid] = np.nan
    a, b = (np.asarray(record.edge_src, dtype=int), np.asarray(record.edge_dst, dtype=int))
    delta = xy[b] - xy[a]
    q2 = direction_spacing(delta, cov2[a], cov2[b])
    dn = s['nucleus_center'][b] - s['nucleus_center'][a]
    dc = s['cell_center'][b] - s['cell_center'][a]
    qn = direction_spacing(dn, nc.numpy()[a], nc.numpy()[b])
    qc = direction_spacing(dc, cc.numpy()[a], cc.numpy()[b])
    e2 = np.column_stack((np.linalg.norm(delta, axis=1) / DISTANCE_SCALE_UM, q2, n2[b, 0] - n2[a, 0], n2[b, 1] - n2[a, 1])).astype(np.float32)
    e3 = np.column_stack((abs(dn[:, 2]) / np.maximum(rad[a] + rad[b], 1e-06), qn, qc, qn - q2, d[b, 0] - d[a, 0], d[b, 4] - d[a, 4], d[b, 1] - d[a, 1], d[b, 5] - d[a, 5])).astype(np.float32)
    e3[~(valid[a] & valid[b])] = np.nan
    return dict(node2d=np.column_stack((n2, rays)), node3d=n3, include=inc, valid3d=valid, edge2d=e2, edge3d=e3, edge_index=np.stack((a, b)), nucleus_id=ids, center_xy=xy, **s)

def al_load(ds):
    from celllift.core_and_nucleus_prediction.data import GraphReader, GraphRecord, scene_path
    root = DATA / 'model_input_v1' / ds
    reader = GraphReader(root / '03_graph_cache_v2')

    def get(gid):
        record = GraphRecord.from_bytes(reader.get(gid))
        scene = torch.load(scene_path(root / '04_v16c_inputs_v2/selected_scene', gid), map_location='cpu', weights_only=False)
        return (canonical(record, scene), record)
    return (root, get)

def lizard():
    root, get = al_load('lizard')
    dest = OUT / 'lizard/shared'
    dest.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(root / '04_labels_splits_v2/nucleus_labels.parquet')
    rois = pd.read_parquet(root / '04_labels_splits_v2/rois.parquet').set_index('roi_id')
    groups = {}
    pairs = {}
    seen_nodes = set()
    seen_edges = set()
    owned_global = {(str(r.roi_id), int(r.nucleus_id)) for r in labels[labels.owned.astype(bool)].itertuples()}
    for j, (gid, lab) in enumerate(labels.groupby('graph_id', sort=True)):
        g, rec = get(gid)
        roi = str(lab.iloc[0].roi_id)
        cid = str(rois.loc[roi, 'group_id'])
        split = str(lab.iloc[0].official_split)
        table = lab.set_index('nucleus_id')
        nid = g['nucleus_id']
        classes = np.array([int(table.loc[int(n), 'class_id']) - 1 for n in nid])
        owned = np.array([bool(table.loc[int(n), 'owned']) for n in nid])
        a, b = g['edge_index']
        _, ix = np.unique(np.sort(np.stack((a, b), 1), 1), axis=0, return_index=True)
        ix = ix[a[ix] != b[ix]]
        valid = g['include'] & g['valid3d']
        dist = []
        for name in ('nucleus_center', 'cell_center'):
            delta = g[name][b] - g[name][a]
            d3 = np.linalg.norm(delta, axis=1)
            d2 = np.linalg.norm(delta[:, :2], axis=1)
            dist.extend((d3, d2, d3 - d2))
        distances = np.stack(dist, 1)
        for typ in range(6):
            key = (split, cid, typ)
            store = groups.setdefault(key, dict(nodes=[], two=[], edges=[], e2=[], e2valid=[], dist=[], n=0, valid=0))
            ni = [i for i in np.flatnonzero(owned & (classes == typ)) if (roi, int(nid[i])) not in seen_nodes]
            seen_nodes.update(((roi, int(nid[i])) for i in ni))
            ni = np.array(ni, int)
            store['n'] += len(ni)
            vi = ni[valid[ni]]
            store['valid'] += len(vi)
            store['nodes'].append(g['node3d'][vi])
            store['two'].append(g['node2d'][ni, :2])
        for i in ix:
            if not (owned[a[i]] or owned[b[i]]):
                continue
            ek = (roi, min(int(nid[a[i]]), int(nid[b[i]])), max(int(nid[a[i]]), int(nid[b[i]])))
            if ek in seen_edges:
                continue
            seen_edges.add(ek)
            ca, cb = (int(classes[a[i]]), int(classes[b[i]]))
            central_types = {typ for typ, idx in ((ca, a[i]), (cb, b[i])) if (roi, int(nid[idx])) in owned_global}
            for typ in central_types:
                store = groups.setdefault((split, cid, typ), dict(nodes=[], two=[], edges=[], e2=[], e2valid=[], dist=[], n=0, valid=0))
                store['e2'].append(g['edge2d'][i])
                if valid[a[i]] & valid[b[i]]:
                    store['edges'].append(g['edge3d'][i])
                    store['dist'].append(distances[i])
                    store['e2valid'].append(g['edge2d'][i])
            if valid[a[i]] & valid[b[i]]:
                pairs.setdefault((split, cid, min(ca, cb), max(ca, cb)), []).append(np.r_[g['edge3d'][i], distances[i]])
        if j % 50 == 0:
            print('lizard', j, flush=True)
    rows = []
    for (split, cid, typ), s in groups.items():
        if not s['n']:
            continue
        out = dict(dataset='lizard', task='nucleus6', split=split, cluster_id=cid, cluster_kind='source_group', sample_id=f'{cid}:{typ}', label_id=typ, node_count=s['n'], valid_node_count=s['valid'], valid3d_fraction=s['valid'] / s['n'], log_node_count=np.log1p(s['n']))
        summarize(np.concatenate(s['nodes']), NODE, out)
        summarize(np.concatenate(s['two']), TWO[:2], out)
        e = np.array(s['edges']).reshape(-1, 8)
        e[:, 4:] = abs(e[:, 4:])
        summarize(e, EDGE, out)
        e2 = np.array(s['e2']).reshape(-1, 4)
        e2[:, 2:] = abs(e2[:, 2:])
        summarize(e2, TWO[2:], out)
        summarize(np.array(s['dist']).reshape(-1, 6), DIST, out)
        rows.append(out)
        e2v = np.array(s['e2valid']).reshape(-1, 4)
        e2v[:, 2:] = abs(e2v[:, 2:])
        summarize(e2v, TWO[2:], out, 'paired_2d_')
        out.update(edge_count=len(e2), valid_edge_count=len(e2v))
    new = pd.DataFrame(rows)
    if (dest / 'units.parquet').exists():
        old = pd.read_parquet(dest / 'units.parquet').set_index(['split', 'sample_id'])
        aligned = new.set_index(['split', 'sample_id'])
        delta = np.nanmax(abs(old[PRIMARY].to_numpy() - aligned.loc[old.index, PRIMARY].to_numpy()))
        write(dest / 'paired_extension_equivalence.json', dict(primary_max_difference=float(delta), phenotype_rerun_required=bool(delta > 0)))
    new.to_parquet(dest / 'units.parquet', index=False)
    rows = []
    for (split, cid, a, b), vs in pairs.items():
        out = dict(split=split, cluster_id=cid, type_a=a, type_b=b, edges=len(vs))
        v = np.array(vs)
        v[:, 4:8] = abs(v[:, 4:8])
        summarize(v, EDGE + DIST, out)
        rows.append(out)
    pd.DataFrame(rows).to_parquet(dest / 'type_pairs.parquet', index=False)
    write(dest / 'extract_status.json', dict(status='complete', owned_nuclei=len(seen_nodes), unique_edges=len(seen_edges), class_source_units=len(groups)))

def run(ds):
    torch.set_num_threads(1)
    if ds == 'tcga_brca':
        brca()
    elif ds == 'lizard':
        lizard()
    else:
        from .arvaniti_extract import run as arvaniti
        arvaniti()
