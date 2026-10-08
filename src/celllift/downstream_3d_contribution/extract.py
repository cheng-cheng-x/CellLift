from celllift.runtime import resource_path as _public_resource
from .common import *
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def summarize(a, names, out, prefix=''):
    a = np.asarray(a)
    for j, n in enumerate(names):
        v = a[:, j]
        v = v[np.isfinite(v)]
        q = np.quantile(v, [0.25, 0.5, 0.75]) if len(v) else [np.nan] * 3
        out[prefix + n + '__median'] = q[1]
        out[prefix + n + '__iqr'] = q[2] - q[0]

def arrays(g, node_mask=None, edge_mask=None):
    include = g['include'].astype(bool)
    valid = include & g['valid3d'].astype(bool)
    if node_mask is not None:
        include = include & node_mask
        valid = valid & node_mask
    src, dst = g['edge_index']
    pairs = np.sort(np.stack((src, dst), 1), 1)
    _, ix = np.unique(pairs, axis=0, return_index=True)
    eligible = include[src] | include[dst] if node_mask is not None else include[src] & include[dst]
    if edge_mask is not None:
        eligible = g['include'].astype(bool)[src] & g['include'].astype(bool)[dst]
    ix = ix[eligible[ix] & (src[ix] != dst[ix])]
    if edge_mask is not None:
        ix = ix[edge_mask[ix]]
    allvalid = g['include'].astype(bool) & g['valid3d'].astype(bool)
    vi = ix[allvalid[src[ix]] & allvalid[dst[ix]]]
    e2 = g['edge2d'][ix].copy()
    e2[:, 2:] = abs(e2[:, 2:])
    e2v = g['edge2d'][vi].copy()
    e2v[:, 2:] = abs(e2v[:, 2:])
    e3 = g['edge3d'][vi].copy()
    e3[:, 4:] = abs(e3[:, 4:])
    out = dict(node_count=int(include.sum()), valid_node_count=int(valid.sum()), valid3d_fraction=float(valid.sum() / max(1, include.sum())), log_node_count=float(np.log1p(include.sum())), edge_count=len(ix), valid_edge_count=len(vi))
    summarize(g['node2d'][include, :2], TWO[:2], out)
    summarize(e2, TWO[2:], out)
    summarize(e2v, TWO[2:], out, 'paired_2d_')
    summarize(g['node3d'][valid], NODE, out)
    summarize(e3, EDGE, out)
    ds = []
    for key in ('nucleus_center', 'cell_center'):
        if key not in g:
            ds.extend([np.full(len(vi), np.nan)] * 3)
            continue
        d = np.asarray(g[key], float)[dst[vi]] - np.asarray(g[key], float)[src[vi]]
        d3 = np.linalg.norm(d, axis=1)
        d2 = np.linalg.norm(d[:, :2], axis=1)
        if np.any(d3 + 1e-07 < d2):
            raise ValueError('3D distance smaller than its projection')
        ds.extend((d3, d2, d3 - d2))
    summarize(np.stack(ds, 1), DIST, out)
    return out

def one(e):
    with np.load(e['path'], allow_pickle=False) as z:
        g = {k: z[k] for k in z.files if k not in ('dino',)}
    out = arrays(g)
    out.update(graph_id=e['graph_id'], **{k: v for k, v in e.get('metadata', {}).items() if k != 'graph_id'})
    out['rgb_path'] = e.get('rgb_path', '')
    out['source_path'] = e['path']
    return out

def hierarchy(frame, keys, cols):
    return frame.groupby(keys, dropna=False, sort=False)[cols].median().reset_index()

def official(ds):
    dest = OUT / ds / 'shared'
    dest.mkdir(parents=True, exist_ok=True)
    es = read(DATA / 'v16c_parallel_routes_v1' / ds / 'parallel_v1/scene_index.json')['graphs']
    chunks = []
    for start in range(0, len(es), 256):
        path = dest / 'shards' / f'{start // 256:05d}.parquet'
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            with ThreadPoolExecutor(max_workers=8) as pool:
                f = pd.DataFrame(pool.map(one, es[start:start + 256]))
            f.to_parquet(path, index=False)
        chunks.append(pd.read_parquet(path))
        if start % 2048 == 0:
            print(ds, 'extracted', min(start + 256, len(es)), len(es), flush=True)
    f = pd.concat(chunks, ignore_index=True)
    f.to_parquet(dest / 'graphs.parquet', index=False)
    numeric = [c for c in f if pd.api.types.is_numeric_dtype(f[c]) and c not in ('fold', 'label_id')]
    records = []
    missing = []
    for split in ('train', 'test'):
        rows = read(DATA / 'official_split_full_v16c/splits' / ds / f'{split}_rows.json')
        lookup = f.set_index('graph_id', drop=False)
        for r in rows:
            if ds == 'sicapv2':
                g = lookup.loc[[r['graph_id']]]
            elif ds == 'bracs':
                absent = [gid for gid in r['graph_ids'] if gid not in lookup.index]
                if absent:
                    missing.append(dict(sample_id=r['sample_id'], split=split, missing_graph_ids=absent, requested_graphs=len(r['graph_ids'])))
                g = lookup.loc[[gid for gid in r['graph_ids'] if gid in lookup.index]]
            else:
                g = f[f.patient_id.astype(str) == str(r['patient_id'])]
            if g.empty:
                missing.append(dict(sample_id=r['sample_id'], split=split, status='no available graphs'))
                continue
            if ds == 'tcga_crc_msi':
                slide = g.graph_id.str.split(':', n=1).str[-1].str.split('_(', regex=False).str[0]
                values = g.assign(slide_id=slide).groupby('slide_id')[numeric].median().median()
            else:
                values = g[numeric].median()
            out = values.to_dict()
            out.update(dataset=ds, task=TASKS[ds][0], split=split, sample_id=str(r['sample_id']), cluster_id=str(r.get('patient_id', r.get('wsi_id'))), label_id=int(r['label_id']), graph_count=len(g), cluster_kind='parent_wsi' if ds == 'bracs' else 'patient')
            if ds != 'sicapv2':
                for c in PRIMARY:
                    out['between_graph_' + c] = g[c].quantile(0.75) - g[c].quantile(0.25)
            out['graph_coverage'] = len(g) / len(r['graph_ids']) if ds == 'bracs' else 1.0
            records.append(out)
    t = pd.DataFrame(records)
    t.to_parquet(dest / 'units.parquet', index=False)
    if ds == 'sicapv2':
        old = pd.read_parquet(RESULT / 'sicap_3d_contribution_v1/shared/descriptors.parquet').set_index('sample_id')
        new = t.set_index('sample_id')
        delta = np.nanmax(abs(new.loc[old.index, PRIMARY].to_numpy() - old[PRIMARY].to_numpy()))
        write(dest / 'sicap_equivalence.json', dict(max_absolute_error=float(delta), passed=bool(delta == 0)))
        if delta != 0:
            raise ValueError('SICAP primary descriptor mismatch')
    write(dest / 'missing_graphs.json', missing)
    write(dest / 'extract_status.json', dict(status='complete_with_missing_graphs' if missing else 'complete', graphs=len(f), units=len(t), split_counts=t.groupby('split').size().to_dict(), distance_missing={c: int(t[c].isna().sum()) for c in DISTANCE}))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=TASKS)
    a = p.parse_args()
    if a.dataset in ('sicapv2', 'bracs', 'tcga_crc_msi'):
        official(a.dataset)
    else:
        from .special_extract import run
        run(a.dataset)
if __name__ == '__main__':
    main()
