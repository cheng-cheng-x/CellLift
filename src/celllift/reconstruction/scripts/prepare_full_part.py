from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import sys
from celllift.runtime import json
import time
import hashlib
import lmdb
from dataclasses import fields
from celllift.runtime import ResourcePath as Path
import numpy as np
import pandas as pd
from celllift.runtime import torch
from PIL import Image
from celllift.reconstruction.src.common import ROOT, RESULT, paths, layout, atomic_json, atomic_save, record
from celllift.reconstruction.src.full_inputs import RawSource, confidence_index, sample_one
from celllift.reconstruction.src.upstream.moments import graph_from_record
from celllift.reconstruction.src.upstream.he_feature_cache import DINOv2S14Encoder, decode_he_feature
from celllift.reconstruction.src.geometry import coefficients
from celllift.reconstruction.src.reference import reference

def main(rank, world):
    layout()
    torch.set_num_threads(4)
    p = paths()
    start = time.time()
    rows = pd.read_parquet(p['graph_cache'] / 'graph_index.parquet')
    rows = rows[rows.split.isin(['train', 'val'])].sort_values(['split', 'layer_idx']).to_dict('records')
    lookup = pd.read_parquet(p['prepared'] / 'layer_lookup.parquet').set_index('roi_layer_id')
    quality, groups = confidence_index('train')
    source = RawSource()
    he = json.loads((p['small_dino'] / 'manifest.json').read_text())
    encoder = None
    envs = {}
    items = []
    for pos, row in enumerate(rows):
        if pos % world != rank:
            continue
        uid = row['graph_id']
        split = row['split']
        stem = split + '_' + hashlib.sha1(uid.encode()).hexdigest()[:20] + '.pt'
        gf = 'graphs/' + stem
        sf = 'supervision/' + stem
        dest = p['cache'] / gf
        meta_path = RESULT / '02_inputs/items' / (stem + '.json')
        if meta_path.exists():
            items.append(json.loads(meta_path.read_text()))
            continue
        rec = source.graph_record(uid)
        g, middle = graph_from_record(rec, source.read_target, source.fits, 0.46)
        if len(source.fits) > 100000:
            source.fits.clear()
        if uid in he['records']:
            shard = he['records'][uid]['shard']
            if shard not in envs:
                envs[shard] = lmdb.open(str(p['small_dino'] / ('he_feature_%02d.lmdb' % shard)), readonly=True, lock=False, readahead=False)
            with envs[shard].begin() as tx:
                ids, feature = decode_he_feature(tx.get(uid.encode()))
            if not np.array_equal(ids, g.nucleus_id.numpy()):
                raise ValueError('Reused DINO UID mismatch')
            feature = torch.from_numpy(feature)
        else:
            if encoder is None:
                encoder = DINOv2S14Encoder(Path(he['encoder_source']), Path(he['encoder_weight'])).cuda().eval()
            image = np.asarray(Image.open(lookup.loc[uid, 'source_png_path']).convert('RGB'))
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                field = encoder(image)
                feature = sample_one(field, torch.from_numpy(rec.xy_px.copy()).cuda()).half().cpu()
        with torch.no_grad():
            root = g.fitted_root_xy.cuda()
            cf = coefficients(root)
            query = reference(root)[1].cpu()
        coeff = {f.name: getattr(cf, f.name).cpu() for f in fields(cf)}
        metadata = dict(graph_id=uid, track_id=rec.track_id, layer_idx=int(rec.layer_idx), split=split, middle_input_source='self_uid_raw_mask_uniform_pixel_union')
        atomic_save(dest, dict(graph={f.name: getattr(g, f.name) for f in fields(g)}, metadata=metadata, input=dict(ids=g.nucleus_id, dino_features=feature, coeff=coeff, query=query)))
        if split == 'train':
            idx = quality.iloc[groups[int(rec.layer_idx)]].set_index('anchor_nucleus_id')
            atomic_save(p['cache'] / sf, source.observations(uid, g, idx))
        rays = g.nucleus_rays_um.double()
        item = dict(uid=uid, graph_id=uid, split=split, track_id=rec.track_id, graph_file=gf, supervision_file=sf, nodes=len(g.nucleus_id), moment=[float(rays.sum()), float(rays.square().sum()), rays.numel()])
        atomic_json(meta_path, item)
        items.append(item)
        if len(items) % 8 == 0:
            atomic_json(RESULT / 'runtime' / ('prepare_%d_status.json' % rank), record(done=len(items), total=len(rows[rank::world]), seconds=time.time() - start))
            print(rank, len(items), flush=True)
    atomic_json(RESULT / '02_inputs' / ('part_%d.json' % rank), items)
    atomic_json(RESULT / 'runtime' / ('prepare_%d_done.json' % rank), record(done=len(items), seconds=time.time() - start))
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
