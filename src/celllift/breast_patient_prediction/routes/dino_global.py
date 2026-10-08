from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from celllift.matched_geometry_controls.dino import load_rgb
from .. import paths
from ..io_utils import atomic_json, atomic_parquet
from ..model_input import layout
from ..model_input.encode import dino_tile_batch_size, load_encoder
from ..model_input.jobs import enrich_tile, load_slide_inventory, load_tile_manifest, slide_groups
from ..protocol import DINO_DIM, MODEL_INPUT_SHARDS
from .config import protocol_meta

def shard_slides(data_root: Path | None, shard: int, shards: int=MODEL_INPUT_SHARDS) -> list[str]:
    manifest = load_tile_manifest(data_root)
    slides = sorted({str(value) for value in manifest['slide_id'].astype(str)})
    return [slide for slide in slides if layout.shard_id(slide, shards) == int(shard)]

def slide_parquet(base: Path, slide_id: str) -> Path:
    return base / 'features' / 'global_dino' / f'{layout.shard_name(slide_id)}__{layout._slide_stem(slide_id)}.parquet'

def encode_slide_global(encoder, job: dict[str, Any], device: str, batch_size: int) -> list[dict[str, Any]]:
    import torch
    rows = []
    tiles = job['tiles']
    step = max(1, int(batch_size))
    for start in range(0, len(tiles), step):
        chunk = tiles[start:start + step]
        images = []
        for tile in chunk:
            rgb = load_rgb(tile['rgb_path'], 'tcga_brca')
            images.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        tensor = torch.from_numpy(np.stack(images)).to(device)
        with torch.inference_mode():
            field = encoder(tensor)
            means = field.mean(dim=(2, 3)).detach().cpu().numpy().astype(np.float32)
        del tensor, field
        for tile, vector in zip(chunk, means):
            if vector.shape != (DINO_DIM,) or not np.isfinite(vector).all():
                raise RuntimeError(f"bad global DINO for {tile['tile_id']}")
            rows.append({'tile_id': tile['tile_id'], 'graph_id': tile['graph_id'], 'patient_id': tile['patient_id'], 'slide_id': tile['slide_id'], 'split': tile['split'], 'dino': vector.tolist()})
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    return rows

def extract_shard(shard: int, *, device: str='cuda', data_root: Path | None=None) -> dict[str, Any]:
    base = layout.ensure(data_root)
    (base / 'features' / 'global_dino').mkdir(parents=True, exist_ok=True)
    manifest = load_tile_manifest(data_root)
    inventory = load_slide_inventory(data_root)
    grouped = slide_groups(manifest.to_dict('records'))
    slides = shard_slides(data_root, shard)
    _, encoder = load_encoder(device)
    batch = dino_tile_batch_size(device)
    done = 0
    for slide_id in slides:
        dest = slide_parquet(base, slide_id)
        if dest.is_file():
            done += 1
            continue
        raw_tiles = grouped.get(slide_id, [])
        tiles = [enrich_tile(row, inventory, base) for row in raw_tiles]
        tiles.sort(key=lambda item: str(item['tile_id']))
        job = {'slide_id': slide_id, 'tiles': tiles}
        rows = encode_slide_global(encoder, job, device, batch)
        atomic_parquet(dest, rows)
        done += 1
    payload = {**protocol_meta(), 'shard': int(shard), 'slides': len(slides), 'written': done}
    atomic_json(base / 'logs' / 'status' / f'global_dino_shard_{int(shard):03d}.json', payload)
    return payload

def merge_global_dino(data_root: Path | None=None) -> dict[str, Any]:
    import pandas as pd
    base = layout.root(data_root)
    frames = [pd.read_parquet(path) for path in sorted((base / 'features' / 'global_dino').glob('*.parquet'))]
    if not frames:
        raise RuntimeError('no global_dino shard parquets')
    merged = pd.concat(frames, ignore_index=True)
    dest = base / 'features' / 'tile_dino_global.parquet'
    rows = merged.to_dict('records')
    for row in rows:
        vector = np.asarray(row['dino'], np.float32).reshape(-1)
        if vector.shape != (DINO_DIM,) or not np.isfinite(vector).all():
            raise RuntimeError(f"non-finite global DINO {row.get('tile_id')}")
    atomic_parquet(dest, rows)
    nucleus = base / 'features' / 'tile_dino.parquet'
    payload = {**protocol_meta(), 'rows': len(rows), 'path': str(dest), 'nucleus_dino_untouched': nucleus.is_file(), 'unique_tiles': int(merged['tile_id'].astype(str).nunique())}
    atomic_json(base / 'qc' / 'tile_dino_global.json', payload)
    return payload
