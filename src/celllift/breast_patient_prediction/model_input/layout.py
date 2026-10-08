from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from .. import paths
from ..io_utils import stable_shard
from ..protocol import MODEL_INPUT_SHARDS
TREE_FOLDERS = ('rgb', 'masks', 'graphs', 'selected_scenes', 'features', 'features/geometry_tokens', 'features/scene_graphs', 'indices', 'qc', 'qc/pilot_overlays', 'logs', 'logs/status', 'logs/claims')

def root(data: Path | None=None) -> Path:
    return paths.model_input_dir(data)

def ensure(data: Path | None=None) -> Path:
    base = root(data)
    for folder in TREE_FOLDERS:
        (base / folder).mkdir(parents=True, exist_ok=True)
    paths.result_root().mkdir(parents=True, exist_ok=True)
    return base

def shard_id(slide_id: str, shards: int=MODEL_INPUT_SHARDS) -> int:
    return stable_shard(str(slide_id), shards)

def shard_name(slide_id: str, shards: int=MODEL_INPUT_SHARDS) -> str:
    return f'shard_{shard_id(slide_id, shards):03d}'

def rgb_path(base: Path, slide_id: str, tile_id: str) -> Path:
    return base / 'rgb' / shard_name(slide_id) / f'{tile_id}.png'

def mask_npz_path(base: Path, slide_id: str) -> Path:
    return base / 'masks' / shard_name(slide_id) / f'{_slide_stem(slide_id)}.npz'

def instances_path(base: Path, slide_id: str) -> Path:
    return base / 'masks' / shard_name(slide_id) / f'{_slide_stem(slide_id)}.instances.json.gz'

def graph_path(base: Path, slide_id: str) -> Path:
    return base / 'graphs' / f'{shard_name(slide_id)}__{_slide_stem(slide_id)}.npz'

def scene_path(base: Path, slide_id: str) -> Path:
    return base / 'selected_scenes' / f'{shard_name(slide_id)}__{_slide_stem(slide_id)}.pt'

def geometry_path(base: Path, slide_id: str) -> Path:
    return base / 'features' / 'geometry_tokens' / f'{shard_name(slide_id)}__{_slide_stem(slide_id)}.npz'

def scene_graph_path(base: Path, slide_id: str) -> Path:
    return base / 'features' / 'scene_graphs' / f'{shard_name(slide_id)}__{_slide_stem(slide_id)}.npz'

def slide_status_path(base: Path, slide_id: str) -> Path:
    return base / 'logs' / 'status' / f'{_slide_stem(slide_id)}.json'

def claim_dir(base: Path, stage: str, slide_id: str) -> Path:
    return base / 'logs' / 'claims' / stage / _slide_stem(slide_id)

def _slide_stem(slide_id: str) -> str:
    text = str(slide_id).replace('\\', '/').split('/')[-1]
    if text.lower().endswith('.svs'):
        text = text[:-4]
    return ''.join((ch if ch.isalnum() or ch in '._-' else '-' for ch in text)) or 'slide'
