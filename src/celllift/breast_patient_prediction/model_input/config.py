from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
from .. import paths
from ..io_utils import atomic_json, sha256_file
from ..protocol import CELLPOSE_CELLPROB, CELLPOSE_DIAMETER_PX, CELLPOSE_FLOW, DINO_DIM, DINO_SOURCE_ROOT, DINO_WEIGHT_PATH, FIELD_UM, GRAPH_K, GRAPH_NORMALIZER_UM, GRAPH_RADIUS_UM, IMAGE_SIZE, MODEL_INPUT_SHARDS, MODEL_INPUT_VERSION, PROTOCOL_ID, RAY_COUNT, TARGET_MPP, TRAINING_FEATURE_STATS, ProjectionScene_CHECKPOINT, ProjectionScene_CHECKPOINT_SHA256, ProjectionScene_INPUT_MANIFEST, ProjectionScene_SOURCE_ROOT
from . import layout

def default_config(data_root: Path | None=None) -> dict[str, Any]:
    root = paths.data_root(data_root)
    return {'protocol_id': PROTOCOL_ID, 'model_input_version': MODEL_INPUT_VERSION, 'dataset': 'tcga_brca', 'note': 'Shared B/G2/G3/F2/F3 cache. No training. Splits and labels are frozen.', 'resolution': {'target_mpp': TARGET_MPP, 'image_size': IMAGE_SIZE, 'field_um': FIELD_UM, 'interpolation': 'LANCZOS', 'rgba_background': [255, 255, 255], 'crc_pad': False}, 'cellpose': {'environment': str(paths.CELLPOSE_PYTHON), 'version': '3.1.1.3', 'nucleus_model': 'nuclei', 'nucleus_diameter_px': CELLPOSE_DIAMETER_PX, 'flow_threshold': CELLPOSE_FLOW, 'cellprob_threshold': CELLPOSE_CELLPROB, 'require_cuda': True, 'vectorized_get_masks': True, 'batch_size': 8, 'input_mode': 'nucleus_only'}, 'graph': {'ray_count': RAY_COUNT, 'k_neighbors': GRAPH_K, 'radius_um': GRAPH_RADIUS_UM, 'edge_normalizer_um': GRAPH_NORMALIZER_UM, 'moments': 'x+0.5/y+0.5 and I/12', 'training_feature_stats': TRAINING_FEATURE_STATS, 'feature_stats_policy': 'frozen_projection_scene_training_only'}, 'dino': {'source_root': DINO_SOURCE_ROOT, 'weight_path': DINO_WEIGHT_PATH, 'dim': DINO_DIM, 'tile_pool': 'nucleus_token_mean_or_patch_token_mean_if_empty', 'crc_pad': False}, 'projection_scene': {'source_root': ProjectionScene_SOURCE_ROOT, 'checkpoint': ProjectionScene_CHECKPOINT, 'input_manifest': ProjectionScene_INPUT_MANIFEST, 'checkpoint_sha256': ProjectionScene_CHECKPOINT_SHA256, 'device_limits': {'A800': [8, 8192], 'other': [4, 4096]}}, 'views': {'rgb_tiles': 'tile 384D from the same frozen DINO forward; PNG retained', 'geometry_tokens': 'observed rays + selected 3D; G2 uses nucleus-mask 2D only', 'scene_graphs': 'identity-aligned parallel-b payload with full 3x3 transforms', 'f2': 'disable all 3D-derived channels', 'f3': 'geometry isolated from B'}, 'runtime': {'shards': MODEL_INPUT_SHARDS, 'cpu_threads': 8, 'rgb_workers': 16, 'graph_workers': 8, 'one_gpu_per_job': True, 'host_priority': ['Configured resource', 'Configured resource', 'Configured resource', 'Configured resource', 'Configured resource']}, 'paths': {'data_root': str(root), 'model_input_root': str(layout.root(root)), 'result_root': str(paths.result_root()), 'tile_manifest': str(paths.tile_dir(root) / 'tile_manifest.parquet'), 'patient_labels': str(paths.label_dir(root) / 'patient_labels.parquet'), 'python_reconstruct': str(paths.PRETRAIN_PYTHON), 'python_cellpose': str(paths.CELLPOSE_PYTHON)}}

def write_config(data_root: Path | None=None, *, verify_checkpoint: bool=True) -> dict[str, Any]:
    cfg = default_config(data_root)
    base = layout.ensure(data_root)
    if verify_checkpoint:
        checkpoint = Path(cfg['projection_scene']['checkpoint'])
        digest = sha256_file(checkpoint)
        if digest != ProjectionScene_CHECKPOINT_SHA256:
            raise RuntimeError(f'ProjectionScene checkpoint SHA mismatch: {digest}')
        cfg['projection_scene']['checkpoint_sha256_verified'] = digest
        stats = Path(cfg['graph']['training_feature_stats'])
        if not stats.is_file():
            raise FileNotFoundError(stats)
        cfg['graph']['training_feature_stats_sha256'] = sha256_file(stats)
        dino = Path(cfg['dino']['weight_path'])
        if not dino.is_file():
            raise FileNotFoundError(dino)
        cfg['dino']['weight_sha256'] = sha256_file(dino)
    atomic_json(base / 'config.json', cfg)
    return cfg
