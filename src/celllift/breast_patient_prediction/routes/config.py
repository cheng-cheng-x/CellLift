from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Any
import os
from .. import paths
from ..protocol import ER_IDS, HISTO_IDS, LUMAB_IDS, PROTOCOL_ID, ROUTES_BATCH_ID, ROUTES_conditional_geometry_BATCH_ID, SEED, SUBTYPE4, TASKS
BATCH_ID = os.environ.get('TCGA_BRCA_ROUTES_BATCH', ROUTES_BATCH_ID)
DINO_DIM = 384
TILE_EMBED = 128
ATTN_DIM = 64
GEOM_EMBED = 64
PROTO_WIDTH = 64
MAX_TILES = 128
TOKEN_DIM = 41
MASK_DIM = 36
THREE_D_DIM = 5
NODE_2D = 38
NODE_3D = 12
EDGE_2D = 4
EDGE_3D = 8
NODE_DIM = NODE_2D + NODE_3D
EDGE_DIM = EDGE_2D + EDGE_3D
ACCUM = 4
DROPOUT = 0.1
COMMON_ARMS = ('B', 'G2', 'G3', 'GR', 'H2', 'H3')
EXTRA_ARMS = {'subtype4': ('G23', 'G2R', 'P2', 'P3'), 'luma_lumb': ('G23', 'G2R', 'P2', 'P3'), 'idc_ilc': ('E2', 'ES', 'ER'), 'er_ihc': ()}
FUSION_EXPERTS = {'subtype4': ('G2', 'G3', 'GR', 'G23', 'G2R'), 'luma_lumb': ('G2', 'G3', 'GR', 'G23', 'G2R'), 'idc_ilc': ('G2', 'G3', 'GR', 'E2', 'ES', 'ER'), 'er_ihc': ('G2', 'G3', 'GR')}
WAVE1_ARMS = ('B', 'G2', 'G3', 'H2', 'H3')
IMAGE_ARMS = frozenset({'B', 'H2', 'H3', 'P2', 'P3'})
RESIDUAL_ARMS = frozenset({'GR', 'G2R'})
SCENE_ARMS = frozenset({'P2', 'P3', 'E2', 'ES', 'ER'})
GEOM_ARMS = frozenset({'G2', 'G3', 'GR', 'G23', 'G2R', 'H2', 'H3', 'P2', 'P3', 'E2', 'ES', 'ER'})
TASK_SPEC: dict[str, dict[str, Any]] = {'subtype4': {'classes': 4, 'label': 'subtype4', 'label_id': 'subtype4_id', 'primary': 'macro_f1', 'names': list(SUBTYPE4)}, 'luma_lumb': {'classes': 2, 'label': 'luma_lumb', 'label_id': 'luma_lumb_id', 'primary': 'auroc', 'names': list(LUMAB_IDS)}, 'idc_ilc': {'classes': 2, 'label': 'idc_ilc', 'label_id': 'idc_ilc_id', 'primary': 'auroc', 'names': list(HISTO_IDS)}, 'er_ihc': {'classes': 2, 'label': 'er', 'label_id': 'er_id', 'primary': 'auroc', 'names': list(ER_IDS)}}
RECIPES = {'B': {'lr': 0.0001, 'wd': 0.0001, 'epochs': 80, 'patience': 12, 'kind': 'B'}, 'G2': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'G'}, 'G3': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'G'}, 'GR': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'G'}, 'G23': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'G'}, 'G2R': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'G'}, 'H2': {'image_lr': 2e-05, 'geom_lr': 0.0001, 'wd': 0.0001, 'epochs': 80, 'patience': 12, 'kind': 'H'}, 'H3': {'image_lr': 2e-05, 'geom_lr': 0.0001, 'wd': 0.0001, 'epochs': 80, 'patience': 12, 'kind': 'H'}, 'P2': {'lr': 0.0001, 'wd': 0.0001, 'epochs': 80, 'patience': 12, 'kind': 'P', 'route_arm': 'B2'}, 'P3': {'lr': 0.0001, 'wd': 0.0001, 'epochs': 80, 'patience': 12, 'kind': 'P', 'route_arm': 'B3'}, 'E2': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'E'}, 'ES': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'E'}, 'ER': {'lr': 0.001, 'wd': 0.0001, 'epochs': 60, 'patience': 10, 'kind': 'E'}}
GEOM_MODE = {'G2': 'g2', 'G3': 'g3', 'GR': 'gr', 'G23': 'g23', 'G2R': 'g2r', 'H2': 'g2', 'H3': 'g3'}
COMPLETE_3D = frozenset({'H3', 'P3', 'B+G3', 'B+GR', 'B+G23', 'B+G2R', 'B+ES', 'B+ER'})
MATCHING_2D = {'H3': 'H2', 'P3': 'P2', 'B+G3': 'B+G2', 'B+GR': 'B+G2', 'B+G23': 'B+G2', 'B+G2R': 'B+G2', 'B+ES': 'B+E2', 'B+ER': 'B+E2'}
REMOTE_DOCS = Path(_resource_path('artifact_0038'))
from celllift.runtime import output_root
LOCAL_DOCS = output_root() / 'breast_patient_tasks' / 'report.md'
REMOTE_DOCS_conditional_geometry = REMOTE_DOCS.with_name('downstream-routes-conditional_geometry.md')
LOCAL_DOCS_conditional_geometry = LOCAL_DOCS.with_name('downstream-routes-conditional_geometry.md')
conditional_geometry_BATCH_ID = ROUTES_conditional_geometry_BATCH_ID

def is_conditional_geometry() -> bool:
    return BATCH_ID == conditional_geometry_BATCH_ID

def active_docs() -> tuple[Path, Path]:
    if is_conditional_geometry():
        return (REMOTE_DOCS_conditional_geometry, LOCAL_DOCS_conditional_geometry)
    return (REMOTE_DOCS, LOCAL_DOCS)

def resolved_arm_dir(task: str, arm: str) -> Path:
    path = arm_dir(task, arm)
    if is_conditional_geometry() and arm == 'B' and (not (path / 'predictions.parquet').is_file()):
        return paths.routes_set_encoding_result_root() / task / 'B' / f'seed{SEED}'
    return path

def task_arms(task: str) -> tuple[str, ...]:
    return COMMON_ARMS + tuple(EXTRA_ARMS[task])

def all_train_jobs() -> list[tuple[str, str]]:
    return [(task, arm) for task in TASKS for arm in task_arms(task)]

def all_fusion_jobs() -> list[tuple[str, str]]:
    return [(task, expert) for task in TASKS for expert in FUSION_EXPERTS[task]]

def result_root() -> Path:
    return paths.routes_result_root()

def arm_dir(task: str, arm: str) -> Path:
    return result_root() / task / arm / f'seed{SEED}'

def fuse_dir(task: str, expert: str) -> Path:
    return result_root() / task / 'fuse' / f'B+{expert}' / f'seed{SEED}'

def probe_dir(task: str) -> Path:
    return result_root() / 'probes' / task

def runtime_dir() -> Path:
    return result_root() / 'runtime'

def tables_dir() -> Path:
    return result_root() / 'tables'

def claim_dir(job_id: str) -> Path:
    return runtime_dir() / 'claims' / job_id.replace(':', '__')

def protocol_meta() -> dict[str, Any]:
    return {'protocol_id': PROTOCOL_ID, 'batch_id': BATCH_ID, 'seed': SEED}
