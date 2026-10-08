from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from celllift.runtime import ResourcePath as Path
SEED = 42
TARGET_MPP = 0.46
CODE_ROOT = Path(_resource_path('artifact_0009'))
ARVANITI_DATA = Path(_resource_path('artifact_0040'))
LIZARD_DATA = Path(_resource_path('artifact_0041'))
_RESULT_PARENT = Path(_resource_path('artifact_0011'))
BASE_RESULT_ROOT = _RESULT_PARENT / 'arvaniti_lizard_projection_scene_seed42'
EVALUATION_RESULT_ROOT = _RESULT_PARENT / 'arvaniti_lizard_projection_scene_seed42_corrected_evaluation'
from celllift.runtime import output_root
RESULT_ROOT = Path(os.environ['AL_RESULT_ROOT']) if os.environ.get('AL_RESULT_ROOT') else output_root() / 'core_and_nucleus'
RECONSTRUCT_PYTHON = _resource_path('artifact_0008')
HOST_PRIORITY = ('Configured resource', 'Configured resource', 'Configured resource', 'Configured resource', 'Configured resource')
ProjectionScene_CHECKPOINT_SHA = '718d6445489266ec0708443262cae8b25aa2e458c963d11eaa9bb922aca06ee3'
AUTHOR_REPO = _resource_path('artifact_0042')
AUTHOR_WEIGHTS = Path(AUTHOR_REPO) / 'model_weights' / 'MobileNet_Gleason_weights.h5'
QWK_NAMES = ('benign', 'Gleason 6', 'Gleason 7', 'Gleason 8', 'Gleason 9', 'Gleason 10')
LIZARD_CLASS_NAMES = ('neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective')
