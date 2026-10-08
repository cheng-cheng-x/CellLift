from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
TARGET_MPP = 0.46
ARVANITI_SOURCE_MPP = 0.23
ARVANITI_SCALE = ARVANITI_SOURCE_MPP / TARGET_MPP
LIZARD_SOURCE_MPP = 0.5
LIZARD_SCALE = LIZARD_SOURCE_MPP / TARGET_MPP
EDGE_NORMALIZER_UM = 471.04
RAY_COUNT = 36
K_NEIGHBORS = 12
RADIUS_UM = 60.0
DINO_CANVAS = 1024
ARVANITI_NATIVE = 3100
ARVANITI_CORE_PX = 1550
ARVANITI_PATCH_NATIVE = 750
ARVANITI_PATCH_TARGET = 375
ARVANITI_STRIDE_NATIVE = 375
ARVANITI_CENTER_NATIVE = 250
WHITE_LIMIT = 180
LIZARD_HALO_PX = 131
LIZARD_TILE_PX = 1024
LIZARD_OWN_PX = LIZARD_TILE_PX - 2 * LIZARD_HALO_PX
SEED = 42
ARVANITI_BOARDS = {'FIT': ('ZT111', 'ZT199', 'ZT204'), 'VAL': ('ZT76',), 'TEST': ('ZT80',)}
ARVANITI_BOARD_PREFIX = {'ZT76': 'ZT76_39', 'ZT80': 'ZT80_38', 'ZT111': 'ZT111_4', 'ZT199': 'ZT199_1', 'ZT204': 'ZT204_6'}
MASK_VALUE_MAP = {'0': 'benign', '1': 'G3', '2': 'G4', '3': 'G5', '4': 'unlabelled'}
LABEL_NAMES = {0: 'benign', 1: 'G3', 2: 'G4', 3: 'G5'}
LIZARD_CLASS_MAP = {1: 'neutrophil', 2: 'epithelial', 3: 'lymphocyte', 4: 'plasma', 5: 'eosinophil', 6: 'connective'}
LIZARD_PAPER_NUCLEI = {'consep': 6018, 'crag': 189043, 'dpath': 168510, 'glas': 55364, 'pannuke': 12978}
PALETTE_RGB = {(0, 255, 0): 0, (0, 0, 255): 1, (255, 255, 0): 2, (255, 0, 0): 3, (255, 255, 255): 4}
HOST_PRIORITY = ('Configured resource', 'Configured resource', 'Configured resource', 'Configured resource', 'Configured resource')
RECONSTRUCT_PYTHON = _resource_path('artifact_0008')
CELLPOSE_PYTHON = _resource_path('artifact_0025')
ProjectionScene_SOURCE = _resource_path('artifact_0032')
ProjectionScene_CHECKPOINT = _resource_path('artifact_0033')
ProjectionScene_MANIFEST = _resource_path('artifact_0034')
DINO_SOURCE = _resource_path('artifact_0035')
DINO_WEIGHTS = _resource_path('artifact_0036')
FEATURE_STATS = _resource_path('artifact_0037')
ARVANITI_RAW = Path(_resource_path('artifact_0055'))
LIZARD_RAW = Path(_resource_path('artifact_0056'))
ARVANITI_DATA = Path(_resource_path('artifact_0040'))
LIZARD_DATA = Path(_resource_path('artifact_0041'))
ARVANITI_RESULT = Path(_resource_path('artifact_0057'))
LIZARD_RESULT = Path(_resource_path('artifact_0058'))
CODE_ROOT = Path(_resource_path('artifact_0009'))
