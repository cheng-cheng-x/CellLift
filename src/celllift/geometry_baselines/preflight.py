from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import Counter
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from .models import PaperFSConv, build_paper_crc_resnet18, crc_adam_groups
from .paper_rgb import read_rows
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS, TOKEN_DIM
EXPECTED = {'sicapv2': {'labels': 12081, 'geometry': 12081, 'folds': 4}, 'tcga_crc_msi': {'labels': 51918, 'geometry': 51915, 'folds': 5}}

def full_preflight(cfg: Mapping[str, Any], *, verify_train_sources: bool=True) -> dict[str, Any]:
    dataset = str(cfg['dataset'])
    expected = EXPECTED[dataset]
    model_input = Path(cfg['paths']['model_input_root'])
    labels_path = model_input / '04_labels_splits' / 'labels_splits.parquet'
    graph_path = model_input / '03_graph_cache' / 'graph_index.parquet'
    labels = read_rows(labels_path)
    graphs = read_rows(graph_path)
    if len(labels) != expected['labels']:
        raise RuntimeError(f"label coverage {len(labels)} != {expected['labels']}")
    if len(graphs) != expected['geometry']:
        raise RuntimeError(f"geometry coverage {len(graphs)} != {expected['geometry']}")
    if len({str(row['graph_id']) for row in labels}) != len(labels):
        raise RuntimeError('duplicate label graph IDs')
    graph_ids = {str(row['graph_id']) for row in graphs}
    missing_geometry = [row for row in labels if str(row['graph_id']) not in graph_ids]
    if len(missing_geometry) != expected['labels'] - expected['geometry']:
        raise RuntimeError('geometry exclusion count drift')
    if dataset == 'tcga_crc_msi':
        reasons = Counter((str(row.get('exclusion_reason', '')) for row in missing_geometry))
        excluded_ids = sorted((str(row['graph_id']) for row in missing_geometry))
    else:
        reasons, excluded_ids = (Counter(), [])
    train_rows = [row for row in labels if str(row['official_split']).lower() == 'train']
    folds = set((int(row['validation_fold']) for row in train_rows))
    if folds != set(range(expected['folds'])):
        raise RuntimeError(f'validation fold coverage drift: {folds}')
    if verify_train_sources:
        from PIL import Image
        for row in train_rows:
            path = Path(str(row['source_path']))
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise RuntimeError(f'non-paper source dimensions {image.size}: {path}')
    fsconv = PaperFSConv()
    if sum((parameter.numel() for parameter in fsconv.parameters())) != 630276:
        raise AssertionError('FSConv parameter count mismatch')
    crc, trainable = build_paper_crc_resnet18(imagenet_weights=False)
    groups = crc_adam_groups(crc)
    if not trainable or {group['name'] for group in groups} != {'layer4.1', 'classifier'}:
        raise AssertionError('CRC trainable set mismatch')
    return {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': dataset, 'label_rows': len(labels), 'geometry_rows': len(graphs), 'train_rows': len(train_rows), 'validation_folds': sorted(folds), 'official_test_manifest_rows': len(labels) - len(train_rows), 'official_test_pixels_touched': False, 'excluded_geometry_graph_ids': excluded_ids, 'excluded_reasons': dict(reasons), 'registered_arms': list(ARMS), 'arm_count': len(ARMS), 'token_dim': TOKEN_DIM, 'encoders': list(ENCODERS), 'seeds': list(SEEDS), 'fsconv_parameters': 630276, 'crc_trainable_names': list(trainable), 'source_check': 'all official TRAIN source files opened and verified 512x512' if verify_train_sources else 'disabled'}
