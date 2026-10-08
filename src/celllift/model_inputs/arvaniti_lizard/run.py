from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import subprocess
import sys
from celllift.runtime import ResourcePath as Path
_MODEL_INPUT = Path(__file__).resolve().parents[1]
if str(_MODEL_INPUT) not in sys.path:
    sys.path.insert(0, str(_MODEL_INPUT))
from celllift.model_inputs.arvaniti_lizard.audit import audit_arvaniti, audit_lizard
from celllift.model_inputs.arvaniti_lizard.constants import ARVANITI_DATA, CELLPOSE_PYTHON, LIZARD_DATA
from celllift.model_inputs.arvaniti_lizard.io_utils import atomic_json
from celllift.model_inputs.arvaniti_lizard.official import freeze_inference_windows, write_tissue_masks
from celllift.model_inputs.arvaniti_lizard.process import build_dataset_graphs, lizard_conditional_geometry_qc, materialize_arvaniti_inference, materialize_arvaniti_windows, materialize_lizard, reuse_projection_scene_conditional_geometry, segment_arvaniti_cores, select_engineering_batch, standardize_arvaniti_cores
from celllift.model_inputs.arvaniti_lizard.qc import write_engineering_qc
from celllift.model_inputs.arvaniti_lizard.splits import freeze_arvaniti, freeze_lizard
from celllift.model_inputs.arvaniti_lizard.projection_scene_cache import cache_projection_scene, infer_projection_scene

def _dump(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))

def main() -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('audit', 'splits', 'engineering-select', 'standardize', 'segment', 'windows', 'lizard-tiles', 'graphs', 'engineering-gpu', 'engineering-cpu', 'engineering-pipeline', 'engineering-infer', 'projection_scene-cache', 'projection_scene-infer', 'qc', 'full-qc', 'full-cpu', 'full-pipeline', 'lizard-conditional_geometry', 'arvaniti-official', 'conditional_geometry-graphs', 'lizard-conditional_geometry-qc', 'reuse-conditional_geometry'))
    parser.add_argument('--dataset', choices=('arvaniti', 'lizard', 'both'), default='both')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--revision', choices=('set_encoding', 'conditional_geometry'), default='set_encoding')
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    if args.stage == 'audit':
        payload = {}
        if args.dataset in {'arvaniti', 'both'}:
            payload['arvaniti'] = audit_arvaniti()
        if args.dataset in {'lizard', 'both'}:
            payload['lizard'] = audit_lizard(revision=args.revision)
        _dump(payload)
        return
    if args.stage == 'splits':
        payload = {}
        if args.dataset in {'arvaniti', 'both'}:
            payload['arvaniti'] = freeze_arvaniti()
        if args.dataset in {'lizard', 'both'}:
            payload['lizard'] = freeze_lizard(revision=args.revision)
        _dump(payload)
        return
    if args.stage == 'engineering-select':
        _dump(select_engineering_batch())
        return
    if args.stage == 'standardize':
        _dump(standardize_arvaniti_cores(engineering=args.engineering, shard_id=args.shard_id, num_shards=args.num_shards))
        return
    if args.stage == 'segment':
        _dump(segment_arvaniti_cores(engineering=args.engineering, shard_id=args.shard_id, num_shards=args.num_shards))
        return
    if args.stage == 'windows':
        _dump(materialize_arvaniti_windows(engineering=args.engineering))
        return
    if args.stage == 'lizard-tiles':
        _dump(materialize_lizard(engineering=args.engineering, revision=args.revision))
        return
    if args.stage == 'graphs':
        payload = {}
        if args.dataset in {'arvaniti', 'both'}:
            payload['arvaniti'] = build_dataset_graphs('arvaniti', engineering=args.engineering, revision=args.revision)
        if args.dataset in {'lizard', 'both'}:
            payload['lizard'] = build_dataset_graphs('lizard', engineering=args.engineering, revision=args.revision)
        _dump(payload)
        return
    if args.stage == 'engineering-cpu':
        select_engineering_batch()
        standardize_arvaniti_cores(engineering=True)
        materialize_lizard(engineering=True)
        _dump({'status': 'CPU_READY'})
        return
    if args.stage in {'engineering-gpu', 'engineering-pipeline'}:
        select_engineering_batch()
        standardize_arvaniti_cores(engineering=True)
        tiles = materialize_lizard(engineering=True)
        env = os.environ.copy()
        extra = str(_MODEL_INPUT)
        env['PYTHONPATH'] = extra + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        subprocess.check_call([CELLPOSE_PYTHON, '-m', 'arvaniti_lizard.run', 'segment', '--engineering'], cwd=str(_MODEL_INPUT), env=env)
        windows = materialize_arvaniti_windows(engineering=True)
        graphs_a = build_dataset_graphs('arvaniti', engineering=True)
        graphs_l = build_dataset_graphs('lizard', engineering=True)
        cache_a = cache_projection_scene('arvaniti', engineering=True)
        cache_l = cache_projection_scene('lizard', engineering=True)
        infer_a = infer_projection_scene('arvaniti', engineering=True, limit=args.limit)
        infer_l = infer_projection_scene('lizard', engineering=True, limit=args.limit)
        qc = write_engineering_qc()
        payload = {'windows': windows, 'tiles': tiles, 'graphs_arvaniti': graphs_a, 'graphs_lizard': graphs_l, 'cache_arvaniti': cache_a, 'cache_lizard': cache_l, 'infer_arvaniti': infer_a, 'infer_lizard': infer_l, 'qc': qc}
        atomic_json(Path(ARVANITI_DATA) / '05_qc' / 'engineering_gate.json', payload)
        atomic_json(Path(LIZARD_DATA) / '05_qc' / 'engineering_gate.json', payload)
        _dump(payload)
        return
    if args.stage == 'engineering-infer':
        infer_a = infer_projection_scene('arvaniti', engineering=True, limit=args.limit)
        infer_l = infer_projection_scene('lizard', engineering=True, limit=args.limit)
        qc = write_engineering_qc()
        payload = {'infer_arvaniti': infer_a, 'infer_lizard': infer_l, 'qc': qc}
        atomic_json(Path(ARVANITI_DATA) / '05_qc' / 'engineering_gate.json', payload)
        atomic_json(Path(LIZARD_DATA) / '05_qc' / 'engineering_gate.json', payload)
        _dump(payload)
        return
    if args.stage == 'projection_scene-cache':
        _dump(cache_projection_scene(args.dataset if args.dataset != 'both' else 'arvaniti', engineering=args.engineering, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards, revision=args.revision))
        return
    if args.stage == 'projection_scene-infer':
        _dump(infer_projection_scene(args.dataset if args.dataset != 'both' else 'arvaniti', engineering=args.engineering, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards, revision=args.revision))
        return
    if args.stage == 'lizard-conditional_geometry':
        audit = audit_lizard(revision='conditional_geometry')
        splits = freeze_lizard(revision='conditional_geometry')
        tiles = materialize_lizard(revision='conditional_geometry')
        _dump({'audit': audit, 'splits': splits, 'tiles': tiles})
        return
    if args.stage == 'arvaniti-official':
        tissue = write_tissue_masks()
        inference = freeze_inference_windows()
        windows = materialize_arvaniti_inference()
        _dump({'tissue': tissue, 'inference': inference, 'windows': windows})
        return
    if args.stage == 'lizard-conditional_geometry-qc':
        _dump(lizard_conditional_geometry_qc())
        return
    if args.stage == 'reuse-conditional_geometry':
        payload = {}
        if args.dataset in {'arvaniti', 'both'}:
            payload['arvaniti'] = reuse_projection_scene_conditional_geometry('arvaniti')
        if args.dataset in {'lizard', 'both'}:
            payload['lizard'] = reuse_projection_scene_conditional_geometry('lizard')
        _dump(payload)
        return
    if args.stage == 'conditional_geometry-graphs':
        payload = {}
        if args.dataset in {'arvaniti', 'both'}:
            payload['arvaniti'] = build_dataset_graphs('arvaniti', revision='conditional_geometry')
        if args.dataset in {'lizard', 'both'}:
            payload['lizard'] = build_dataset_graphs('lizard', revision='conditional_geometry')
        _dump(payload)
        return
    if args.stage == 'qc':
        _dump(write_engineering_qc())
        return
    if args.stage == 'full-qc':
        from celllift.model_inputs.arvaniti_lizard.qc import write_full_qc
        _dump(write_full_qc())
        return
    if args.stage == 'full-cpu':
        payload = {'windows': materialize_arvaniti_windows(engineering=False), 'tiles': materialize_lizard(engineering=False), 'graphs_arvaniti': build_dataset_graphs('arvaniti', engineering=False), 'graphs_lizard': build_dataset_graphs('lizard', engineering=False)}
        _dump(payload)
        return
    if args.stage == 'full-pipeline':
        from celllift.model_inputs.arvaniti_lizard.process import prepare_tree
        prepare_tree(Path(ARVANITI_DATA))
        prepare_tree(Path(LIZARD_DATA))
        standardize_arvaniti_cores(engineering=False)
        env = os.environ.copy()
        extra = str(_MODEL_INPUT)
        env['PYTHONPATH'] = extra + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        subprocess.check_call([CELLPOSE_PYTHON, '-m', 'arvaniti_lizard.run', 'segment'], cwd=str(_MODEL_INPUT), env=env)
        payload = {'windows': materialize_arvaniti_windows(engineering=False), 'tiles': materialize_lizard(engineering=False), 'graphs_arvaniti': build_dataset_graphs('arvaniti', engineering=False), 'graphs_lizard': build_dataset_graphs('lizard', engineering=False), 'cache_arvaniti': cache_projection_scene('arvaniti', engineering=False), 'cache_lizard': cache_projection_scene('lizard', engineering=False)}
        atomic_json(Path(ARVANITI_DATA) / '05_qc' / 'full_cache_status.json', payload)
        atomic_json(Path(LIZARD_DATA) / '05_qc' / 'full_cache_status.json', payload)
        _dump(payload)
        return
    raise SystemExit(args.stage)
if __name__ == '__main__':
    main()
