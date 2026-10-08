from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import importlib.util
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.conditional_geometry.common import mask_token_store
import celllift.conditional_geometry.common as common
import celllift.conditional_geometry.mask_experiment as mask_experiment
from celllift.conditional_geometry.mask_experiment import ARMS, ExperimentArm
from celllift.conditional_geometry.mask_protocol import CONFIRM_SEEDS, ENCODERS, MASK_PROTOCOL_ID
import celllift.conditional_geometry.models as models
from celllift.conditional_geometry.models.mask_downstream import MaskDualSetFusion
ROOT = Path(__file__).resolve().parents[1]
set_encoding_ROOT = ROOT.parent / 'set_encoding'

def _load_set_encoding_trainer(experiment_module, fusion_class, expected_protocol: str):
    if str(set_encoding_ROOT) not in sys.path:
        sys.path.append(str(set_encoding_ROOT))
    common.token_store = mask_token_store
    sys.modules['common.token_store'] = mask_token_store
    sys.modules['experiment'] = experiment_module
    models.DualSetFusion = fusion_class
    import celllift.set_encoding.tasks as tasks
    spec = importlib.util.spec_from_file_location('set_encoder_mask_mask_conditioning_audited_trainer', set_encoding_ROOT / 'scripts' / 'run_downstream.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load audited validation trainer for mask mask_conditioning')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCREENING_PROTOCOL_ID != expected_protocol:
        raise RuntimeError('trainer did not bind to mask-mask_conditioning protocol')
    return module

def run_mask_residual_downstream(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', encoders: Sequence[str] | None=None, seeds: Sequence[int] | None=None, arms: Sequence[str] | None=None) -> dict[str, Any]:
    rgb_mode = str(cfg.get('probe', {}).get('conditioning_mode')) == 'rgb_plus_mask_rays'
    if rgb_mode:
        import celllift.conditional_geometry.rgb_mask_experiment as rgb_mask_experiment
        from celllift.conditional_geometry.models.rgb_mask_downstream import RGBMaskDualSetFusion
        from celllift.conditional_geometry.rgb_mask_protocol import RGB_MASK_PROTOCOL_ID
        experiment_module = rgb_mask_experiment
        fusion_class = RGBMaskDualSetFusion
        expected_protocol = RGB_MASK_PROTOCOL_ID
    elif str(cfg.get('protocol_id')) == 'matched_mask_full3d_vs_residual3d_raw_residual_comparison':
        import celllift.conditional_geometry.full3d_experiment as full3d_experiment
        from celllift.conditional_geometry.full3d_protocol import PROTOCOL_ID as FULL3D_PROTOCOL_ID
        experiment_module = full3d_experiment
        fusion_class = MaskDualSetFusion
        expected_protocol = FULL3D_PROTOCOL_ID
    else:
        experiment_module = mask_experiment
        fusion_class = MaskDualSetFusion
        expected_protocol = MASK_PROTOCOL_ID
    if str(cfg.get('protocol_id', expected_protocol)) != expected_protocol:
        raise RuntimeError('mask downstream config protocol mismatch')
    trainer = _load_set_encoding_trainer(experiment_module, fusion_class, expected_protocol)
    mask_token_store.configure(cfg)
    runtime_cfg = dict(cfg)
    runtime_cfg['paths'] = dict(cfg['paths'])
    runtime_cfg['training'] = dict(cfg['downstream'])
    budget = int(cfg['downstream']['token_budget'])
    runtime_cfg['training']['token_budget'] = {'meanpool': budget, 'deepsets': budget}
    if rgb_mode:
        original_rgb_loader = trainer._load_rgb
        rgb_path = Path(cfg['paths']['set_encoding_data_root']) / '05_rgb_features' / f'fold_{int(fold):02d}' / 'seed_42' / 'rgb_features.parquet'
        trainer._load_rgb = lambda _: original_rgb_loader(rgb_path)
    else:
        trainer._load_rgb = lambda _: {}
    original_read_rows = trainer._read_rows

    def train_only_metadata(path: Path):
        if path.name in {'graph_index.parquet', 'patch_manifest.parquet'}:
            import pyarrow.parquet as pq
            split_column = 'split' if path.name == 'graph_index.parquet' else 'official_split'
            rows = pq.read_table(path, filters=[[(split_column, '=', 'TRAIN')], [(split_column, '=', 'train')]], partitioning=None).to_pylist()
            if any((str(row.get(split_column, '')).lower() != 'train' for row in rows)):
                raise RuntimeError('official TEST metadata crossed the mask-mask_conditioning filter')
            return rows
        return original_read_rows(path)
    trainer._read_rows = train_only_metadata

    def persist_mask_shuffle(store, data_root: Path, *, phase: str, fold: int, seed: int):
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path(data_root) / '00_manifest' / 'controls' / phase / f'fold_{fold:02d}' / f'seed_{seed}'
        destination, manifest_path = (root / 'anchor_shuffle.parquet', root / 'manifest.json')
        if manifest_path.is_file() and destination.is_file():
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous.get('status') == 'PASS' and previous.get('seed') == seed and (previous.get('fold') == fold) and (previous.get('phase') == phase) and (previous.get('sha256') == trainer._sha256(destination)):
                return previous
            raise RuntimeError(f'existing mask shuffle mapping fails provenance validation: {manifest_path}')
        root.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
        writer = None
        total = len(store._row_graph)
        try:
            for start in range(0, total, 500000):
                stop = min(total, start + 500000)
                donor = store._donor[start:stop]
                table = pa.table({'graph_id': pa.array(store._row_graph[start:stop]), 'anchor_id': pa.array(store._row_anchor[start:stop]), 'donor_graph_id': pa.array(store._row_graph[donor]), 'donor_anchor_id': pa.array(store._row_anchor[donor]), 'partition': pa.array(store._row_role[start:stop]), 'count_decile': pa.array(store._row_decile[start:stop]), 'donor_count_decile': pa.array(store._row_decile[donor])})
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table, row_group_size=100000)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise RuntimeError('mask shuffle mapping is empty')
        os.replace(temporary, destination)
        payload = {'status': 'PASS', 'phase': phase, 'fold': int(fold), 'seed': int(seed), 'rows': int(total), 'path': str(destination), 'sha256': trainer._sha256(destination), 'constraints': 'cross-graph,same-partition,role-specific-count-rank-decile,no-cross-decile-fallback', 'writer': 'arrow-columnar-batches-mask_conditioning'}
        trainer._atomic_json(manifest_path, payload)
        return payload
    trainer._persist_shuffle_mapping = persist_mask_shuffle
    selected_encoders = tuple(encoders or cfg['downstream'].get('encoders', ENCODERS))
    selected_seeds = tuple(map(int, seeds or cfg['downstream'].get('seeds', CONFIRM_SEEDS)))
    arm_table = experiment_module.ARMS
    selected_arms = tuple((arm_table[name] for name in arms or arm_table.keys()))
    return trainer.run_grid(runtime_cfg, selected_arms, encoders=selected_encoders, seeds=selected_seeds, fold=int(fold), validation_only=True, device=device)
