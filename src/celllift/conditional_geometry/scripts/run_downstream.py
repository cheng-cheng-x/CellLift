from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import importlib.util
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.conditional_geometry.common import token_store
from celllift.conditional_geometry.experiment import ARMS, ExperimentArm
import celllift.conditional_geometry.models as models
from celllift.conditional_geometry.protocol import CONFIRM_SEEDS, ENCODERS, PROTOCOL_ID
conditional_geometry_ROOT = Path(__file__).resolve().parents[1]
set_encoding_ROOT = conditional_geometry_ROOT.parent / 'set_encoding'

def _load_set_encoding_trainer():
    if str(set_encoding_ROOT) not in sys.path:
        sys.path.append(str(set_encoding_ROOT))
    import celllift.set_encoding.tasks as tasks
    spec = importlib.util.spec_from_file_location('conditional_geometry_audited_trainer', set_encoding_ROOT / 'scripts' / 'run_downstream.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load audited validation trainer')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SCREENING_PROTOCOL_ID != PROTOCOL_ID:
        raise RuntimeError('trainer did not bind to the conditional_geometry protocol')
    return module

def run_residual_downstream(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', encoders: Sequence[str] | None=None, seeds: Sequence[int] | None=None, arms: Sequence[str] | None=None) -> dict[str, Any]:
    trainer = _load_set_encoding_trainer()
    token_store.configure(cfg)
    runtime_cfg = dict(cfg)
    runtime_cfg['paths'] = dict(cfg['paths'])
    runtime_cfg['training'] = dict(cfg['downstream'])
    budget = int(cfg['downstream']['token_budget'])
    runtime_cfg['training']['token_budget'] = {'meanpool': budget, 'deepsets': budget}
    original_load_rgb = trainer._load_rgb
    original_read_rows = trainer._read_rows
    set_encoding_data = Path(cfg['paths']['set_encoding_data_root'])

    def redirected_rgb(requested: Path):
        fold_folder = requested.parent.parent.name
        seed_folder = requested.parent.name
        return original_load_rgb(set_encoding_data / '05_rgb_features' / fold_folder / seed_folder / requested.name)
    trainer._load_rgb = redirected_rgb

    def train_only_metadata(path: Path):
        if path.name in {'graph_index.parquet', 'patch_manifest.parquet'}:
            import pyarrow.parquet as pq
            split_column = 'split' if path.name == 'graph_index.parquet' else 'official_split'
            table = pq.read_table(path, filters=[[(split_column, '=', 'TRAIN')], [(split_column, '=', 'train')]], partitioning=None)
            rows = table.to_pylist()
            if any((str(row.get(split_column, '')).lower() != 'train' for row in rows)):
                raise RuntimeError('official TEST metadata crossed the conditional_geometry filter')
            return rows
        return original_read_rows(path)
    trainer._read_rows = train_only_metadata
    original_persist_shuffle = trainer._persist_shuffle_mapping

    def persist_conditional_geometry_shuffle(store, data_root: Path, *, phase: str, fold: int, seed: int):
        payload = original_persist_shuffle(store, data_root, phase=phase, fold=fold, seed=seed)
        payload['constraints'] = 'cross-graph,same-partition,role-specific-count-rank-decile,no-cross-decile-fallback'
        manifest_path = Path(data_root) / '00_manifest' / 'controls' / phase / f'fold_{fold:02d}' / f'seed_{seed}' / 'manifest.json'
        trainer._atomic_json(manifest_path, payload)
        return payload
    trainer._persist_shuffle_mapping = persist_conditional_geometry_shuffle
    selected_encoders = tuple(encoders or cfg['downstream'].get('encoders', ENCODERS))
    selected_seeds = tuple(map(int, seeds or cfg['downstream'].get('seeds', CONFIRM_SEEDS)))
    selected_arms: tuple[ExperimentArm, ...] = tuple((ARMS[name] for name in arms or ARMS.keys()))
    return trainer.run_grid(runtime_cfg, selected_arms, encoders=selected_encoders, seeds=selected_seeds, fold=int(fold), validation_only=True, device=device)
