from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from celllift.runtime import ResourcePath as Path
from celllift.runtime import json
from types import ModuleType
import sys
from .io_utils import sha256_file, sha256_tree

@dataclass(frozen=True)
class ProjectionSceneModules:
    package: ModuleType
    data: ModuleType
    model: ModuleType
    geometry: ModuleType
    reference: ModuleType
    compatibility_schemas: ModuleType
    cache_io: ModuleType
    he_features: ModuleType

def load_projection_scene(source_root: str | Path, alias: str='projection_scene_core') -> ProjectionSceneModules:
    root = Path(source_root).resolve()
    src = root / 'src'
    if not (src / '__init__.py').is_file():
        raise FileNotFoundError(src / '__init__.py')
    if alias not in sys.modules:
        spec = spec_from_file_location(alias, src / '__init__.py', submodule_search_locations=[str(src)])
        if spec is None or spec.loader is None:
            raise RuntimeError('cannot construct isolated ProjectionScene package')
        package = module_from_spec(spec)
        sys.modules[alias] = package
        spec.loader.exec_module(package)
    return ProjectionSceneModules(sys.modules[alias], import_module(f'{alias}.data'), import_module(f'{alias}.model'), import_module(f'{alias}.geometry'), import_module(f'{alias}.reference'), import_module(f'{alias}.upstream.compatibility_schemas'), import_module(f'{alias}.upstream.cache_io'), import_module(f'{alias}.upstream.he_feature_cache'))

def locked_hashes(source_root: str | Path, checkpoint: str | Path, input_manifest: str | Path, dino_source: str | Path, dino_weights: str | Path) -> dict[str, str]:
    checkpoint = Path(checkpoint)
    complete = checkpoint.parents[1] / 'runtime' / 'complete.json'
    payload = {'projection_scene_source_sha256': sha256_tree(Path(source_root) / 'src'), 'projection_scene_checkpoint_sha256': sha256_file(checkpoint), 'projection_scene_input_manifest_sha256': sha256_file(input_manifest), 'dinov2_source_sha256': sha256_tree(dino_source), 'dinov2_weights_sha256': sha256_file(dino_weights)}
    if complete.is_file():
        record = json.loads(complete.read_text(encoding='utf-8'))
        payload['projection_scene_complete_record_sha256'] = sha256_file(complete)
        payload['projection_scene_complete_record_source_sha256'] = str(record.get('source_sha256', ''))
        payload['projection_scene_complete_record_status'] = str(record.get('status', ''))
    return payload
