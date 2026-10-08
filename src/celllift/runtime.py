"""Resource configuration shared by the CellLift analysis modules."""
from pathlib import Path
import importlib
import json as _json
import os
ROOT = Path(__file__).resolve().parents[2]
ResourcePath = Path

def configuration():
    file = Path(os.environ.get('CELLLIFT_CONFIG', ROOT / 'configs/resources.json'))
    return (file, _json.loads(file.read_text(encoding='utf-8')) if file.exists() else {})

def resource_path(key):
    file, config = configuration()
    value = config.get('resources', {}).get(key)
    if not value:
        index_file = ROOT / 'docs/resource_index.json'
        index = _json.loads(index_file.read_text(encoding='utf-8')) if index_file.exists() else {}
        packaged = index.get(key, {}).get('packaged')
        if packaged:
            return str(ROOT / packaged)
    if not value:
        raise ValueError(f'Configure resource {key} in configs/resources.json')
    path = Path(os.path.expandvars(value)).expanduser()
    return str(path if path.is_absolute() else (file.parent / path).resolve())

def resolve(value):
    if isinstance(value, str) and value.startswith('resource://'):
        return resource_path(value.removeprefix('resource://'))
    if isinstance(value, list):
        return [resolve(x) for x in value]
    if isinstance(value, dict):
        return {key: resolve(item) for key, item in value.items()}
    return value

class JSON:

    def __getattr__(self, name):
        return getattr(_json, name)

    def load(self, *args, **kwargs):
        return resolve(_json.load(*args, **kwargs))

    def loads(self, *args, **kwargs):
        return resolve(_json.loads(*args, **kwargs))

class YAML:

    def __getattr__(self, name):
        function = getattr(importlib.import_module('yaml'), name)
        if name in {'load', 'safe_load', 'full_load'}:
            return lambda *args, **kwargs: resolve(function(*args, **kwargs))
        return function

class Torch:

    def __getattr__(self, name):
        return getattr(importlib.import_module('torch'), name)
json = JSON()
yaml = YAML()
torch = Torch()

def output_root():
    file, config = configuration()
    path = Path(config.get('output_root', '../runs'))
    return path if path.is_absolute() else (file.parent / path).resolve()

def activate():
    """Package imports require no machine-specific initialization."""
    return None
