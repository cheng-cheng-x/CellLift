from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import re
PATIENT_RE = re.compile('^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', re.IGNORECASE)
DX_RE = re.compile('(?:^|[-_.])DX[A-Z0-9]*(?:[-_.]|$)', re.IGNORECASE)
NA_VALUES = {'', 'na', 'nan', 'none', 'null', '[not available]', '[not evaluated]', '[unknown]', '[not applicable]'}

def patient_id(text: str | None) -> str:
    match = PATIENT_RE.search(str(text or '').strip())
    return match.group(1).upper() if match else ''

def is_dx(filename: str) -> bool:
    return bool(DX_RE.search(filename_stem(filename)))

def filename_stem(filename: str) -> str:
    name = str(filename or '')
    if '/' in name or '\\' in name:
        name = name.replace('\\', '/').rsplit('/', 1)[-1]
    if name.lower().endswith('.svs'):
        name = name[:-4]
    return name

def sample_code(filename: str) -> str:
    parts = filename_stem(filename).split('-')
    return parts[3].upper() if len(parts) >= 4 else ''

def sample_type(filename: str) -> str:
    code = sample_code(filename)
    return code[:2] if len(code) >= 2 else ''

def keep_primary_dx(filename: str) -> bool:
    return is_dx(filename) and sample_type(filename) == '01'

def missing(value: object) -> bool:
    text = str(value or '').strip()
    return text.lower() in NA_VALUES
