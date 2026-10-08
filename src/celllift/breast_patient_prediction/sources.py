from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import urllib.request
from celllift.runtime import ResourcePath as Path
from typing import Any
from . import paths
from .io_utils import sha256_file, write_json
from .protocol import CBIO_COMMIT, CBIO_EXPECTED_BYTES, CBIO_LFS_SHA256, CBIO_POINTER_URL, CBIO_URL, PROTOCOL_ID

def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'tcga-brca-downstream-set_encoding'})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())

def pin_sources(root: Path | None=None) -> dict[str, Any]:
    base = paths.ensure_tree(root)
    source = paths.source_dir(base)
    cbio_path = source / 'data_clinical_patient.txt'
    pointer_path = source / 'data_clinical_patient.lfs_pointer.txt'
    _download(CBIO_POINTER_URL, pointer_path)
    _download(CBIO_URL, cbio_path)
    cbio_sha = sha256_file(cbio_path)
    if cbio_path.stat().st_size != CBIO_EXPECTED_BYTES or cbio_sha != CBIO_LFS_SHA256:
        raise RuntimeError(f'cBioPortal file mismatch: size={cbio_path.stat().st_size} sha={cbio_sha}')
    clinical = Path(paths.CLINICAL_PATIENT)
    inventory = Path(paths.STAGE1_SLIDES)
    record = {'protocol_id': PROTOCOL_ID, 'cbio': {'url': CBIO_URL, 'pointer_url': CBIO_POINTER_URL, 'commit': CBIO_COMMIT, 'path': str(cbio_path), 'sha256': cbio_sha, 'bytes': cbio_path.stat().st_size, 'lfs_oid': CBIO_LFS_SHA256}, 'clinical_patient': {'path': str(clinical), 'sha256': sha256_file(clinical), 'bytes': clinical.stat().st_size}, 'stage1_inventory': {'path': str(inventory), 'sha256': sha256_file(inventory), 'bytes': inventory.stat().st_size}}
    record['sha256'] = write_json(source / 'pin.json', record)
    return record
