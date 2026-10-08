from celllift.runtime import resource_path as _public_resource
from .common import *
import zipfile

def main():
    archive = OUT / 'review_bundle.zip'
    files = []
    for p in OUT.rglob('*'):
        if not p.is_file() or p == archive:
            continue
        rel = p.relative_to(OUT)
        parts = rel.parts
        if any((x.startswith('shards') for x in parts)) or 'revisions' in parts or '_deps' in parts or ('logs' in parts):
            continue
        if p.name in ('graphs.parquet', 'patients.parquet', 'inventory.json'):
            continue
        if p.suffix in ('.npz', '.pt'):
            continue
        if 'model' in parts and (not any((x in parts for x in ('summary', 'core_summary', 'worker', 'background', 'local_reader1', 'local_reader2')))) and (p.name not in ('input_paths.json', 'ig_gate.json', 'ig_test_gate.json')):
            continue
        if p.name == 'leave_one_cluster_out.csv':
            continue
        if p.suffix not in ('.csv', '.json', '.md', '.pdf', '.png', '.svg', '.parquet'):
            continue
        files.append(p)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in files:
            z.write(p, p.relative_to(OUT))
    write(OUT / 'review_bundle_manifest.json', dict(files=[str(p.relative_to(OUT)) for p in files], remote_retained=['raw graph and node arrays', 'per-unit model records', 'leave-one-cluster-out full tables'], bytes=archive.stat().st_size))
    print('export', len(files), archive.stat().st_size, flush=True)
if __name__ == '__main__':
    main()
