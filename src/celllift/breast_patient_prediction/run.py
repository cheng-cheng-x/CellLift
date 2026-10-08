from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import ResourcePath as Path
from .audit import audit_overlap
from .labels import build_labels
from .paths import data_root
from .sources import pin_sources
from .splits import build_splits
from .summarize import write_docs_summary
from .tiles import merge_tiles, sample_tile_shards
DEFAULT_DOCS = Path(_resource_path('artifact_0039'))
from celllift.runtime import output_root
LOCAL_DOCS = output_root() / 'breast_patient_tasks' / 'report.md'

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Prepare TCGA-BRCA four-task manifests.')
    parser.add_argument('command', choices=('pin-sources', 'build-labels', 'build-splits', 'sample-tiles', 'merge-tiles', 'audit-overlap', 'summarize', 'prepare-tables', 'all'))
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--docs-path', default=None)
    args = parser.parse_args(argv)
    if args.data_root:
        import os
        os.environ['TCGA_BRCA_DATA_ROOT'] = args.data_root
    command = args.command
    if command == 'pin-sources':
        print(pin_sources())
    elif command == 'build-labels':
        print(build_labels())
    elif command == 'build-splits':
        print(build_splits())
    elif command == 'sample-tiles':
        print(sample_tile_shards(shard_index=args.shard_index, shard_count=args.shard_count))
    elif command == 'merge-tiles':
        print(merge_tiles())
    elif command == 'audit-overlap':
        print(audit_overlap())
    elif command == 'summarize':
        docs = Path(args.docs_path) if args.docs_path else DEFAULT_DOCS if DEFAULT_DOCS.parent.is_dir() else LOCAL_DOCS
        print(write_docs_summary(docs))
    elif command == 'prepare-tables':
        pin_sources()
        build_labels()
        print(build_splits())
    elif command == 'all':
        pin_sources()
        build_labels()
        build_splits()
        sample_tile_shards(shard_index=args.shard_index, shard_count=args.shard_count)
        merge_tiles()
        audit_overlap()
        docs = Path(args.docs_path) if args.docs_path else DEFAULT_DOCS if DEFAULT_DOCS.parent.is_dir() else LOCAL_DOCS
        print(write_docs_summary(docs))
    print('data_root', data_root())
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
