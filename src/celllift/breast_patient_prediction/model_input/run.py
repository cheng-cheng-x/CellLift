from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import os
from celllift.runtime import ResourcePath as Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='TCGA-BRCA shared model-input cache')
    parser.add_argument('command', choices=('write-config', 'write-jobs', 'export-rgb', 'pilot', 'worker', 'segment-slide', 'progress', 'wait', 'pack', 'summarize'))
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', default='rgb', choices=('rgb', 'mask', 'graph', 'scene'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--docs-path', default=None)
    parser.add_argument('--job-json', default=None)
    args = parser.parse_args(argv)
    if args.data_root:
        import os
        os.environ['TCGA_BRCA_DATA_ROOT'] = args.data_root
    from .. import paths
    if args.command == 'write-config':
        from .config import write_config
        print(write_config())
    elif args.command == 'write-jobs':
        from .schedule import write_job_files
        print(write_job_files())
    elif args.command == 'export-rgb':
        from .rgb import export_rgb
        print(export_rgb(workers=args.workers))
    elif args.command == 'pilot':
        from .pilot import run_pilot
        print(run_pilot(device=args.device))
    elif args.command == 'segment-slide':
        from ..io_utils import read_json
        from .segment import segment_slide_job
        if not args.job_json:
            raise SystemExit('--job-json is required for segment-slide')
        print(segment_slide_job(read_json(Path(args.job_json))))
    elif args.command == 'worker':
        from .schedule import worker_loop
        print(worker_loop(args.stage, device=args.device, retry_failed=args.retry_failed))
    elif args.command == 'progress':
        from .schedule import progress
        print(progress())
    elif args.command == 'wait':
        from .schedule import wait_complete
        print(wait_complete())
    elif args.command == 'pack':
        from .pack import coverage_summary, write_indices
        print(write_indices())
        print(coverage_summary())
    elif args.command == 'summarize':
        from .summarize import write_results_page
        docs = Path(args.docs_path) if args.docs_path else None
        print(write_results_page(docs))
    print('data_root', paths.data_root())
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
