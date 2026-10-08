from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='TCGA-BRCA downstream routes set_encoding')
    parser.add_argument('command', choices=('write-jobs', 'extract-global-dino', 'merge-global-dino', 'probe', 'train', 'fuse', 'fuse-watch', 'summarize', 'worker', 'progress'))
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--task', default=None)
    parser.add_argument('--arm', default=None)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--docs-path', default=None)
    args = parser.parse_args(argv)
    if args.data_root:
        os.environ['TCGA_BRCA_DATA_ROOT'] = args.data_root
    if args.command == 'write-jobs':
        from .schedule import write_jobs
        print(write_jobs())
    elif args.command == 'extract-global-dino':
        from .dino_global import extract_shard
        print(extract_shard(args.shard, device=args.device))
    elif args.command == 'merge-global-dino':
        from .dino_global import merge_global_dino
        print(merge_global_dino())
    elif args.command == 'probe':
        from .probe import fit_probe
        if not args.task:
            raise SystemExit('--task is required')
        print(fit_probe(args.task, device=args.device))
    elif args.command == 'train':
        from .train import train_arm
        if not args.task or not args.arm:
            raise SystemExit('--task and --arm are required')
        print(train_arm(args.task, args.arm, device=args.device))
    elif args.command == 'fuse':
        from .fuse import fuse_ready
        print(fuse_ready(args.task))
    elif args.command == 'fuse-watch':
        from .fuse import fuse_watch
        print(fuse_watch())
    elif args.command == 'summarize':
        from .fuse import collect_tables, render_results
        from celllift.runtime import ResourcePath as Path
        tables = collect_tables()
        print(render_results(Path(args.docs_path) if args.docs_path else None, tables))
    elif args.command == 'worker':
        from .schedule import worker_loop
        print(worker_loop(device=args.device))
    elif args.command == 'progress':
        from .schedule import progress
        print(progress())
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
