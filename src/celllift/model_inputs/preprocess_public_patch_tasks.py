from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import subprocess
import sys
from celllift.runtime import ResourcePath as Path
from celllift.model_inputs.common.graph import run_graph
from celllift.model_inputs.common.inventory import build_inventory
from celllift.model_inputs.common.standardize import run_standardize
from celllift.model_inputs.common.segment import run_segment
from celllift.model_inputs.common.utils import read_config
from celllift.model_inputs.common.verify import verify

def main() -> None:
    parser = argparse.ArgumentParser(description='Build model-ready public pathology patch graphs')
    parser.add_argument('--dataset', choices=('sicapv2', 'crc_msi'), required=True)
    parser.add_argument('--stage', choices=('inventory', 'standardize', 'pilot', 'segment', 'graph', 'verify', 'all'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--pilot', action='store_true', help='Run the requested stage on the frozen pilot manifest')
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError('invalid shard id/count')
    cfg = read_config(args.config)
    if cfg.get('dataset') != args.dataset:
        raise RuntimeError(f"config dataset {cfg.get('dataset')} does not match CLI {args.dataset}")
    if args.stage == 'inventory':
        result = build_inventory(cfg, args.dataset)
    elif args.stage == 'standardize':
        result = run_standardize(cfg, args.dataset, args.shard_id, args.num_shards, pilot=args.pilot)
    elif args.stage == 'segment':
        result = run_segment(cfg, args.dataset, args.shard_id, args.num_shards, pilot=args.pilot)
    elif args.stage == 'graph':
        result = run_graph(cfg, args.dataset, args.shard_id, args.num_shards, pilot=args.pilot)
    elif args.stage == 'verify':
        result = verify(cfg, args.dataset, pilot=args.pilot)
    elif args.stage == 'pilot':
        run_standardize(cfg, args.dataset, args.shard_id, args.num_shards, pilot=True)
        run_segment(cfg, args.dataset, args.shard_id, args.num_shards, pilot=True)
        graph_python = str(cfg['runtime']['graph_python'])
        base = [graph_python, str(Path(__file__).resolve()), '--dataset', args.dataset, '--config', str(args.config)]
        subprocess.check_call(base + ['--stage', 'graph', '--shard-id', str(args.shard_id), '--num-shards', str(args.num_shards), '--pilot'])
        if args.num_shards == 1:
            subprocess.check_call(base + ['--stage', 'verify', '--pilot'])
            from celllift.model_inputs.common.utils import read_json
            result = read_json(Path(cfg['paths']['data_root']) / '05_qc/pilot_gate.status.json')
        else:
            result = {'status': 'PILOT_SHARD_COMPLETE'}
    else:
        if args.shard_id == 0:
            build_inventory(cfg, args.dataset)
        run_standardize(cfg, args.dataset, args.shard_id, args.num_shards)
        run_segment(cfg, args.dataset, args.shard_id, args.num_shards)
        graph_python = str(cfg['runtime']['graph_python'])
        base = [graph_python, str(Path(__file__).resolve()), '--dataset', args.dataset, '--config', str(args.config)]
        subprocess.check_call(base + ['--stage', 'graph', '--shard-id', str(args.shard_id), '--num-shards', str(args.num_shards)])
        if args.num_shards == 1:
            subprocess.check_call(base + ['--stage', 'verify'])
            from celllift.model_inputs.common.utils import read_json
            result = read_json(Path(cfg['paths']['data_root']) / '05_qc/final_gate.status.json')
        else:
            result = {'status': 'SHARD_COMPLETE'}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
if __name__ == '__main__':
    main()
