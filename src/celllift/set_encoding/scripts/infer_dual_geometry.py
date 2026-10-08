from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import sys
from celllift.runtime import ResourcePath as Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.common.dual_geometry import infer_dual_geometry
from celllift.set_encoding.common.compatibility_score_adapter import FROZEN_CHECKPOINT_SHA256

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--graph-index', type=Path, required=True)
    parser.add_argument('--graph-root', type=Path, required=True)
    parser.add_argument('--training-code-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--graph-shards', type=int, default=64)
    parser.add_argument('--feature-stats', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-nodes', type=int, default=21797)
    parser.add_argument('--max-edges', type=int, default=261808)
    parser.add_argument('--rows-per-shard', type=int, default=200000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-sha256', default=FROZEN_CHECKPOINT_SHA256)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    result = infer_dual_geometry(dataset_name=args.dataset, graph_index=args.graph_index, graph_root=args.graph_root, training_code_root=args.training_code_root, checkpoint=args.checkpoint, output_dir=args.output_dir, graph_shards=args.graph_shards, feature_stats=args.feature_stats, limit=args.limit, max_nodes=args.max_nodes, max_edges=args.max_edges, rows_per_shard=args.rows_per_shard, workers=args.workers, device=args.device, expected_checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
