from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any
from celllift.runtime import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celllift.set_encoding.common.dual_geometry import _route_empty_pose, wide_prediction_to_long
from celllift.set_encoding.common.compatibility_score_adapter import FROZEN_CHECKPOINT_SHA256, NodeEdgeBudgetBatchSampler, PublicShapeCurriculumGraphDataset, load_frozen_compatibility_score_model, validate_empty_inference_batch

def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

@torch.no_grad()
def preflight_compatibility_score_public(*, dataset_name: str, graph_index: str | Path, graph_root: str | Path, training_code_root: str | Path, checkpoint: str | Path, graph_shards: int=64, feature_stats: str | Path | None=None, smoke_count: int=64, max_nodes: int=21797, max_edges: int=261808, device: str='cpu', expected_checkpoint_sha256: str=FROZEN_CHECKPOINT_SHA256, output: str | Path | None=None) -> dict[str, Any]:
    dataset = PublicShapeCurriculumGraphDataset(graph_index, graph_root, training_code_root, shards=graph_shards, feature_stats=feature_stats, limit=smoke_count, dataset_name=dataset_name)
    model, checkpoint_info, modules = load_frozen_compatibility_score_model(training_code_root, checkpoint, expected_sha256=expected_checkpoint_sha256)
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    model.to(target).float().eval()
    sampler = NodeEdgeBudgetBatchSampler(dataset, max_nodes=max_nodes, max_edges=max_edges)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate, num_workers=0)
    graphs = anchors = objects = batches = 0
    for raw_batch in loader:
        validation = validate_empty_inference_batch(raw_batch)
        batch = modules['predict'].move_to_device(raw_batch, target)
        outputs = model(batch['node_features'], batch['edge_index'], batch['edge_feat'], batch['xy_um'], batch['nucleus_rays_um'], batch.get('undirected_edge_index'), batch.get('non_overlap_edge_mask'))
        routed = _route_empty_pose(outputs, batch, checkpoint_info.config, modules['losses'])
        columns = modules['predict'].prediction_columns(routed, raw_batch)
        rows, qc = wide_prediction_to_long(columns, raw_batch, dataset=dataset_name, checkpoint=checkpoint_info)
        if len(rows) != 2 * validation['nodes']:
            raise RuntimeError('preflight serialization lost an object row')
        graphs += validation['graphs']
        anchors += qc['anchors']
        objects += qc['objects']
        batches += 1
    dataset.close()
    if graphs != len(dataset) or objects != 2 * anchors:
        raise RuntimeError('preflight coverage mismatch')
    payload = {'status': 'PASS', 'dataset': dataset_name, 'smoke_graphs': graphs, 'smoke_anchors': anchors, 'smoke_objects': objects, 'batches': batches, 'sample_schema': 'compatibility_score GraphInferenceSample with empty nucleus/cell pose evidence', 'inference_route': 'unmatched_predicted_pose', 'inference_precision': 'fp32', 'checkpoint_path': checkpoint_info.path, 'checkpoint_sha256': checkpoint_info.sha256, 'checkpoint_run_id': checkpoint_info.run_id, 'checkpoint_epoch': checkpoint_info.epoch, 'checkpoint_model_tensors': checkpoint_info.model_tensor_count, 'device': str(target)}
    if output is not None:
        _atomic_json(Path(output), payload)
    return payload

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--graph-index', type=Path, required=True)
    parser.add_argument('--graph-root', type=Path, required=True)
    parser.add_argument('--training-code-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--graph-shards', type=int, default=64)
    parser.add_argument('--feature-stats', type=Path)
    parser.add_argument('--smoke-count', type=int, default=64)
    parser.add_argument('--max-nodes', type=int, default=21797)
    parser.add_argument('--max-edges', type=int, default=261808)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint-sha256', default=FROZEN_CHECKPOINT_SHA256)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    result = preflight_compatibility_score_public(dataset_name=args.dataset, graph_index=args.graph_index, graph_root=args.graph_root, training_code_root=args.training_code_root, checkpoint=args.checkpoint, graph_shards=args.graph_shards, feature_stats=args.feature_stats, smoke_count=args.smoke_count, max_nodes=args.max_nodes, max_edges=args.max_edges, device=args.device, expected_checkpoint_sha256=args.checkpoint_sha256, output=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
