from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import math
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from torch.utils.data import DataLoader
from .compatibility_score_adapter import FROZEN_CHECKPOINT_SHA256, NodeEdgeBudgetBatchSampler, PublicShapeCurriculumGraphDataset, CurriculumCheckpointInfo, load_frozen_compatibility_score_model, sha256_file, validate_empty_inference_batch
GEOMETRY_KEY = ('dataset', 'graph_id', 'anchor_id', 'object_type')

def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _node_graph_ids(batch: Mapping[str, Any], count: int) -> list[str]:
    ptr = batch['graph_ptr']
    if torch.is_tensor(ptr):
        ptr = ptr.detach().cpu().tolist()
    graph_ids = list(batch['graph_ids'])
    if len(ptr) != len(graph_ids) + 1 or int(ptr[-1]) != count:
        raise RuntimeError('graph_ptr and graph_ids do not cover prediction nodes')
    return [str(graph_ids[index]) for index in range(len(graph_ids)) for _ in range(int(ptr[index + 1]) - int(ptr[index]))]

def _q_from_columns(columns: Mapping[str, Sequence[Any]], prefix: str, index: int) -> np.ndarray:
    xx = float(columns[f'{prefix}_Q_xx'][index])
    xy = float(columns[f'{prefix}_Q_xy'][index])
    xz = float(columns[f'{prefix}_Q_xz'][index])
    yy = float(columns[f'{prefix}_Q_yy'][index])
    yz = float(columns[f'{prefix}_Q_yz'][index])
    zz = float(columns[f'{prefix}_Q_zz'][index])
    return np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], np.float64)

def wide_prediction_to_long(columns: Mapping[str, Sequence[Any]], batch: Mapping[str, Any], *, dataset: str, checkpoint: CurriculumCheckpointInfo) -> tuple[list[dict[str, Any]], dict[str, int]]:
    count = len(columns['nucleus_id'])
    graph_ids = _node_graph_ids(batch, count)
    anchor_ids = [int(value) for value in columns['nucleus_id']]
    split = [str(value) for value in columns['split']]
    raw_rays = batch.get('node_features_raw_um')
    border = batch.get('anchor_border_flag')
    if raw_rays is None or tuple(raw_rays.shape) != (count, 36):
        raise RuntimeError('raw observed nucleus rays must have shape [N,36]')
    if border is None or tuple(border.shape) != (count,):
        raise RuntimeError('observed anchor border flags must have shape [N]')
    if torch.is_tensor(raw_rays):
        raw_rays = raw_rays.detach().cpu().numpy()
    else:
        raw_rays = np.asarray(raw_rays)
    if torch.is_tensor(border):
        border = border.detach().cpu().numpy()
    else:
        border = np.asarray(border)
    if not np.all(np.isfinite(raw_rays)) or np.any(raw_rays < 0) or np.any(raw_rays.sum(axis=1) <= 0):
        raise RuntimeError('observed nucleus rays must be finite, non-negative and non-empty')
    if len(set(zip(graph_ids, anchor_ids))) != count:
        raise RuntimeError('duplicate (graph_id, anchor_id) in an inference batch')
    rows: list[dict[str, Any]] = []
    qc = {'anchors': count, 'objects': 0, 'invalid_spd': 0, 'invalid_axes': 0}
    for index in range(count):
        for object_type in ('nucleus', 'cell'):
            q = _q_from_columns(columns, object_type, index)
            axes = np.asarray([columns[f'{object_type}_axis_a_um'][index], columns[f'{object_type}_axis_b_um'][index], columns[f'{object_type}_axis_c_um'][index]], dtype=np.float64)
            center = np.asarray([columns[f'{object_type}_center_x_um'][index], columns[f'{object_type}_center_y_um'][index], columns[f'{object_type}_center_z_um'][index]], dtype=np.float64)
            rotation = np.asarray([columns[f'{object_type}_rotation_{row}{column}'][index] for row in range(3) for column in range(3)], dtype=np.float64).reshape(3, 3)
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(axes)) and np.all(np.isfinite(center)) and np.all(np.isfinite(rotation))):
                raise RuntimeError('compatibility_score emitted NaN/Inf geometry')
            eigenvalues = np.linalg.eigvalsh(q)
            if np.any(eigenvalues <= 0):
                qc['invalid_spd'] += 1
                raise RuntimeError('compatibility_score emitted non-SPD Q')
            if np.any(axes <= 0) or not axes[0] >= axes[1] >= axes[2]:
                qc['invalid_axes'] += 1
                raise RuntimeError('compatibility_score emitted non-positive or unordered axes')
            route_key = f'{object_type}_pose_route_matched'
            route_matched = bool(columns.get(route_key, np.zeros(count, bool))[index])
            if route_matched:
                raise RuntimeError('observation-free public sample entered matched pose route')
            row: dict[str, Any] = {'dataset': dataset, 'graph_id': graph_ids[index], 'anchor_id': anchor_ids[index], 'object_type': object_type, 'split': split[index], 'track_id': str(columns['track_id'][index]), 'component_id': str(columns['component_id'][index]), 'layer_idx': int(columns['layer_idx'][index]), 'section_id': int(columns['section_id'][index]), 'anchor_x_um': float(columns['anchor_x_um'][index]), 'anchor_y_um': float(columns['anchor_y_um'][index]), 'anchor_border_flag': bool(border[index]), 'nucleus_rays_um': raw_rays[index].astype(np.float32).tolist(), 'center_x_um': float(center[0]), 'center_y_um': float(center[1]), 'center_z_um': float(center[2]), 'center_z_coordinate': 'compatibility_score_middle_slab_0_to_5_um', 'q_xx': float(q[0, 0]), 'q_xy': float(q[0, 1]), 'q_xz': float(q[0, 2]), 'q_yy': float(q[1, 1]), 'q_yz': float(q[1, 2]), 'q_zz': float(q[2, 2]), 'axis_a_um': float(axes[0]), 'axis_b_um': float(axes[1]), 'axis_c_um': float(axes[2]), 'volume_um3': float(4.0 / 3.0 * math.pi * np.prod(axes)), 'pose_route_matched': False, 'inference_route': 'unmatched_predicted_pose', 'observation_count': int(columns[f'{object_type}_observation_count'][index]), 'gold_observation_count': int(columns['gold_observation_count'][index]), 'silver_observation_count': int(columns['silver_observation_count'][index]), 'checkpoint_path': checkpoint.path, 'checkpoint_sha256': checkpoint.sha256, 'checkpoint_run_id': checkpoint.run_id, 'checkpoint_epoch': checkpoint.epoch}
            for rotation_row in range(3):
                for rotation_column in range(3):
                    row[f'rotation_{rotation_row}{rotation_column}'] = float(rotation[rotation_row, rotation_column])
            if row['observation_count'] or row['gold_observation_count'] or row['silver_observation_count']:
                raise RuntimeError('public geometry row contains observation evidence')
            rows.append(row)
            qc['objects'] += 1
    if len(rows) != 2 * count:
        raise RuntimeError('dual geometry did not emit two objects per anchor')
    return (rows, qc)

def _route_empty_pose(outputs: Any, batch: Mapping[str, Any], cfg: Mapping[str, Any], losses_module: Any) -> dict[str, Any]:
    count = int(outputs.nucleus_center_um.shape[0])
    loss_cfg = cfg['loss']
    route_kwargs = {'num_nodes': count, 'silver_multiplier': float(loss_cfg.get('silver_weight', 1.0)), 'grid_size': int(loss_cfg.get('slab_centroid_grid_size', 16)), 'refine_iterations': int(loss_cfg.get('slab_centroid_refine_iterations', 6)), 'root_iterations': int(loss_cfg.get('slab_centroid_root_iterations', 24)), 'reachable_epsilon_um': float(loss_cfg.get('slab_centroid_reachable_epsilon_um', 0.001)), 'evidence_phase_z': bool(loss_cfg.get('evidence_phase_z', False)), 'evidence_phase_teacher_only': bool(loss_cfg.get('evidence_phase_teacher_only', False)), 'evidence_phase_mode': str(loss_cfg.get('evidence_phase_mode', 'area_weighted_plane_mean')), 'evidence_phase_pose_iterations': int(loss_cfg.get('evidence_phase_pose_iterations', 1)), 'joint_teacher_center_weight': float(loss_cfg.get('joint_teacher_center_weight', 1.0)), 'pixel_size_um': float(cfg['experiment']['native_mpp']), 'slab_thickness_um': float(cfg['experiment']['section_spacing_um'])}
    nucleus = losses_module.object_pose_routed_geometry(outputs.nucleus_center_um, outputs.nucleus_semi_axes_um, outputs.nucleus_roll, outputs.nucleus_pose_axis, batch['nucleus_pose_evidence'], **route_kwargs)
    cell = losses_module.object_pose_routed_geometry(outputs.cell_center_um, outputs.cell_semi_axes_um, outputs.cell_roll, outputs.cell_pose_axis, batch['cell_pose_evidence'], **route_kwargs)
    if torch.any(nucleus.matched) or torch.any(cell.matched):
        raise RuntimeError('empty public evidence unexpectedly produced a matched route')
    routed = dict(vars(outputs))
    routed.update({'nucleus_q': nucleus.final_q, 'cell_q': cell.final_q, 'nucleus_center_um': nucleus.final_center_um, 'cell_center_um': cell.final_center_um, 'nucleus_pose_route_matched': nucleus.matched, 'cell_pose_route_matched': cell.matched, 'nucleus_teacher_residual_um': nucleus.teacher_residual_um, 'cell_teacher_residual_um': cell.teacher_residual_um})
    return routed

class AtomicParquetShardWriter:

    def __init__(self, output_dir: str | Path, *, rows_per_shard: int=200000, resume: bool=False) -> None:
        if rows_per_shard <= 0:
            raise ValueError('rows_per_shard must be positive')
        if rows_per_shard % 2:
            raise ValueError('rows_per_shard must be even to keep anchor object pairs together')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.output_dir.glob('dual_geometry_*.parquet'))
        if existing and (not resume):
            raise RuntimeError('dual-geometry output already has shards; use a new directory or a PASS manifest')
        self.rows_per_shard = int(rows_per_shard)
        self.buffer: list[dict[str, Any]] = []
        self.paths: list[Path] = existing
        self.existing_rows = 0
        if existing:
            import pyarrow.parquet as pq
            expected_names = [f'dual_geometry_{index:04d}.parquet' for index in range(len(existing))]
            if [path.name for path in existing] != expected_names:
                raise RuntimeError('existing dual-geometry shard numbering is not contiguous')
            counts = [int(pq.ParquetFile(path).metadata.num_rows) for path in existing]
            if any((count <= 0 or count % 2 for count in counts)):
                raise RuntimeError('existing dual-geometry shards contain invalid row counts')
            if any((count != self.rows_per_shard for count in counts[:-1])):
                raise RuntimeError('only the final existing geometry shard may be partial')
            if counts[-1] > self.rows_per_shard:
                raise RuntimeError('existing geometry shard exceeds rows_per_shard')
            self.existing_rows = sum(counts)

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        while len(self.buffer) >= self.rows_per_shard:
            block = self.buffer[:self.rows_per_shard]
            del self.buffer[:self.rows_per_shard]
            self._write(block)

    def _write(self, rows: list[dict[str, Any]]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError('pyarrow is required for geometry output') from exc
        destination = self.output_dir / f'dual_geometry_{len(self.paths):04d}.parquet'
        temporary = destination.with_name(f'.{destination.name}.tmp.{os.getpid()}')
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd', row_group_size=65536)
        os.replace(temporary, destination)
        self.paths.append(destination)

    def close(self) -> list[Path]:
        if self.buffer:
            self._write(self.buffer)
            self.buffer = []
        return list(self.paths)

@torch.no_grad()
def infer_dual_geometry(*, dataset_name: str, graph_index: str | Path, graph_root: str | Path, training_code_root: str | Path, checkpoint: str | Path, output_dir: str | Path, graph_shards: int=64, feature_stats: str | Path | None=None, limit: int | None=None, max_nodes: int=21797, max_edges: int=261808, rows_per_shard: int=200000, workers: int=4, device: str='cuda', expected_checkpoint_sha256: str=FROZEN_CHECKPOINT_SHA256) -> dict[str, Any]:
    destination = Path(output_dir)
    manifest_path = destination / 'dual_geometry_manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            if previous.get('checkpoint_sha256') != expected_checkpoint_sha256:
                raise RuntimeError('existing geometry manifest uses another checkpoint')
            return previous
    dataset = PublicShapeCurriculumGraphDataset(graph_index, graph_root, training_code_root, shards=graph_shards, feature_stats=feature_stats, limit=limit, dataset_name=dataset_name)
    full_rows = list(dataset.rows)
    expected_anchors = sum((int(row['node_count']) for row in full_rows))
    writer = AtomicParquetShardWriter(destination, rows_per_shard=rows_per_shard, resume=True)
    if writer.existing_rows > 2 * expected_anchors:
        raise RuntimeError('existing geometry shards exceed the indexed anchor coverage')
    existing_anchors = writer.existing_rows // 2
    prefix_anchors = 0
    start_graph = 0
    while start_graph < len(full_rows) and prefix_anchors + int(full_rows[start_graph]['node_count']) <= existing_anchors:
        prefix_anchors += int(full_rows[start_graph]['node_count'])
        start_graph += 1
    skip_first_anchors = existing_anchors - prefix_anchors
    dataset.rows = full_rows[start_graph:]
    if not dataset.rows and existing_anchors != expected_anchors:
        raise RuntimeError('resume accounting exhausted the graph index early')
    model, checkpoint_info, modules = load_frozen_compatibility_score_model(training_code_root, checkpoint, expected_sha256=expected_checkpoint_sha256)
    target = torch.device(device)
    if target.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    model.to(target).float().eval()
    sampler = NodeEdgeBudgetBatchSampler(dataset, max_nodes=max_nodes, max_edges=max_edges)
    loader_options: dict[str, Any] = {'batch_sampler': sampler, 'collate_fn': dataset.collate, 'num_workers': int(workers), 'pin_memory': target.type == 'cuda', 'persistent_workers': False}
    if workers:
        loader_options['prefetch_factor'] = 2
    loader = DataLoader(dataset, **loader_options)
    anchor_count = existing_anchors
    object_count = writer.existing_rows
    graph_count = start_graph
    pending_skip = skip_first_anchors
    for raw_batch in loader:
        validate_empty_inference_batch(raw_batch)
        batch = modules['predict'].move_to_device(raw_batch, target)
        outputs = model(batch['node_features'], batch['edge_index'], batch['edge_feat'], batch['xy_um'], batch['nucleus_rays_um'], batch.get('undirected_edge_index'), batch.get('non_overlap_edge_mask'))
        routed = _route_empty_pose(outputs, batch, checkpoint_info.config, modules['losses'])
        columns = modules['predict'].prediction_columns(routed, raw_batch)
        rows, qc = wide_prediction_to_long(columns, raw_batch, dataset=dataset_name, checkpoint=checkpoint_info)
        if pending_skip:
            if pending_skip > qc['anchors']:
                raise RuntimeError('resume prefix exceeds the first inference batch')
            rows = rows[2 * pending_skip:]
            qc['anchors'] -= pending_skip
            qc['objects'] -= 2 * pending_skip
            pending_skip = 0
        writer.append(rows)
        anchor_count += qc['anchors']
        object_count += qc['objects']
        graph_count += int(raw_batch['num_graphs'])
    paths = writer.close()
    dataset.close()
    if pending_skip:
        raise RuntimeError('resume prefix was never consumed')
    if graph_count != len(full_rows) or anchor_count != expected_anchors:
        raise RuntimeError(f'coverage mismatch: graphs={graph_count}/{len(full_rows)}, anchors={anchor_count}/{expected_anchors}')
    if object_count != 2 * anchor_count:
        raise RuntimeError('dual geometry coverage is not two rows per anchor')
    manifest = {'status': 'PASS', 'dataset': dataset_name, 'graph_count': graph_count, 'anchor_count': anchor_count, 'object_count': object_count, 'object_types': ['nucleus', 'cell'], 'key': list(GEOMETRY_KEY), 'inference_precision': 'fp32', 'pose_evidence': 'empty', 'inference_route': 'unmatched_predicted_pose', 'center_z_coordinate': 'compatibility_score_middle_slab_0_to_5_um', 'centered_slab_conversion': 'center_z_centered_um = center_z_um - 2.5', 'checkpoint_path': checkpoint_info.path, 'checkpoint_sha256': checkpoint_info.sha256, 'checkpoint_run_id': checkpoint_info.run_id, 'checkpoint_epoch': checkpoint_info.epoch, 'shards': [{'path': str(path), 'sha256': sha256_file(path), 'size': path.stat().st_size} for path in paths]}
    _atomic_json(manifest_path, manifest)
    return manifest
