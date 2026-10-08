from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import hashlib
from celllift.runtime import json
import os
import sys
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
from celllift.breast_roi_baseline.cache import atomic_json, build_anchor_cache, sha256_file
from celllift.breast_roi_baseline.data import collect_wsi_mpp, load_bracs_manifest, preflight_bracs_data
from celllift.breast_roi_baseline.evaluation import evaluate_fusion
from celllift.breast_roi_baseline.prepare import prepare_bracs_tiles
from celllift.breast_roi_baseline.probe import fit_nested_probe
from celllift.breast_roi_baseline.splits import attach_development_folds, attach_final_folds
from celllift.breast_roi_baseline.training import BRACSFeatureStore, train_expert

def load_config(path: str | Path) -> dict[str, Any]:
    from celllift.runtime import yaml
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if value.get('dataset') != 'bracs':
        raise RuntimeError('BRACS config dataset mismatch')
    return value

def _roots(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    return (Path(cfg['paths']['data_root']), Path(cfg['paths']['result_root']))

def run_preflight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    report = preflight_bracs_data(cfg['paths']['wsi_manifest'], cfg['paths']['roi_manifest'], output=None)
    checkpoint = Path(cfg['paths']['checkpoint'])
    feature_stats = Path(cfg['paths']['training_feature_stats'])
    upstream_code = Path(cfg['paths']['upstream_code_root'])
    for path, name in ((checkpoint, 'checkpoint'), (feature_stats, 'training feature stats'), (upstream_code, 'upstream code root')):
        if not path.exists():
            raise FileNotFoundError(f'{name} is missing: {path}')
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != cfg['upstream']['checkpoint_sha256']:
        raise RuntimeError('frozen compatibility_score checkpoint SHA256 mismatch')
    report['upstream'] = {'run_id': cfg['upstream']['run_id'], 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'training_feature_stats': str(feature_stats), 'training_feature_stats_sha256': sha256_file(feature_stats), 'upstream_code_root': str(upstream_code)}
    report['resolution_contract'] = {'target_mpp': float(cfg['resolution']['target_mpp']), 'tile_size_px': int(cfg['resolution']['tile_size_px']), 'physical_field_um': float(cfg['resolution']['physical_field_um'])}
    atomic_json(data_root / '00_manifest' / 'preflight.json', report)
    return report

def _folds_by_roi(manifest) -> dict[str, dict[str, int | None]]:
    labels7 = list(('N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC'))
    rows = [{'roi_id': roi.roi_id, 'wsi_id': roi.wsi_id, 'split_new': roi.split, 'label_7': labels7.index(roi.label_7)} for roi in manifest.rois]
    development = attach_development_folds(rows)
    final = {row['roi_id']: row for row in attach_final_folds(development)}
    return {row['roi_id']: {'validation_fold': row['validation_fold'], 'final_validation_fold': final[row['roi_id']]['final_validation_fold']} for row in development}

def _shard_hex(value: str) -> str:
    return hashlib.blake2b(value.encode(), digest_size=1).hexdigest()

def _stable_shard(value: str, count: int) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'big') % count

def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)

def build_model_input_manifest(cfg: Mapping[str, Any], tile_manifest: str | Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    data_root, _ = _roots(cfg)
    tiles = pq.read_table(tile_manifest, partitioning=None).to_pylist()
    rows = []
    for tile in tiles:
        tile_id, graph_id = (str(tile['tile_id']), str(tile['graph_id']))
        subdir = _shard_hex(tile_id)
        segment = data_root / '02_masks_graphs' / 'cellpose'
        rows.append({'dataset_id': 'breast_roi_baseline', 'patch_id': tile_id, 'graph_id': graph_id, 'roi_id': str(tile['roi_id']), 'wsi_id': str(tile['wsi_id']), 'source_path': str(tile['tile_path']), 'rgb_path': str(tile['tile_path']), 'source_sha256': str(tile['tile_sha256']), 'source_mpp': float(tile['native_mpp']), 'target_mpp': 0.46, 'actual_target_mpp': 0.46, 'mpp_provenance': str(tile['mpp_source']), 'patient_id': str(tile['wsi_id']), 'slide_id': str(tile['roi_id']), 'official_split': str(tile['split_new']), 'validation_fold': tile.get('validation_fold'), 'final_validation_fold': tile.get('final_validation_fold'), 'label_id': int(tile['label_7']), 'label_name': str(tile['label_7_name']), 'label_3': int(tile['label_3']), 'label_3_name': str(tile['label_3_name']), 'label_scope': 'roi_ground_truth', 'output_width_px': 1024, 'output_height_px': 1024, 'physical_width_um': 471.04, 'physical_height_um': 471.04, 'source_to_target_affine': json.dumps([1, 0, 0, 0, 1, 0, 0, 0, 1]), 'nucleus_mask_path': str(segment / 'nucleus_masks' / subdir / f'{tile_id}.npy'), 'cell_mask_path': str(segment / 'cell_masks' / subdir / f'{tile_id}.npy'), 'nucleus_instances_path': str(segment / 'nucleus_instances' / subdir / f'{tile_id}.json.gz'), 'cell_instances_path': str(segment / 'cell_instances' / subdir / f'{tile_id}.json.gz'), 'pairs_path': str(segment / 'pairs' / subdir / f'{tile_id}.json.gz'), 'graph_shard': _stable_shard(graph_id, 64), 'node_count': None, 'edge_count': None, 'processing_status': 'prepared', 'exclusion_reason': ''})
    patch_path = data_root / '00_manifest' / 'patch_manifest.parquet'
    _atomic_parquet(patch_path, rows)
    _atomic_parquet(data_root / '04_labels_splits' / 'labels_splits.parquet', rows)
    payload = {'status': 'PASS', 'tiles': len(rows), 'path': str(patch_path), 'sha256': sha256_file(patch_path)}
    atomic_json(data_root / '00_manifest' / 'model_input_manifest.json', payload)
    return payload

def run_prepare_tiles(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    manifest = load_bracs_manifest(cfg['paths']['wsi_manifest'], cfg['paths']['roi_manifest'])
    referenced = {roi.wsi_id for roi in manifest.rois}
    mpp = collect_wsi_mpp([wsi for wsi in manifest.wsis if wsi.wsi_id in referenced])
    tile_manifest = data_root / '00_manifest' / 'tile_manifest.parquet'
    result = prepare_bracs_tiles(manifest, mpp, data_root / '01_tiles_046mpp', tile_manifest, target_mpp=float(cfg['resolution']['target_mpp']), tile_size=int(cfg['resolution']['tile_size_px']), summary_output=data_root / '00_manifest' / 'tiles.json', folds_by_roi=_folds_by_roi(manifest), workers=int(cfg['runtime']['tile_workers']))
    result['model_input'] = build_model_input_manifest(cfg, tile_manifest)
    return result

def _model_input_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_root, result_root = _roots(cfg)
    return {'dataset': 'bracs', 'input': {'mode': 'nucleus_only'}, 'paths': {'data_root': str(data_root), 'result_root': str(result_root)}, 'runtime': {'graph_workers': int(cfg['runtime']['graph_workers']), 'postprocess_workers': int(cfg['runtime'].get('segment_postprocess_workers', 4)), 'max_pending_writes': int(cfg['runtime'].get('segment_max_pending_writes', 16))}, 'cellpose': {'require_cuda': True, 'vectorized_get_masks': True, 'sorted_percentile': True, 'version': '3.1.1.3', 'nucleus_model': 'nuclei', 'nucleus_diameter_px': 17, 'flow_threshold': 0.8, 'cellprob_threshold': -1.0, 'batch_size': int(cfg['runtime'].get('segment_batch_size', 8)), 'image_chunk_size': int(cfg['runtime'].get('segment_image_chunk_size', 8))}, 'graph': {'ray_count': 36, 'k_neighbors': 12, 'radius_um': 60.0, 'edge_normalizer_um': 471.04, 'cache_shards': 64, 'lmdb_map_size_bytes': 4 << 30, 'training_feature_stats': str(cfg['paths']['training_feature_stats'])}}

def run_build_masks(cfg: Mapping[str, Any], shard_id: int, num_shards: int) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[4]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from celllift.model_inputs.common.segment import run_segment
    return run_segment(_model_input_cfg(cfg), 'bracs', shard_id, num_shards, pilot=False)

def run_build_graphs(cfg: Mapping[str, Any], shard_id: int, num_shards: int) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[4]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from celllift.model_inputs.common.graph import run_graph
    return run_graph(_model_input_cfg(cfg), 'bracs', shard_id, num_shards, pilot=False)

def finalize_graphs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    data_root, _ = _roots(cfg)
    graph_root = data_root / '03_graph_cache'
    patch_rows = pq.read_table(data_root / '00_manifest' / 'patch_manifest.parquet', partitioning=None).to_pylist()
    segment_paths = sorted((data_root / '00_manifest').glob('segment_full_shard_*.parquet'))
    graph_paths = sorted(graph_root.glob('graph_index_full_workshard_*.parquet'))
    exclusion_paths = sorted(graph_root.glob('excluded_full_workshard_*.parquet'))
    if not segment_paths or not graph_paths:
        raise RuntimeError('full segment/graph shards are incomplete')
    segment = pa.concat_tables([pq.read_table(path, partitioning=None) for path in segment_paths])
    if len(segment) != len(patch_rows) or any((value != 'complete' for value in segment['status'].to_pylist())):
        raise RuntimeError('segmentation coverage/status failed')
    graphs = pa.concat_tables([pq.read_table(path, partitioning=None) for path in graph_paths]).to_pylist()
    excluded = pa.concat_tables([pq.read_table(path, partitioning=None) for path in exclusion_paths]).to_pylist()
    if len(graphs) + len(excluded) != len(patch_rows):
        raise RuntimeError('graph plus exclusion coverage mismatch')
    graph_path = graph_root / 'graph_index.parquet'
    exclusion_path = graph_root / 'excluded.parquet'
    _atomic_parquet(graph_path, sorted(graphs, key=lambda row: int(row['layer_idx'])))
    _atomic_parquet(exclusion_path, excluded)
    payload = {'status': 'PASS', 'tiles': len(patch_rows), 'graphs': len(graphs), 'excluded': len(excluded), 'zero_nuclei': sum(('zero_nuclei' in str(row.get('exclusion_reason', '')) for row in excluded)), 'graph_index': str(graph_path), 'graph_index_sha256': sha256_file(graph_path), 'exclusions': str(exclusion_path), 'exclusions_sha256': sha256_file(exclusion_path)}
    atomic_json(data_root / '08_qc' / 'graph_gate.json', payload)
    return payload

def run_infer_geometry(cfg: Mapping[str, Any], device: str) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[4]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from celllift.set_encoding.common.dual_geometry import infer_dual_geometry
    data_root, _ = _roots(cfg)
    return infer_dual_geometry(dataset_name='bracs', graph_index=data_root / '03_graph_cache' / 'graph_index.parquet', graph_root=data_root / '03_graph_cache', training_code_root=cfg['paths']['upstream_code_root'], checkpoint=cfg['paths']['checkpoint'], output_dir=data_root / '03_dual_geometry', graph_shards=int(cfg['runtime']['graph_shards']), feature_stats=data_root / '03_graph_cache' / 'inference_feature_stats.json', rows_per_shard=int(cfg['runtime']['rows_per_geometry_shard']), device=device, expected_checkpoint_sha256=cfg['upstream']['checkpoint_sha256'])

def run_build_direct3d(cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    return build_anchor_cache(geometry_manifest=data_root / '03_dual_geometry' / 'dual_geometry_manifest.json', tile_manifest=data_root / '00_manifest' / 'tile_manifest.parquet', output_dir=data_root / '04_direct3d')

def run_probe(cfg: Mapping[str, Any], phase: str, fold: int, device: str) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    return fit_nested_probe(anchor_manifest=data_root / '04_direct3d' / 'anchor_cache_manifest.json', phase=phase, outer_fold=fold, output_dir=data_root / '05_probe_residuals' / phase / f'fold_{fold:02d}', device=device, train_batch_size=int(cfg['probe'].get('train_batch_size', 65536)), prediction_batch_size=int(cfg['probe'].get('prediction_batch_size', 131072)))

def run_build_residual3d(cfg: Mapping[str, Any], phase: str) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    manifests = []
    for fold in range(int(cfg['split']['folds'])):
        path = data_root / '05_probe_residuals' / phase / f'fold_{fold:02d}' / 'manifest.json'
        if not path.is_file():
            raise RuntimeError(f'missing probe manifest: {path}')
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('status') != 'PASS' or not Path(payload['output']).is_file():
            raise RuntimeError(f'probe fold is not complete: {path}')
        if sha256_file(payload['output']) != payload['output_sha256']:
            raise RuntimeError(f'probe output checksum mismatch: {path}')
        manifests.append(payload)
    output = {'status': 'PASS', 'phase': phase, 'folds': len(manifests), 'manifests': manifests}
    atomic_json(data_root / '05_probe_residuals' / phase / 'residual_manifest.json', output)
    return output

def run_build_shuffles(cfg: Mapping[str, Any], phase: str, fold: int, seed: int) -> dict[str, Any]:
    data_root, _ = _roots(cfg)
    path = data_root / '07_shuffle_maps' / phase / f'fold_{fold:02d}' / f'seed_{seed}' / 'mapping.parquet'
    store = BRACSFeatureStore(anchor_manifest=data_root / '04_direct3d' / 'anchor_cache_manifest.json', tile_manifest=data_root / '00_manifest' / 'tile_manifest.parquet', expert='E3_MASK_SHUF_DIRECT3D', phase=phase, fold=fold, seed=seed, shuffle_map_path=path)
    output = {'status': 'PASS', 'phase': phase, 'fold': fold, 'seed': seed, 'path': str(path), 'sha256': sha256_file(path), 'target_anchors': len(store.donor)}
    atomic_json(path.with_suffix('.json'), output)
    return output

def run_train_expert(cfg: Mapping[str, Any], *, phase: str, fold: int, seed: int, task: str, encoder: str, expert: str, device: str, tile_budget: int | None=None, object_budget: int | None=None) -> dict[str, Any]:
    data_root, result_root = _roots(cfg)
    residual = data_root / '05_probe_residuals' / phase / f'fold_{fold:02d}' / 'probe_predictions.parquet'
    if 'RESIDUAL' not in expert:
        residual = None
    shuffle_map = data_root / '07_shuffle_maps' / phase / f'fold_{fold:02d}' / f'seed_{seed}' / 'mapping.parquet'
    if 'SHUF' not in expert:
        shuffle_map = None
    if phase == 'final_test':
        freeze = result_root / 'protocol_freeze.json'
        if not freeze.is_file() or json.loads(freeze.read_text(encoding='utf-8')).get('status') != 'FROZEN':
            raise RuntimeError('official TEST is locked until protocol_freeze.json has status FROZEN')
    return train_expert(anchor_manifest=data_root / '04_direct3d' / 'anchor_cache_manifest.json', tile_manifest=data_root / '00_manifest' / 'tile_manifest.parquet', residual_path=residual, shuffle_map_path=shuffle_map, expert=expert, task=task, encoder=encoder, phase=phase, fold=fold, seed=seed, device=device, tile_budget=int(tile_budget or cfg['training']['tile_budget']), object_budget=int(object_budget or cfg['training']['object_budget']), max_epochs=int(cfg['training']['max_epochs']), patience=int(cfg['training']['patience']), output_dir=result_root / 'experts' / phase / task / encoder / expert / f'fold_{fold:02d}' / f'seed_{seed}')

def run_evaluate(cfg: Mapping[str, Any], *, phase: str, task: str, encoder: str) -> dict[str, Any]:
    _, result_root = _roots(cfg)
    paths = sorted((result_root / 'experts' / phase / task / encoder).glob('E*/fold_*/seed_*/predictions.parquet'))
    expected = 6 * 5 * 5
    if len(paths) != expected:
        raise RuntimeError(f'expected {expected} expert prediction files, found {len(paths)}')
    return evaluate_fusion(prediction_paths=paths, output_dir=result_root / 'metrics' / phase / task / encoder, task=task, encoder=encoder, phase=phase, bootstrap_replicates=int(cfg['runtime']['bootstrap_replicates']))

def run_report(cfg: Mapping[str, Any], *, phase: str, task: str, encoder: str) -> dict[str, Any]:
    _, result_root = _roots(cfg)
    source = result_root / 'metrics' / phase / task / encoder / 'manifest.json'
    if not source.is_file():
        raise RuntimeError(f'evaluation manifest is missing: {source}')
    payload = json.loads(source.read_text(encoding='utf-8'))
    lines = [f'# BRACS {phase} {task} {encoder}', '', f"Status: `{payload.get('status')}`", '', '## Expert-only', '', '| Expert | macro-F1 | balanced accuracy | accuracy | macro AUROC |', '|---|---:|---:|---:|---:|']
    for expert, values in sorted(payload['expert_metrics'].items()):
        lines.append(f"| {expert} | {values['macro_f1']:.6f} | {values['balanced_accuracy']:.6f} | {values['accuracy']:.6f} | {values['macro_ovr_auroc']:.6f} |")
    for protocol, values in payload['fusion'].items():
        lines.extend(['', f'## {protocol}', '', '| Arm | macro-F1 | balanced accuracy | accuracy | macro AUROC |', '|---|---:|---:|---:|---:|'])
        for arm, metric in sorted(values['arms'].items()):
            lines.append(f"| {arm} | {metric['macro_f1']:.6f} | {metric['balanced_accuracy']:.6f} | {metric['accuracy']:.6f} | {metric['macro_ovr_auroc']:.6f} |")
    report = result_root / 'metrics' / phase / task / encoder / 'report.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'status': 'PASS', 'report': str(report), 'sha256': sha256_file(report)}
