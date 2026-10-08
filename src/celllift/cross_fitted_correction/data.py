from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from .protocol import assert_validation_only_path

@dataclass(frozen=True)
class Sample:
    graph_id: str
    patient_id: str
    label_id: int
    fold: int
    rgb_logits: np.ndarray
    nucleus_tokens: np.ndarray
    cell_tokens: np.ndarray
    nucleus_summary: np.ndarray | None = None
    cell_summary: np.ndarray | None = None

def canonical_rgb_logit(value: Any) -> np.ndarray:
    logits = np.asarray(value, np.float64)
    if logits.shape == (4,):
        maximum = float(logits.max())
        log_norm = maximum + float(np.log(np.exp(logits - maximum).sum()))
        return (logits - log_norm).astype(np.float32)
    if logits.shape == (2,):
        return np.asarray(logits[1] - logits[0], np.float32)
    raise ValueError(f'unexpected paper RGB logit shape: {logits.shape}')

def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    assert_validation_only_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path, partitioning=None).to_pylist()

def load_oof_rgb(cfg: Mapping[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    root = assert_validation_only_path(cfg['paths']['geometry_baselines_result_root']) / 'paper_baseline'
    result: dict[str, dict[str, Any]] = {}
    for fold in range(int(cfg['split']['validation_folds'])):
        path = root / f'fold_{fold:02d}' / f'seed_{seed}' / 'validation_predictions.parquet'
        for row in read_parquet(path):
            graph_id = str(row['graph_id'])
            if graph_id in result:
                raise RuntimeError(f'duplicate OOF RGB prediction: {graph_id}')
            if int(row['fold']) != fold:
                raise RuntimeError('paper RGB fold metadata mismatch')
            result[graph_id] = {**row, 'fold': fold}
    return result

def partition_oof_ids(cfg: Mapping[str, Any], fold: int, oof: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    graph_manifest = Path(cfg['paths']['geometry_baselines_data_root']) / '00_manifest' / 'downstream_manifest.parquet'
    manifest_rows = read_parquet(graph_manifest)
    geometry_ids = {str(row['graph_id']) for row in manifest_rows if str(row.get('split', row.get('official_split', 'train'))).lower() == 'train'}
    ids = sorted(set(oof) & geometry_ids)
    excluded = sorted(set(oof) - geometry_ids)
    expected_excluded = 0 if cfg['dataset'] == 'sicapv2' else 2
    if len(excluded) != expected_excluded:
        raise RuntimeError(f'unexpected RGB/geometry exclusions: {len(excluded)} != {expected_excluded}')
    return ([key for key in ids if int(oof[key]['fold']) != fold], [key for key in ids if int(oof[key]['fold']) == fold])

def load_fold_store(cfg: Mapping[str, Any], fold: int, seed: int, oof: Mapping[str, Mapping[str, Any]]):
    from celllift.geometry_baselines.geometry_training import GeometryFoldTokenStore, configure
    geometry_baselines_cfg = {'dataset': cfg['dataset'], 'paths': {'model_input_root': cfg['paths']['model_input_root'], 'set_encoding_data_root': cfg['paths']['set_encoding_data_root'], 'mask_conditioning_data_root': cfg['paths']['mask_conditioning_data_root'], 'data_root': cfg['paths']['geometry_baselines_data_root'], 'result_root': cfg['paths']['geometry_baselines_result_root']}, 'probe': {'conditioning_mode': 'mask_rays_only', 'inner_folds': 3}}
    configure(geometry_baselines_cfg)
    graph_manifest = Path(cfg['paths']['geometry_baselines_data_root']) / '00_manifest' / 'downstream_manifest.parquet'
    train_ids, validation_ids = partition_oof_ids(cfg, fold, oof)
    ids = sorted(train_ids + validation_ids)
    partition = {key: 'train' for key in train_ids} | {key: 'validation' for key in validation_ids}
    store = GeometryFoldTokenStore.from_parquet(Path(cfg['paths']['set_encoding_data_root']) / '02_nucleus_tokens', Path(cfg['paths']['set_encoding_data_root']) / '03_cell_tokens', Path(cfg['paths']['set_encoding_data_root']) / '04_ncr_features', graph_manifest, training_graph_ids=train_ids, seed=seed, included_graph_ids=ids, partition_by_graph=partition)
    if set(store.graph_ids) != set(ids):
        raise RuntimeError('OOF RGB and geometry graph coverage differ')
    return (store, train_ids, validation_ids)

def multistat(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 41 or len(values) == 0:
        raise ValueError('MultiStat expects a non-empty [N,41] set')
    summary = np.concatenate((values.mean(0, dtype=np.float64), values.std(0, dtype=np.float64), np.quantile(values, 0.1, axis=0), np.quantile(values, 0.5, axis=0), np.quantile(values, 0.9, axis=0), np.asarray([np.log1p(len(values))]))).astype(np.float32)
    if summary.shape != (206,) or not np.isfinite(summary).all():
        raise RuntimeError('invalid MultiStat summary')
    return summary

def build_samples(store: Any, graph_ids: Sequence[str], arm: Any, oof: Mapping[str, Mapping[str, Any]], *, pooler: str) -> list[Sample]:
    rows = store.samples(arm, graph_ids)
    output = []
    for row in rows:
        rgb = oof[row.graph_id]
        logits = canonical_rgb_logit(rgb['logits'])
        output.append(Sample(graph_id=row.graph_id, patient_id=str(rgb['patient_id']), label_id=int(rgb['label_id']), fold=int(rgb['fold']), rgb_logits=logits, nucleus_tokens=row.nucleus_tokens, cell_tokens=row.cell_tokens, nucleus_summary=multistat(row.nucleus_tokens) if pooler == 'multistat' else None, cell_summary=multistat(row.cell_tokens) if pooler == 'multistat' else None))
    return output

def build_multistat_samples_cached(store: Any, graph_ids: Sequence[str], arm: Any, oof: Mapping[str, Mapping[str, Any]], *, cache_path: Path) -> list[Sample]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = list(graph_ids)
    if cache_path.is_file():
        saved = np.load(cache_path, allow_pickle=False)
        cached_ids = saved['graph_ids'].astype(str).tolist()
        if cached_ids != expected_ids:
            raise RuntimeError(f'MultiStat cache graph order mismatch: {cache_path}')
        nucleus, cell = (saved['nucleus'], saved['cell'])
    else:
        rows = store.samples(arm, graph_ids)
        nucleus = np.stack([multistat(row.nucleus_tokens) for row in rows])
        cell = np.stack([multistat(row.cell_tokens) for row in rows])
        temporary = cache_path.with_name(f'.{cache_path.name}.tmp.{os.getpid()}')
        with temporary.open('wb') as stream:
            np.savez(stream, graph_ids=np.asarray(expected_ids), nucleus=nucleus, cell=cell)
        os.replace(temporary, cache_path)
    if nucleus.shape != (len(expected_ids), 206) or cell.shape != nucleus.shape:
        raise RuntimeError('MultiStat cache shape mismatch')
    dummy = np.zeros((1, 41), np.float32)
    output = []
    for index, graph_id in enumerate(expected_ids):
        rgb = oof[graph_id]
        logits = canonical_rgb_logit(rgb['logits'])
        output.append(Sample(graph_id=graph_id, patient_id=str(rgb['patient_id']), label_id=int(rgb['label_id']), fold=int(rgb['fold']), rgb_logits=logits, nucleus_tokens=dummy, cell_tokens=dummy, nucleus_summary=nucleus[index], cell_summary=cell[index]))
    return output
