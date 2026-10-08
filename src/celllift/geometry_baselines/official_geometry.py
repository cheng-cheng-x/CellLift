from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
import hashlib
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping, Sequence
import numpy as np
from celllift.runtime import torch
from .geometry_training import GraphTokenSample, _load_trainer, preaggregate_meanpool
from .io_utils import atomic_json, atomic_parquet, sha256
from .models import GeometryClassifier
from .protocol import ARMS, ENCODERS, PROTOCOL_ID, SEEDS, Arm, require_test_gate
from .tokens import collate
_STORE_CACHE: dict[tuple[str, int, int], 'OfficialGeometryStore'] = {}
_TRAINER = None

def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256('|'.join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], 'little')

def _fixed(array, width: int) -> np.ndarray:
    combined = array.combine_chunks()
    return np.asarray(combined.values.to_numpy(zero_copy_only=False), np.float32).reshape(len(combined), width)

def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

class OfficialGeometryStore:

    def __init__(self, cfg: Mapping[str, Any], fold: int, seed: int) -> None:
        require_test_gate(cfg['paths']['result_root'])
        self.cfg, self.fold, self.seed = (cfg, int(fold), int(seed))
        self.root = Path(cfg['paths']['data_root']) / '04_residual3d_tokens' / 'official_test' / f'fold_{fold:02d}'
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'PASS' or manifest.get('protocol_id') != PROTOCOL_ID:
            raise RuntimeError(f'official fold token manifest is not compatible: {self.root}')
        import pyarrow.parquet as pq
        metadata_rows = pq.read_table(Path(cfg['paths']['model_input_root']) / '03_graph_cache' / 'graph_index.parquet', columns=['graph_id', 'patient_id', 'split'], filters=[('split', 'in', ['TEST', 'test', 'Test'])], partitioning=None).to_pylist()
        self.metadata = {str(row['graph_id']): {'patient_id': str(row['patient_id']), 'split': 'TEST'} for row in metadata_rows}
        base: dict[str, list[np.ndarray]] = defaultdict(list)
        raw_n: dict[str, list[np.ndarray]] = defaultdict(list)
        raw_c: dict[str, list[np.ndarray]] = defaultdict(list)
        res_n: dict[str, list[np.ndarray]] = defaultdict(list)
        res_c: dict[str, list[np.ndarray]] = defaultdict(list)
        valid: dict[str, list[np.ndarray]] = defaultdict(list)
        anchors: dict[str, list[np.ndarray]] = defaultdict(list)
        for file in sorted(self.root.glob('tokens_*.parquet')):
            table = pq.read_table(file, partitioning=None)
            graph = np.asarray(table['graph_id'].to_pylist(), object)
            values = {'base': _fixed(table['mask2d'], 36), 'raw_n': _fixed(table['raw_nucleus3d'], 5), 'raw_c': _fixed(table['raw_cell3d'], 5), 'res_n': _fixed(table['residual_nucleus3d'], 5), 'res_c': _fixed(table['residual_cell3d'], 5)}
            anchor = np.asarray(table['anchor_id'].to_numpy(zero_copy_only=False))
            is_valid = np.asarray(table['valid_ncr3d'].to_numpy(zero_copy_only=False), bool)
            start = 0
            while start < len(graph):
                stop = start + 1
                while stop < len(graph) and graph[stop] == graph[start]:
                    stop += 1
                key = str(graph[start])
                base[key].append(values['base'][start:stop])
                raw_n[key].append(values['raw_n'][start:stop])
                raw_c[key].append(values['raw_c'][start:stop])
                res_n[key].append(values['res_n'][start:stop])
                res_c[key].append(values['res_c'][start:stop])
                valid[key].append(is_valid[start:stop])
                anchors[key].append(anchor[start:stop])
                start = stop
        self.graph_ids = sorted(base)
        if set(self.graph_ids) != set(self.metadata):
            raise RuntimeError('official token graphs do not match the label-blind TEST graph index')
        self.base = {key: np.concatenate(value) for key, value in base.items()}
        self.raw_n = {key: np.concatenate(value) for key, value in raw_n.items()}
        self.raw_c = {key: np.concatenate(value) for key, value in raw_c.items()}
        self.res_n = {key: np.concatenate(value) for key, value in res_n.items()}
        self.res_c = {key: np.concatenate(value) for key, value in res_c.items()}
        self.valid = {key: np.concatenate(value) for key, value in valid.items()}
        self.anchors = {key: np.concatenate(value) for key, value in anchors.items()}
        if any((len(self.base[key]) == 0 for key in self.graph_ids)):
            raise RuntimeError('empty official geometry graph')
        self._prepare_shuffle()

    def _prepare_shuffle(self) -> None:
        ordered = sorted(self.graph_ids, key=lambda graph: (len(self.base[graph]), graph))
        decile = {graph: min(9, rank * 10 // len(ordered)) for rank, graph in enumerate(ordered)}
        mask_donor: dict[str, str] = {}
        for value in range(10):
            cell = [graph for graph in ordered if decile[graph] == value]
            if len(cell) < 2:
                raise RuntimeError(f'official TEST count decile {value} has fewer than two graphs')
            rng = np.random.default_rng(_stable_seed(PROTOCOL_ID, 'TEST', self.fold, self.seed, 'mask', value))
            shift = int(rng.integers(1, len(cell)))
            mask_donor.update(dict(zip(cell, cell[shift:] + cell[:shift])))
        self.mask_donor = mask_donor
        self.ranges: dict[str, slice] = {}
        cursor = 0
        row_graph, row_decile = ([], [])
        for graph in self.graph_ids:
            count = len(self.base[graph])
            self.ranges[graph] = slice(cursor, cursor + count)
            cursor += count
            row_graph.extend([graph] * count)
            row_decile.extend([decile[graph]] * count)
        self.row_graph = np.asarray(row_graph, object)
        self.row_decile = np.asarray(row_decile, np.int8)
        self.flat_raw_n = np.concatenate([self.raw_n[graph] for graph in self.graph_ids])
        self.flat_raw_c = np.concatenate([self.raw_c[graph] for graph in self.graph_ids])
        self.flat_res_n = np.concatenate([self.res_n[graph] for graph in self.graph_ids])
        self.flat_res_c = np.concatenate([self.res_c[graph] for graph in self.graph_ids])
        self.donor = np.empty(len(self.row_graph), np.int64)
        rng = np.random.default_rng(_stable_seed(PROTOCOL_ID, 'TEST', self.fold, self.seed, '3d'))
        for value in range(10):
            target = np.flatnonzero(self.row_decile == value)
            if len(np.unique(self.row_graph[target])) < 2:
                raise RuntimeError(f'official TEST anchor decile {value} has fewer than two graphs')
            proposal = target[rng.integers(0, len(target), size=len(target))]
            bad = self.row_graph[proposal] == self.row_graph[target]
            for _ in range(64):
                if not bad.any():
                    break
                proposal[bad] = target[rng.integers(0, len(target), size=int(bad.sum()))]
                bad = self.row_graph[proposal] == self.row_graph[target]
            if bad.any():
                for index in np.flatnonzero(bad):
                    choices = target[self.row_graph[target] != self.row_graph[target[index]]]
                    proposal[index] = choices[int(rng.integers(0, len(choices)))]
            self.donor[target] = proposal
        if np.any(self.row_graph[self.donor] == self.row_graph):
            raise RuntimeError('official 3D donor map contains self-graph donors')
        map_root = Path(self.cfg['paths']['data_root']) / '05_shuffle_maps' / 'official_test' / f'fold_{self.fold:02d}' / f'seed_{self.seed}'
        donor_path = map_root / 'three_d_donor_offsets.npy'
        _write_npy(donor_path, self.donor)
        import pyarrow as pa
        mask_path = map_root / 'mask_graph_donors.parquet'
        atomic_parquet(mask_path, [{'graph_id': graph, 'donor_graph_id': self.mask_donor[graph], 'count_decile': decile[graph]} for graph in self.graph_ids])
        atomic_json(map_root / 'manifest.json', {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'phase': 'official_test', 'fold': self.fold, 'seed': self.seed, 'graphs': len(self.graph_ids), 'anchors': len(self.donor), 'mask_map': str(mask_path), 'mask_map_sha256': sha256(mask_path), 'three_d_donor_offsets': str(donor_path), 'three_d_donor_offsets_sha256': sha256(donor_path), 'implicit_target_order': 'lexical graph_id, then cached anchor order', 'constraints': 'TEST-only/count-decile/cross-graph; raw/residual share paired nucleus-cell offsets'})

    def samples(self, arm: Arm, graph_ids: Sequence[str] | None=None) -> list[GraphTokenSample]:
        if arm.arm_id.startswith('R'):
            arm = ARMS['O' + arm.arm_id[1:]]
        output = []
        for graph in self.graph_ids if graph_ids is None else graph_ids:
            count = len(self.base[graph])
            if arm.mask_mode == 'none':
                mask = np.zeros((count, 36), np.float32)
            elif arm.mask_mode == 'real':
                mask = self.base[graph]
            else:
                mask = self.base[self.mask_donor[graph]]
                count = len(mask)
            if arm.three_d_mode == 'none':
                nucleus3d = cell3d = np.zeros((count, 5), np.float32)
            elif arm.three_d_mode.startswith('shuffled_'):
                donor = self.donor[self.ranges[graph]]
                residual = arm.three_d_mode.endswith('residual')
                nucleus3d = (self.flat_res_n if residual else self.flat_raw_n)[donor].copy()
                cell3d = (self.flat_res_c if residual else self.flat_raw_c)[donor].copy()
                invalid = ~self.valid[graph]
                nucleus3d[invalid, -1] = 0.0
                cell3d[invalid, -1] = 0.0
            else:
                residual = arm.three_d_mode == 'residual'
                nucleus3d = self.res_n[graph] if residual else self.raw_n[graph]
                cell3d = self.res_c[graph] if residual else self.raw_c[graph]
            if len(nucleus3d) != count:
                index = np.floor(np.arange(count, dtype=np.float64) * len(nucleus3d) / count).astype(np.int64)
                nucleus3d, cell3d = (nucleus3d[index], cell3d[index])
            nucleus = np.concatenate((mask, nucleus3d), axis=1).astype(np.float32, copy=False)
            cell = np.concatenate((mask, cell3d), axis=1).astype(np.float32, copy=False)
            if nucleus.shape != (count, 41) or cell.shape != nucleus.shape:
                raise RuntimeError('official 41D token schema drift')
            if arm.mask_mode == 'none' and (np.any(nucleus[:, :36]) or np.any(cell[:, :36])):
                raise RuntimeError('official 3D-only view leaked MASK2D')
            if arm.three_d_mode == 'none' and (np.any(nucleus[:, 36:]) or np.any(cell[:, 36:])):
                raise RuntimeError('official MASK-only view leaked 3D')
            output.append(GraphTokenSample(graph, nucleus, cell, self.metadata[graph]))
        return output

def _store(cfg: Mapping[str, Any], fold: int, seed: int) -> OfficialGeometryStore:
    key = (str(cfg['paths']['data_root']), int(fold), int(seed))
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = OfficialGeometryStore(cfg, fold, seed)
    return _STORE_CACHE[key]

def _trainer():
    global _TRAINER
    if _TRAINER is None:
        _TRAINER = _load_trainer()
    return _TRAINER

def _checkpoint(cfg: Mapping[str, Any], encoder: str, arm: Arm, seed: int, fold: int) -> Path:
    return Path(cfg['paths']['result_root']) / 'geometry_only' / 'screen' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{fold}' / 'best.pt'

def _input(batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for name in ('nucleus_tokens', 'cell_tokens', 'nucleus_count', 'cell_count'):
        if name in batch:
            result[name] = torch.as_tensor(batch[name], dtype=torch.float32, device=device)
    for name in ('nucleus_mask', 'cell_mask'):
        result[name] = torch.as_tensor(batch[name], dtype=torch.bool, device=device)
    return result

def _batches(samples: Sequence[GraphTokenSample], budget: int) -> list[list[GraphTokenSample]]:
    ordered = sorted(samples, key=lambda row: (len(row.nucleus_tokens), row.graph_id))
    output: list[list[GraphTokenSample]] = []
    current: list[GraphTokenSample] = []
    maximum = 0
    for sample in ordered:
        size = len(sample.nucleus_tokens)
        next_maximum = max(maximum, size)
        if current and next_maximum * (len(current) + 1) > budget:
            output.append(current)
            current = []
            maximum = 0
        current.append(sample)
        maximum = max(maximum, size)
    if current:
        output.append(current)
    return output

@torch.no_grad()
def predict_official_geometry_job(cfg: Mapping[str, Any], fold: int, seed: int, encoder: str, geometry_id: str, *, device: str='cuda', tile_route: bool=False) -> dict[str, Any]:
    require_test_gate(cfg['paths']['result_root'])
    if encoder not in ENCODERS or seed not in SEEDS:
        raise ValueError('official geometry job is outside the frozen registry')
    arm = ARMS['O' + geometry_id[1:]]
    target = torch.device(device)
    store = _store(cfg, fold, seed)
    samples = store.samples(arm)
    if encoder == 'meanpool':
        samples = preaggregate_meanpool(samples)
    route = 'tile' if tile_route else 'patient'
    output = Path(cfg['paths']['result_root']) / 'official_test' / f'geometry_{route}' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{fold}'
    job_path = output / 'job.json'
    if job_path.is_file():
        previous = json.loads(job_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'PASS':
            expected = {'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': fold, 'seed': seed, 'encoder': encoder, 'arm_id': arm.arm_id, 'route': route, 'test_labels_read': False}
            if any((previous.get(key) != value for key, value in expected.items())):
                raise RuntimeError(f'official geometry completed job provenance mismatch: {job_path}')
            for name in ('checkpoint', 'predictions'):
                path = Path(previous[name])
                if not path.is_file() or sha256(path) != previous.get(f'{name}_sha256'):
                    raise RuntimeError(f'official geometry {name} checksum mismatch: {job_path}')
            return {**previous, 'reused': True}
    budget = int(cfg['geometry_training'].get('token_budget', 32000))
    rows: list[dict[str, Any]] = []
    if tile_route:
        if cfg['dataset'] != 'tcga_crc_msi':
            raise RuntimeError('tile route is CRC-only')
        model = GeometryClassifier('tcga_crc_msi', encoder, dropout=float(cfg['geometry_training']['dropout'])).to(target)
        checkpoint = Path(cfg['paths']['result_root']) / 'tile_fusion' / 'experts' / encoder / arm.arm_id / f'seed_{seed}' / f'fold_{fold}' / 'best.pt'
        model.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=False)['model_state_dict'], strict=True)
        model.eval()
        for group in _batches(samples, budget):
            batch = collate(group)
            scores = torch.sigmoid(model(**_input(batch, target))['logits']).float().cpu().numpy()
            for sample, score in zip(group, scores):
                rows.append({'graph_id': sample.graph_id, 'patient_id': sample.metadata['patient_id'], 'score': float(score)})
    else:
        trainer = _trainer()
        model = trainer._build_models(cfg['dataset'], arm, encoder, float(cfg['geometry_training']['dropout'])).to(target)
        checkpoint = _checkpoint(cfg, encoder, arm, seed, fold)
        model.load_state_dict(torch.load(checkpoint, map_location=target, weights_only=False)['model'], strict=True)
        model.eval()
        if cfg['dataset'] == 'sicapv2':
            for group in _batches(samples, budget):
                values = model(**_input(collate(group), target))['grade_logits'].float().cpu().numpy()
                for sample, logits in zip(group, values):
                    rows.append({'graph_id': sample.graph_id, 'patient_id': sample.metadata['patient_id'], 'logits': logits.tolist()})
        else:
            patients: dict[str, list[GraphTokenSample]] = defaultdict(list)
            for sample in samples:
                patients[str(sample.metadata['patient_id'])].append(sample)
            for patient, patient_samples in sorted(patients.items()):
                embeddings = []
                for group in _batches(patient_samples, budget):
                    embeddings.append(model.fusion(**_input(collate(group), target)))
                tile_embeddings = torch.cat(embeddings).unsqueeze(0)
                mask = torch.ones((1, tile_embeddings.shape[1]), dtype=torch.bool, device=target)
                logit = float(model.mil(tile_embeddings, mask)['logits'].float().cpu()[0])
                rows.append({'patient_id': patient, 'logit': logit, 'probability': float(torch.sigmoid(torch.tensor(logit)))})
    prediction = output / 'test_predictions.parquet'
    atomic_parquet(prediction, rows)
    payload = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'dataset': cfg['dataset'], 'fold': fold, 'seed': seed, 'encoder': encoder, 'arm_id': arm.arm_id, 'route': route, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint), 'predictions': str(prediction), 'predictions_sha256': sha256(prediction), 'rows': len(rows), 'test_labels_read': False}
    atomic_json(job_path, payload)
    return payload
