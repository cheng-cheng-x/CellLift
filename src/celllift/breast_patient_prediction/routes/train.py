from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import os
from collections import OrderedDict
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.runtime import torch
from torch import nn
from ..io_utils import atomic_json, atomic_npz, atomic_parquet, read_json
from ..protocol import SEED
from .arms import ExpertMIL, FusionMIL, GeometryMIL, ImageMIL, PrototypeMIL, build_model, collect_fit_nucleus_dino_from_store, isolate_tile
from .bags import ResidualCache, SlideCache, build_bags, gpu_tile_chunk, load_tile_geometry, load_tile_scene, split_bags
from .config import ACCUM, DINO_DIM, GEOM_MODE, IMAGE_ARMS, RECIPES, RESIDUAL_ARMS, SCENE_ARMS, TASK_SPEC, arm_dir, probe_dir, protocol_meta, result_root
from .evaluate import better, primary_from_probs, softmax
from .scale import NodePooledScale, fit_node_pooled_scale
from .scene_scale import SceneStats, apply_scene
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

def _seed_all(seed: int=SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _device(name: str) -> torch.device:
    if name.startswith('cuda') and torch.cuda.is_available():
        return torch.device(name)
    return torch.device('cpu')

def _optimizer(model: nn.Module, recipe: dict[str, Any]):
    if recipe['kind'] == 'H':
        geom = list(model.geometry.parameters())
        geom_ids = {id(item) for item in geom}
        other = [item for item in model.parameters() if id(item) not in geom_ids]
        return torch.optim.AdamW([{'params': other, 'lr': recipe['image_lr']}, {'params': geom, 'lr': recipe['geom_lr']}], weight_decay=recipe['wd'])
    return torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=recipe['wd'])
SCENE_E_KEYS = ('node2d', 'node3d', 'edge2d', 'edge3d', 'edge_index', 'include', 'valid3d')
SCENE_P_KEYS = SCENE_E_KEYS + ('dino',)
SCENE_RAM_PATIENTS = 8

def _scene_keys(need_dino: bool) -> tuple[str, ...]:
    return SCENE_P_KEYS if need_dino else SCENE_E_KEYS

def _compact_scene(row: dict[str, Any], *, need_dino: bool, transform=None) -> dict[str, Any]:
    if transform is not None:
        row = transform(row)
    node2d = np.asarray(row.get('node2d', np.zeros((0, 38), np.float32)), np.float32)
    n = 0 if row.get('empty') else int(len(node2d))
    if n == 0:
        payload = {'node2d': np.zeros((0, 38), np.float32), 'node3d': np.zeros((0, 12), np.float32), 'edge2d': np.zeros((0, 4), np.float32), 'edge3d': np.zeros((0, 8), np.float32), 'edge_index': np.zeros((2, 0), np.int64), 'include': np.zeros(0, bool), 'valid3d': np.zeros(0, bool)}
        if need_dino:
            payload['dino'] = np.zeros((0, DINO_DIM), np.float32)
        return payload
    payload = {'node2d': np.array(row['node2d'], np.float32, copy=True), 'node3d': np.array(row['node3d'], np.float32, copy=True), 'edge2d': np.array(row.get('edge2d', np.zeros((0, 4), np.float32)), np.float32, copy=True), 'edge3d': np.array(row.get('edge3d', np.zeros((0, 8), np.float32)), np.float32, copy=True), 'edge_index': np.array(row.get('edge_index', np.zeros((2, 0), np.int64)), np.int64, copy=True), 'include': np.array(row['include'], bool, copy=True), 'valid3d': np.array(row.get('valid3d', np.zeros(n, bool)), bool, copy=True)}
    if need_dino:
        payload['dino'] = np.array(row.get('dino', np.zeros((n, DINO_DIM), np.float32)), np.float32, copy=True)
    return payload

def _pack_scenes(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    packed: dict[str, np.ndarray] = {'n_tiles': np.asarray(len(rows), np.int64)}
    for key in keys:
        pieces = [np.asarray(row[key]) for row in rows]
        axis = 1 if key == 'edge_index' else 0
        counts = np.asarray([int(item.shape[axis]) if item.ndim > axis else 0 for item in pieces], np.int64)
        ptr = np.zeros(len(counts) + 1, np.int64)
        ptr[1:] = np.cumsum(counts)
        if key == 'edge_index':
            blob = np.zeros((2, int(ptr[-1])), np.int64)
            for item, begin, end in zip(pieces, ptr[:-1], ptr[1:]):
                if end > begin:
                    blob[:, begin:end] = item
        else:
            tail = pieces[0].shape[1:] if pieces and pieces[0].ndim >= 1 else ()
            dtype = pieces[0].dtype if pieces else np.float32
            blob = np.zeros((int(ptr[-1]), *tail), dtype=dtype)
            cursor = 0
            for item in pieces:
                n = len(item)
                if n:
                    blob[cursor:cursor + n] = item
                    cursor += n
        packed[key] = blob
        packed[f'{key}_ptr'] = ptr
    return packed

def _unpack_scenes(packed: dict[str, np.ndarray], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    n = int(packed['n_tiles'])
    rows = []
    for index in range(n):
        row = {}
        for key in keys:
            ptr = packed[f'{key}_ptr']
            begin, end = (int(ptr[index]), int(ptr[index + 1]))
            row[key] = packed[key][:, begin:end] if key == 'edge_index' else packed[key][begin:end]
        rows.append(row)
    return rows

class FeatureStore:

    def __init__(self, base: Path, arm: str, task: str):
        self.geom = SlideCache(base, 'geometry', max_slides=2048) if arm in GEOM_MODE or arm in SCENE_ARMS else None
        self.scene = SlideCache(base, 'scene', max_slides=256) if arm in SCENE_ARMS else None
        self.residual = ResidualCache(probe_dir(task), max_slides=2048) if arm in RESIDUAL_ARMS else None
        self.arm = arm
        self.task = task
        self.mode = GEOM_MODE.get(arm)
        self.need_dino = arm in {'P2', 'P3'}
        self._scene_key_names = _scene_keys(self.need_dino)
        self.scale: NodePooledScale | None = None
        self.scene_transform = None
        self._scene_disk = result_root() / 'cache' / ('scenes_p_conditional_geometry' if self.need_dino else 'scenes_e_conditional_geometry') / arm
        self._keep_all_scenes = not self.need_dino
        self._tokens: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
        self._scenes: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def _compute_tokens(self, bag: dict[str, Any], node_cap: int | None) -> list[dict[str, Any]]:
        rows = []
        for tile in bag['tiles']:
            geom = load_tile_geometry(self.geom, tile)
            residual = None
            if self.residual is not None:
                residual = self.residual.get(tile['slide_id'], tile['graph_id'], geom.get('nucleus_id', np.arange(len(geom.get('rays', [])))))
            rows.append(isolate_tile(geom, self.mode, residual, node_cap, self.scale))
        return rows

    def tokens(self, bag: dict[str, Any], node_cap: int | None) -> list[dict[str, Any]]:
        key = (bag['patient_id'], node_cap)
        cached = self._tokens.get(key)
        if cached is None:
            cached = self._compute_tokens(bag, node_cap)
            self._tokens[key] = cached
        return cached

    def _scene_path(self, patient_id: str) -> Path:
        return self._scene_disk / f'{patient_id}.npz'

    def _remember_scenes(self, patient_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._scenes[patient_id] = rows
        self._scenes.move_to_end(patient_id)
        if not self._keep_all_scenes:
            while len(self._scenes) > SCENE_RAM_PATIENTS:
                self._scenes.popitem(last=False)
        return rows

    def _write_scenes(self, patient_id: str, rows: list[dict[str, Any]]) -> None:
        path = self._scene_path(patient_id)
        if path.is_file():
            return
        self._scene_disk.mkdir(parents=True, exist_ok=True)
        atomic_npz(path, compressed=False, **_pack_scenes(rows, self._scene_key_names))

    def _load_disk_scenes(self, patient_id: str) -> list[dict[str, Any]] | None:
        path = self._scene_path(patient_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            packed = {key: data[key] for key in data.files}
        return _unpack_scenes(packed, self._scene_key_names)

    def scenes(self, bag: dict[str, Any]) -> list[dict[str, Any]]:
        patient_id = bag['patient_id']
        cached = self._scenes.get(patient_id)
        if cached is not None:
            self._scenes.move_to_end(patient_id)
            return cached
        loaded = self._load_disk_scenes(patient_id)
        if loaded is not None:
            return self._remember_scenes(patient_id, loaded)
        rows = [_compact_scene(load_tile_scene(self.scene, tile), need_dino=self.need_dino, transform=self.scene_transform) for tile in bag['tiles']]
        self._write_scenes(patient_id, rows)
        return self._remember_scenes(patient_id, rows)

    def prefetch(self, bags: list[dict[str, Any]], node_cap: int | None, *, scenes: bool) -> None:
        from collections import defaultdict
        pending = bags
        disk_hits = 0
        if scenes:
            pending = []
            for bag in bags:
                if not self._scene_path(bag['patient_id']).is_file():
                    pending.append(bag)
                    continue
                disk_hits += 1
                if self._keep_all_scenes:
                    loaded = self._load_disk_scenes(bag['patient_id'])
                    if loaded is not None:
                        self._remember_scenes(bag['patient_id'], loaded)
        by_slide: dict[str, list[tuple[dict[str, Any], int, dict[str, str]]]] = defaultdict(list)
        scratch: dict[str, list] = {}
        for bag in pending:
            for index, tile in enumerate(bag['tiles']):
                by_slide[str(tile['slide_id'])].append((bag, index, tile))
            if scenes:
                scratch[bag['patient_id']] = [None] * len(bag['tiles'])
        for slide_id in sorted(by_slide):
            for bag, index, tile in by_slide[slide_id]:
                if scenes:
                    packed = scratch[bag['patient_id']]
                    if packed[index] is None:
                        packed[index] = _compact_scene(load_tile_scene(self.scene, tile), need_dino=self.need_dino, transform=self.scene_transform)
                    if all((item is not None for item in packed)):
                        self._write_scenes(bag['patient_id'], packed)
                        if self._keep_all_scenes:
                            self._remember_scenes(bag['patient_id'], packed)
                        scratch.pop(bag['patient_id'], None)
                elif self.mode is not None:
                    key = (bag['patient_id'], node_cap)
                    packed = self._tokens.get(key)
                    if packed is None:
                        packed = [None] * len(bag['tiles'])
                        self._tokens[key] = packed
                    if packed[index] is None:
                        geom = load_tile_geometry(self.geom, tile)
                        residual = None
                        if self.residual is not None:
                            residual = self.residual.get(tile['slide_id'], tile['graph_id'], geom.get('nucleus_id', np.arange(len(geom.get('rays', [])))))
                        packed[index] = isolate_tile(geom, self.mode, residual, node_cap, self.scale)
            if self.geom is not None:
                self.geom._store.pop(slide_id, None)
            if self.scene is not None:
                self.scene._store.pop(slide_id, None)
            if self.residual is not None:
                self.residual._store.pop(slide_id, None)
        if scenes:
            for patient_id, packed in list(scratch.items()):
                if all((item is not None for item in packed)):
                    self._write_scenes(patient_id, packed)
                    if self._keep_all_scenes:
                        self._remember_scenes(patient_id, packed)
        print(f'prefetched arm={self.arm} bags={len(bags)} token_patients={len(self._tokens)} scene_patients={len(self._scenes)} scene_disk_hits={disk_hits}', flush=True)

    def drop_raw_slides(self) -> None:
        if self.geom is not None:
            self.geom._store.clear()
        if self.scene is not None:
            self.scene._store.clear()
        if self.residual is not None:
            self.residual._store.clear()

def _store(base: Path, arm: str, task: str) -> FeatureStore:
    return FeatureStore(base, arm, task)

def _fit_token_scale(store: FeatureStore, bags: list[dict[str, Any]]) -> NodePooledScale:
    from collections import defaultdict
    by_slide: dict[str, list] = defaultdict(list)
    for bag in bags:
        for tile in bag['tiles']:
            by_slide[str(tile['slide_id'])].append(tile)
    rays, geom, include, valid = ([], [], [], [])
    for slide_id in sorted(by_slide):
        for tile in by_slide[slide_id]:
            row = load_tile_geometry(store.geom, tile)
            raw_rays = np.asarray(row['rays'], np.float32)
            raw_geom = np.asarray(row['node3d'], np.float32)[:, :9] if len(row['node3d']) else np.zeros((0, 9), np.float32)
            mask = np.asarray(row['include'], bool).reshape(-1)
            finite = np.isfinite(raw_geom).all(axis=1) if len(raw_geom) else np.zeros(0, bool)
            rays.append(raw_rays)
            geom.append(raw_geom)
            include.append(mask)
            valid.append(mask & np.asarray(row['valid3d'], bool) & finite)
        store.geom._store.pop(slide_id, None)
    if not rays:
        raise RuntimeError('token scale saw no FIT tiles')
    return fit_node_pooled_scale(np.concatenate(rays), np.concatenate(geom), np.concatenate(include), np.concatenate(valid))

def _shared_geometry_scale_path(task: str) -> Path:
    return result_root() / 'cache' / 'fit_geometry_scale' / task / 'geometry_scale.json'

def _load_cached_geometry_scale(task: str, dest: Path) -> NodePooledScale | None:
    candidates = [_shared_geometry_scale_path(task), dest / 'geometry_scale.json']
    for arm in GEOM_MODE:
        candidates.append(arm_dir(task, arm) / 'geometry_scale.json')
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            return NodePooledScale.from_dict(read_json(path))
        except Exception:
            continue
    return None

def _fit_scene_stats(store: FeatureStore, bags: list[dict[str, Any]], kind: str) -> dict[str, np.ndarray]:
    from collections import defaultdict
    stats = SceneStats(kind)
    by_slide: dict[str, list] = defaultdict(list)
    for bag in bags:
        for tile in bag['tiles']:
            by_slide[str(tile['slide_id'])].append((bag['patient_id'], tile))
    for slide_id in sorted(by_slide):
        for patient_id, tile in by_slide[slide_id]:
            stats.add(patient_id, load_tile_scene(store.scene, tile))
        store.scene._store.pop(slide_id, None)
    return stats.finalize()

def _shared_scene_stats_path(task: str, kind: str) -> Path:
    return result_root() / 'cache' / f'fit_scene_stats_{kind.lower()}' / task / 'scene_stats.json'

def _stats_from_json(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == 'schema':
            out[key] = np.asarray(value, np.int64)
        else:
            out[key] = np.asarray(value, np.float32)
    return out

def _load_or_fit_scene_stats(store: FeatureStore, fit_bags: list[dict[str, Any]], task: str, kind: str) -> dict[str, np.ndarray]:
    shared = _shared_scene_stats_path(task, kind)
    if shared.is_file():
        try:
            return _stats_from_json(read_json(shared))
        except Exception:
            pass
    stats = _fit_scene_stats(store, fit_bags, kind)
    atomic_json(shared, {key: np.asarray(value).tolist() for key, value in stats.items()})
    return stats

def _bind_inputs(store: FeatureStore, fit_bags: list[dict[str, Any]], arm: str, dest: Path) -> None:
    if store.mode is not None:
        scale = _load_cached_geometry_scale(store.task, dest)
        if scale is None:
            scale = _fit_token_scale(store, fit_bags)
            print(f'fit geometry_scale task={store.task} arm={arm}', flush=True)
        else:
            print(f'reuse geometry_scale task={store.task} arm={arm}', flush=True)
        store.scale = scale
        payload = scale.as_dict()
        atomic_json(_shared_geometry_scale_path(store.task), payload)
        atomic_json(dest / 'geometry_scale.json', payload)
    if arm in SCENE_ARMS:
        import hashlib
        kind = 'P' if arm.startswith('P') else 'E'
        route_arm = RECIPES[arm].get('route_arm', arm)
        stats = _load_or_fit_scene_stats(store, fit_bags, store.task, kind)
        digest = hashlib.sha256(stats['mean'].tobytes() + stats['std'].tobytes()).hexdigest()[:16]
        store._scene_disk = result_root() / 'cache' / f'scenes_{kind.lower()}_conditional_geometry' / arm / digest
        frozen = {key: np.array(value, copy=True) for key, value in stats.items()}
        store.scene_transform = lambda scene, frozen=frozen, route_arm=route_arm, kind=kind: apply_scene(scene, frozen, route_arm, kind)
        atomic_json(dest / 'scene_stats.json', {key: np.asarray(value).tolist() for key, value in stats.items()})

def _geometry_groups(model: nn.Module) -> list[list[nn.Parameter]]:
    if isinstance(model, GeometryMIL):
        if model.combined:
            return [list(model.dual.stream_2d.parameters()), list(model.dual.stream_3d.parameters())]
        return [list(model.single.parameters())]
    if isinstance(model, FusionMIL):
        return [list(model.geometry.parameters())]
    if isinstance(model, PrototypeMIL):
        return [list(model.route.geom_embed.parameters()) + list(model.route.V.parameters())]
    if isinstance(model, ExpertMIL):
        return [list(model.expert.node_encoder.parameters()) + list(model.expert.message.parameters()) + list(model.expert.local.parameters())]
    return []

def _group_has_grad(params: list[nn.Parameter]) -> bool:
    for item in params:
        grad = item.grad
        if grad is None or not torch.isfinite(grad).all():
            continue
        if float(grad.detach().abs().sum()) > 0:
            return True
    return False

def _move_optimizer(opt, device) -> None:
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def _save_last(path: Path, model, opt, epoch: int, stale: int, best: dict[str, Any], node_cap, chunk, geom_grad_ok) -> None:
    torch.save({'last_model': {key: value.detach().cpu() for key, value in model.state_dict().items()}, 'optimizer': opt.state_dict(), 'epoch': int(epoch), 'stale': int(stale), 'best_score': best['score'], 'best_epoch': best['epoch'], 'best_state': best['state'], 'best_aux': best.get('aux'), 'node_cap': node_cap, 'chunk': chunk, 'geom_grad_ok': geom_grad_ok}, path)

def _forward(model, bag, store: FeatureStore, device, chunk: int, node_cap: int | None):
    dino = None
    if bag.get('dino_global') is not None:
        dino = torch.as_tensor(bag['dino_global'], device=device)
    if isinstance(model, ImageMIL):
        return model(dino)
    if isinstance(model, GeometryMIL):
        return model(store.tokens(bag, node_cap), chunk=chunk, node_cap=node_cap)
    if isinstance(model, FusionMIL):
        return model(dino, store.tokens(bag, node_cap), chunk=chunk, node_cap=node_cap)
    if isinstance(model, PrototypeMIL):
        return model(dino, store.scenes(bag), chunk=chunk, node_cap=node_cap)
    if isinstance(model, ExpertMIL):
        return model(store.scenes(bag), chunk=chunk, node_cap=node_cap)
    raise RuntimeError(model.__class__.__name__)

def _predict_split(model, bags, store, device, chunk, node_cap, classes: int) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for bag in bags:
            try:
                logits = _forward(model, bag, store, device, chunk, node_cap)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise
            values = np.asarray(logits.detach().float().cpu().numpy(), np.float32).reshape(-1)
            if values.size != classes:
                raise RuntimeError(f'logits {values.shape} != {classes}')
            rows.append({'patient_id': bag['patient_id'], 'split': bag['split'], 'label': int(bag['label']), 'logit': values.astype(np.float32).tolist(), 'prob': softmax(values).astype(np.float64).tolist()})
    return rows

def train_arm(task: str, arm: str, *, device: str='cuda', data_root=None, seed: int=SEED) -> dict[str, Any]:
    dest = arm_dir(task, arm)
    marker = dest / 'metrics.json'
    if marker.is_file():
        return {'status': 'skip', 'path': str(marker)}
    dest.mkdir(parents=True, exist_ok=True)
    recipe = dict(RECIPES[arm])
    spec = TASK_SPEC[task]
    _seed_all(seed)
    need_image = arm in IMAGE_ARMS
    base, bags, _ = build_bags(task, data_root=data_root, need_image=need_image)
    splits = split_bags(bags)
    store = _store(base, arm, task)
    model = build_model(arm, spec['classes'], {**recipe, 'mode': GEOM_MODE.get(arm)})
    last_path = dest / 'last.pt'
    resume = last_path.is_file()
    device_t = _device(device)
    model = model.to(device_t)
    opt = _optimizer(model, recipe)
    loss_fn = nn.CrossEntropyLoss()
    chunk = gpu_tile_chunk(device)
    node_cap = None
    history = []
    best = {'score': float('-inf'), 'epoch': 0, 'state': None}
    stale = 0
    geom_grad_ok = None
    start_epoch = 1
    if resume:
        payload = torch.load(last_path, map_location='cpu')
        model.load_state_dict(payload['last_model'])
        model = model.to(device_t)
        opt = _optimizer(model, recipe)
        opt.load_state_dict(payload['optimizer'])
        _move_optimizer(opt, device_t)
        start_epoch = int(payload['epoch']) + 1
        stale = int(payload.get('stale') or 0)
        best = {'score': payload.get('best_score', float('-inf')), 'epoch': payload.get('best_epoch', 0), 'state': payload.get('best_state'), 'aux': payload.get('best_aux')}
        node_cap = payload.get('node_cap')
        chunk = int(payload.get('chunk') or chunk)
        geom_grad_ok = payload.get('geom_grad_ok')
        hist_path = dest / 'history.json'
        if hist_path.is_file():
            from ..io_utils import read_json
            history = list(read_json(hist_path))
        print(f'resume {task}/{arm} from epoch {start_epoch}', flush=True)
    all_bags = splits['fit'] + splits['val'] + splits['test']
    _bind_inputs(store, splits['fit'], arm, dest)
    if store.mode is not None:
        store.prefetch(all_bags, node_cap, scenes=False)
    if arm in SCENE_ARMS:
        store.prefetch(all_bags, node_cap, scenes=True)
    store.drop_raw_slides()
    if arm in {'P2', 'P3'} and (not resume):
        from celllift.morphology_interaction.route_b.models import fit_prototypes
        mean, basis, centers = fit_prototypes(collect_fit_nucleus_dino_from_store(store, splits['fit'], seed=seed), seed=seed)
        model.load_prototypes(mean, basis, centers)
    fit = list(splits['fit'])
    max_epochs = int(recipe['epochs'])
    if start_epoch > max_epochs:
        start_epoch = max_epochs + 1
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = np.arange(len(fit))
        rng.shuffle(order)
        running = 0.0
        seen = 0
        window: list[int] = []
        for step, index in enumerate(order, start=1):
            window.append(int(index))
            if len(window) < ACCUM and step != len(order):
                continue
            width = len(window)
            opt.zero_grad(set_to_none=True)
            for bag_index in window:
                bag = fit[bag_index]
                try:
                    logits = _forward(model, bag, store, device_t, chunk, node_cap)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if node_cap is None:
                        node_cap = 512
                    elif node_cap > 128:
                        node_cap = max(128, node_cap // 2)
                    else:
                        chunk = max(1, chunk // 2)
                    store._tokens.clear()
                    if store.mode is not None:
                        store.prefetch(all_bags, node_cap, scenes=False)
                    logits = _forward(model, bag, store, device_t, chunk, node_cap)
                target = torch.as_tensor([bag['label']], device=device_t)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                loss = loss_fn(logits, target) / width
                loss.backward()
                running += float(loss.item()) * width
                seen += 1
                if geom_grad_ok is None and getattr(model, 'has_geometry', False):
                    groups = _geometry_groups(model)
                    geom_grad_ok = bool(groups) and all((_group_has_grad(group) for group in groups))
                    if not geom_grad_ok:
                        if seen < 32:
                            geom_grad_ok = None
                        else:
                            raise RuntimeError(f'{task}/{arm} geometry gradient is zero')
            opt.step()
            window = []
        val_rows = _predict_split(model, splits['val'], store, device_t, chunk, node_cap, spec['classes'])
        y = np.asarray([row['label'] for row in val_rows], np.int64)
        probs = np.asarray([row['prob'] for row in val_rows], np.float64)
        score, aux = primary_from_probs(task, y, probs)
        record = {'epoch': epoch, 'fit_loss': running / max(seen, 1), 'val_primary': score, 'val_aux': aux, 'node_cap': node_cap, 'chunk': chunk}
        history.append(record)
        atomic_json(dest / 'history.json', history)
        if better(score, best['score']):
            best = {'score': score, 'epoch': epoch, 'state': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'aux': aux}
            stale = 0
            torch.save({'model': best['state'], 'epoch': epoch, 'score': score}, dest / 'checkpoint.pt')
        else:
            stale += 1
        _save_last(last_path, model, opt, epoch, stale, best, node_cap, chunk, geom_grad_ok)
        if stale >= int(recipe['patience']):
            break
    if best['state'] is None:
        raise RuntimeError(f'{task}/{arm} produced no checkpoint')
    model.load_state_dict(best['state'])
    threshold = None
    if spec['classes'] == 2:
        from .evaluate import binary_threshold
        val_rows = _predict_split(model, splits['val'], store, device_t, chunk, node_cap, spec['classes'])
        threshold = binary_threshold(np.asarray([r['label'] for r in val_rows]), np.asarray([r['prob'] for r in val_rows])[:, 1])
    all_rows = []
    metrics = {}
    for name in ('fit', 'val', 'test'):
        rows = _predict_split(model, splits[name], store, device_t, chunk, node_cap, spec['classes'])
        y = np.asarray([row['label'] for row in rows], np.int64)
        probs = np.asarray([row['prob'] for row in rows], np.float64)
        primary, aux = primary_from_probs(task, y, probs, threshold)
        metrics[name] = {'primary': primary, **aux}
        all_rows.extend(rows)
    atomic_parquet(dest / 'predictions.parquet', all_rows)
    payload = {**protocol_meta(), 'status': 'PASS', 'task': task, 'arm': arm, 'best_epoch': best['epoch'], 'best_val_primary': best['score'], 'threshold': threshold, 'geom_grad_ok': geom_grad_ok, 'node_cap': node_cap, 'chunk': chunk, 'metrics': metrics, 'n_fit': len(splits['fit']), 'n_val': len(splits['val']), 'n_test': len(splits['test']), 'resumed': bool(resume)}
    atomic_json(marker, payload)
    return payload
