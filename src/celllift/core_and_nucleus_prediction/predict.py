from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.pretrained import resnet18 as _packaged_resnet18
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any
import numpy as np
from celllift.runtime import torch
from torch.utils.data import DataLoader
from .constants import RESULT_ROOT
from .data import ArvanitiWindowDataset, LizardNucleusDataset, LizardTileDataset, pad_collate_windows
from .evaluate import score_arvaniti_predictions, score_lizard_predictions
from .models import DeepSetsWindow, FeatureFusion, ImageNodeExpert, MobileNetV1Half, NucleusMLP, RelationExpert, SpatialFieldNet, WindowFusion
from .train import ARMS, _out, _save_logits

def _route_arm(arm: str) -> tuple[str, str, str]:
    if arm.startswith(('N', 'J', 'LH', 'LA')) or arm in {'B_crop', 'B_dino'}:
        dataset = 'lizard'
    else:
        dataset = 'arvaniti'
    if arm in {'B_paper', 'B_crop', 'B_dino'}:
        route = 'baseline'
    elif arm.startswith(('G', 'N')):
        route = 'geom'
    elif arm.startswith('J'):
        route = 'relation'
    elif arm in {'H2', 'H23', 'LH2', 'LH23'}:
        route = 'fusion'
    elif arm in {'A2', 'AS', 'LA2', 'LA3'}:
        route = 'interact'
    else:
        route = 'spatial'
    return (dataset, route, arm)

def _load_state(arm: str, device: str):
    dataset, route, name = _route_arm(arm)
    path = RESULT_ROOT / dataset / route / name / 'seed42' / 'model.pt'
    if not path.is_file():
        return (None, path)
    return (torch.load(path, map_location=device), path)

def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    value = value - value.max(axis=-1, keepdims=True)
    exp = np.exp(value)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)

def predict_arvaniti(arm: str, device: str='cuda') -> dict[str, Any]:
    state, ckpt = _load_state(arm, device)
    if state is None:
        return {'status': 'WAIT', 'arm': arm}
    geom = {'B_paper': 'g2', 'H2': 'g2', 'H23': 'g23', 'A2': 'g2', 'AS': 'g23', 'C2': 'g23', 'C3': 'g23'}.get(arm, arm.lower())
    need_dino = arm in {'H2', 'H23', 'A2', 'AS', 'C2', 'C3'}
    model = None
    if arm == 'B_paper':
        model = MobileNetV1Half().to(device)
        model.load_state_dict(state)

        def fn(batch):
            return (model(batch['rgb'].to(device)), batch['label'].to(device))
    elif arm.startswith('G'):
        model = DeepSetsWindow({'g2': 38, 'g3': 9, 'gr': 9, 'g23': 47, 'g2r': 47}[geom]).to(device)
        model.load_state_dict(state)

        def fn(batch):
            return (model(batch['tokens'].to(device), batch['mask'].to(device)), batch['label'].to(device))
    elif arm in {'H2', 'H23'}:
        image = MobileNetV1Half().to(device)
        b = RESULT_ROOT / 'arvaniti/baseline/B_paper/seed42/model.pt'
        if b.is_file():
            image.load_state_dict(torch.load(b, map_location=device))
        image.eval()
        model = WindowFusion(512 + 384, 38 if arm == 'H2' else 47).to(device)
        model.load_state_dict(state)

        def fn(batch):
            rgb = batch['rgb'].to(device) if 'rgb' in batch else torch.zeros(len(batch['label']), 3, 224, 224, device=device)
            emb = torch.cat((image.embedding(rgb), batch['image_summary'].to(device)), -1)
            return (model(emb, batch['tokens'].to(device), batch['mask'].to(device)), batch['label'].to(device))
    elif arm in {'A2', 'AS'}:
        node_dim = 38 if arm == 'A2' else 47
        model = ImageNodeExpert(node_dim, 3, classes=4, pool=True).to(device)
        model.load_state_dict(state)

        def fn(batch):
            logits = []
            for i in range(len(batch['label'])):
                n = int(batch['mask'][i].sum().item())
                tokens = batch['tokens'][i, :n].to(device)
                dino = batch['dino'][i, :n].to(device)
                src = torch.as_tensor(batch['src'][i], device=device, dtype=torch.long).view(-1)
                dst = torch.as_tensor(batch['dst'][i], device=device, dtype=torch.long).view(-1)
                edge = torch.as_tensor(batch['edge'][i], device=device).float()
                if edge.ndim == 1:
                    edge = edge.view(-1, 3)
                logits.append(model(dino, tokens, edge if edge.numel() else torch.zeros((0, 3), device=device), src, dst, batch['mask'][i, :n].to(device)))
            return (torch.stack(logits), batch['label'].to(device))
    else:
        from .fields import arvaniti_field_batch
        model = SpatialFieldNet().to(device)
        model.load_state_dict(state)

        def fn(batch):
            field, dino = arvaniti_field_batch(batch, arm, device)
            return (model(field, dino), batch['label'].to(device))
    model.eval()
    path = ckpt.parent
    reports = {}
    for split, supervised in (('val', True), ('test', True), ('val_infer', False), ('test_infer', False)):
        want = 'val' if split.startswith('val') else 'test'
        npz = path / f'{split}_logits.npz'
        metrics_path = path / f'{split}_metrics.json'
        packed = None
        if npz.is_file() and npz.stat().st_size > 0:
            if metrics_path.is_file():
                try:
                    scored = json.loads(metrics_path.read_text(encoding='utf-8'))
                    qwk = scored.get('core_qwk_p1')
                    if qwk is not None and qwk == qwk:
                        reports[split] = {key: scored[key] for key in scored if key != 'cores'}
                        continue
                except Exception:
                    pass
            packed = dict(np.load(npz))
        else:
            data = ArvanitiWindowDataset(want, geom=geom, supervised=supervised, load_dino=need_dino, load_geometry=arm != 'B_paper')
            if not data.rows:
                continue
            loader = DataLoader(data, batch_size=8 if arm in {'A2', 'AS', 'C2', 'C3'} else 16, shuffle=False, collate_fn=pad_collate_windows)
            ids, cores, logits, labels = ([], [], [], [])
            with torch.no_grad():
                for batch in loader:
                    if arm == 'B_paper' and 'rgb' not in batch:
                        continue
                    out, target = fn(batch)
                    ids.extend(batch['graph_id'])
                    cores.extend(batch['core_id'])
                    logits.append(out.cpu().numpy())
                    labels.append(target.cpu().numpy())
            if not logits:
                continue
            packed = {'graph_id': np.asarray(ids), 'core_id': np.asarray(cores), 'logits': np.concatenate(logits).astype(np.float32), 'label': np.concatenate(labels).astype(np.int64)}
            np.savez_compressed(npz, **packed)
        rows = [{'graph_id': gid, 'core_id': cid, 'probs': _softmax(logit), 'label': int(lab)} for gid, cid, logit, lab in zip(packed['graph_id'], packed['core_id'], packed['logits'], packed['label'])]
        scored = score_arvaniti_predictions(rows, 'val' if want == 'val' else 'test')
        scored.pop('cores', None)
        metrics_path.write_text(json.dumps(scored, indent=2), encoding='utf-8')
        reports[split] = {key: scored[key] for key in scored if key != 'cores'}
    (path / 'official_metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return {'arm': arm, 'ckpt': str(ckpt), **reports}

def predict_lizard(arm: str, device: str='cuda') -> dict[str, Any]:
    state, ckpt = _load_state(arm, device)
    if state is None:
        return {'status': 'WAIT', 'arm': arm}
    path = ckpt.parent
    reports = {}
    for split in ('val', 'test'):
        npz = path / f'{split}_logits.npz'
        metrics_path = path / f'{split}_metrics.json'
        if npz.is_file() and npz.stat().st_size > 0:
            scored = None
            if metrics_path.is_file():
                try:
                    scored = json.loads(metrics_path.read_text(encoding='utf-8'))
                except Exception:
                    scored = None
            if not scored or scored.get('macro_f1') is None:
                packed = np.load(npz)
                group_id = packed['group_id'] if 'group_id' in packed.files else [None] * len(packed['label'])
                rows = [{'label': int(y), 'pred': int(p), 'group_id': g} for y, p, g in zip(packed['label'], packed['logits'].argmax(-1), group_id)]
                scored = score_lizard_predictions(rows)
                metrics_path.write_text(json.dumps(scored, indent=2), encoding='utf-8')
            reports[split] = scored
            continue
        if arm in {'B_dino', 'B_crop'} or arm.startswith('N') or arm in {'LH2', 'LH23'}:
            geom = {'LH2': 'n2', 'LH23': 'n23'}.get(arm, arm.lower() if arm.startswith('N') else 'n2')
            data = LizardNucleusDataset(split, geom=geom, crops=arm in {'B_crop', 'LH2', 'LH23'}, load_dino=True)
            if arm == 'B_dino':
                model = NucleusMLP(384, 6, 128).to(device)
                model.load_state_dict(state)

                def fn(batch):
                    return (model(batch['dino'].to(device)), batch['label'].to(device))
            elif arm == 'B_crop':
                import torchvision
                model = _packaged_resnet18(weights=None)
                model.fc = torch.nn.Linear(512, 6)
                model.load_state_dict(state)
                model = model.to(device)

                def fn(batch):
                    return (model(batch['rgb'].to(device)), batch['label'].to(device))
            elif arm.startswith('N'):
                model = NucleusMLP({'n2': 38, 'n3': 9, 'nr': 9, 'n23': 47, 'n2r': 47}[arm.lower()]).to(device)
                model.load_state_dict(state)

                def fn(batch):
                    return (model(batch['token'].to(device)), batch['label'].to(device))
            else:
                import torchvision
                crop = _packaged_resnet18(weights=None)
                crop.fc = torch.nn.Identity()
                b = RESULT_ROOT / 'lizard/baseline/B_crop/seed42/model.pt'
                if b.is_file():
                    crop.load_state_dict({k: v for k, v in torch.load(b, map_location='cpu').items() if not k.startswith('fc')}, strict=False)
                crop = crop.to(device).eval()
                model = FeatureFusion(512 + 384, 38 if arm == 'LH2' else 47, 6).to(device)
                model.load_state_dict(state)

                def fn(batch):
                    with torch.no_grad():
                        image = torch.cat((crop(batch['rgb'].to(device)), batch['dino'].to(device)), -1)
                    return (model(image, batch['token'].to(device)), batch['label'].to(device))
            model.eval()
            loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=0, collate_fn=lambda items: {'token': torch.from_numpy(np.stack([item['token'] for item in items])), 'dino': torch.from_numpy(np.stack([item['dino'] for item in items])), 'rgb': torch.from_numpy(np.stack([item['crop'] if item.get('crop') is not None else np.zeros((3, 224, 224), np.float32) for item in items])), 'label': torch.tensor([item['label'] for item in items], dtype=torch.long), 'id_miss': torch.tensor([bool(item.get('id_miss')) for item in items]), 'graph_id': [item['graph_id'] for item in items], 'nucleus_id': np.asarray([item['nucleus_id'] for item in items], np.int64), 'group_id': [item.get('group_id') for item in items]})
            ids, nids, groups, logits, labels = ([], [], [], [], [])
            with torch.no_grad():
                for batch in loader:
                    out, target = fn(batch)
                    keep = ~batch['id_miss'].cpu().numpy().astype(bool)
                    if not keep.any():
                        continue
                    ids.extend([gid for gid, flag in zip(batch['graph_id'], keep) if flag])
                    nids.append(batch['nucleus_id'][keep])
                    groups.extend([gid for gid, flag in zip(batch['group_id'], keep) if flag])
                    logits.append(out.cpu().numpy()[keep])
                    labels.append(target.cpu().numpy()[keep])
            packed_logits = np.concatenate(logits) if logits else np.zeros((0, 6), np.float32)
            packed_labels = np.concatenate(labels) if labels else np.zeros((0,), np.int64)
            _save_logits(path, f'{split}_logits.npz', ids, packed_logits, packed_labels, nucleus_id=np.concatenate(nids) if nids else np.zeros((0,), np.int64), group_id=np.asarray(groups))
            rows = [{'label': int(y), 'pred': int(p), 'group_id': g} for y, p, g in zip(packed_labels, packed_logits.argmax(-1), groups)]
        else:
            geom = {'J2': 'j2', 'JS': 'js', 'JR': 'jr', 'LA2': 'a2', 'LA3': 'a3'}[arm]
            data = LizardTileDataset(split, geom=geom, load_dino=arm.startswith('LA'))
            if arm.startswith('J'):
                node_dim = 38 if arm == 'J2' else 47
                edge_dim = 7 if arm == 'JR' else 3
                model = RelationExpert(node_dim, edge_dim, classes=6).to(device)
            else:
                node_dim = 38 if arm == 'LA2' else 47
                edge_dim = 3 if arm == 'LA2' else 7
                model = ImageNodeExpert(node_dim, edge_dim, classes=6, pool=False).to(device)
            model.load_state_dict(state)
            model.eval()
            ids, nids, groups, logits, labels = ([], [], [], [], [])
            with torch.no_grad():
                for item in data:
                    tokens = torch.as_tensor(item['tokens'], device=device).float()
                    src = torch.as_tensor(item['src'], device=device, dtype=torch.long).view(-1)
                    dst = torch.as_tensor(item['dst'], device=device, dtype=torch.long).view(-1)
                    edge = torch.as_tensor(item['edge'], device=device).float()
                    if arm == 'JR':
                        edge3 = torch.as_tensor(item['edge3'], device=device).float()
                        edge = torch.cat((edge, edge3), -1) if edge.numel() else torch.zeros((0, 7), device=device)
                    if arm.startswith('LA'):
                        dino = torch.as_tensor(item['dino'], device=device).float()
                        if arm == 'LA3':
                            edge3 = torch.as_tensor(item['edge3'], device=device).float()
                            edge = torch.cat((edge, edge3), -1) if edge.numel() else torch.zeros((0, 7), device=device)
                        out = model(dino, tokens, edge if edge.numel() else torch.zeros((0, edge_dim), device=device), src, dst, None)
                    else:
                        out = model(tokens, edge if edge.numel() else torch.zeros((0, edge_dim), device=device), src, dst)
                    lab = np.asarray(item['label'])
                    owned = np.asarray(item['owned']) > 0
                    keep = (lab >= 0) & owned
                    if not keep.any():
                        continue
                    pred = out.cpu().numpy()[keep]
                    ids.extend([item['graph_id']] * int(keep.sum()))
                    nids.append(np.asarray(item['ids'])[keep])
                    groups.extend([item.get('group_id')] * int(keep.sum()))
                    logits.append(pred)
                    labels.append(lab[keep])
            packed_logits = np.concatenate(logits) if logits else np.zeros((0, 6), np.float32)
            packed_labels = np.concatenate(labels) if labels else np.zeros((0,), np.int64)
            _save_logits(path, f'{split}_logits.npz', ids, packed_logits, packed_labels, nucleus_id=np.concatenate(nids) if nids else np.zeros((0,), np.int64), group_id=np.asarray(groups))
            rows = [{'label': int(y), 'pred': int(p), 'group_id': g} for y, p, g in zip(packed_labels, packed_logits.argmax(-1), groups)]
        scored = score_lizard_predictions(rows)
        (path / f'{split}_metrics.json').write_text(json.dumps(scored, indent=2), encoding='utf-8')
        reports[split] = scored
    (path / 'official_metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return {'arm': arm, 'ckpt': str(ckpt), **reports}

def rescore_arvaniti_arm(arm: str) -> dict[str, Any]:
    dataset, route, name = _route_arm(arm)
    if dataset != 'arvaniti':
        return {'status': 'SKIP', 'arm': arm}
    path = RESULT_ROOT / dataset / route / name / 'seed42'
    reports = {}
    for split in ('val', 'test', 'val_infer', 'test_infer'):
        npz = path / f'{split}_logits.npz'
        if not npz.is_file():
            continue
        packed = np.load(npz)
        rows = [{'graph_id': str(gid), 'core_id': str(cid), 'probs': _softmax(logit), 'label': int(lab)} for gid, cid, logit, lab in zip(packed['graph_id'], packed['core_id'], packed['logits'], packed['label'])]
        scored = score_arvaniti_predictions(rows, 'val' if split.startswith('val') else 'test')
        scored.pop('cores', None)
        (path / f'{split}_metrics.json').write_text(json.dumps(scored, indent=2), encoding='utf-8')
        reports[split] = scored
    if reports:
        (path / 'official_metrics.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return {'arm': arm, **{k: {kk: vv for kk, vv in v.items() if kk != 'bootstrap'} for k, v in reports.items()}}

def predict_arm(arm: str, device: str='cuda') -> dict[str, Any]:
    if arm not in ARMS:
        raise SystemExit(arm)
    dataset, route, name = _route_arm(arm)
    official = RESULT_ROOT / dataset / route / name / 'seed42' / 'official_metrics.json'
    if official.is_file():
        return {'status': 'EXISTS', 'arm': arm, **json.loads(official.read_text(encoding='utf-8'))}
    path = RESULT_ROOT / dataset / route / name / 'seed42'
    if dataset == 'arvaniti' and all(((path / f'{split}_logits.npz').is_file() for split in ('val', 'test', 'val_infer', 'test_infer'))):
        return rescore_arvaniti_arm(arm)
    return predict_arvaniti(arm, device) if dataset == 'arvaniti' else predict_lizard(arm, device)

def predict_all(device: str='cuda') -> dict[str, Any]:
    return {arm: predict_arm(arm, device) for arm in ARMS}
