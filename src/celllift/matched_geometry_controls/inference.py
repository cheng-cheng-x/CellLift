from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import ResourcePath as Path
from typing import Iterable
from celllift.runtime import json
from .geometry import selected_scene_descriptor_torch
from .io_utils import atomic_json, atomic_torch, safe_component

def graph_batches(items: Iterable[tuple[object, object]], max_graphs: int, max_nodes: int):
    batch = []
    nodes = 0
    for graph, metadata in items:
        count = len(graph.nucleus_id)
        if batch and (len(batch) >= max_graphs or nodes + count > max_nodes):
            yield batch
            batch = []
            nodes = 0
        if count > max_nodes:
            if batch:
                yield batch
                batch = []
                nodes = 0
            yield [(graph, metadata)]
            continue
        batch.append((graph, metadata))
        nodes += count
    if batch:
        yield batch

def device_limits(device_name: str) -> tuple[int, int]:
    name = device_name.upper()
    return (8, 8192) if 'A800' in name else (4, 4096)

def load_model(modules, checkpoint: str | Path, input_manifest: str | Path, device: str='cuda'):
    import torch
    info = json.loads(Path(input_manifest).read_text(encoding='utf-8'))
    config = info['config']
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = modules.model.ConditionalSceneNetwork(info['full_stats'], config).to(device).eval()
    model.attach_scorer(state['scorer_normalization'])
    model.load_state_dict(state['model'])
    return (model, info)

def infer_items(modules, model, items, destination_root: str | Path, device: str='cuda', max_graphs: int | None=None, max_nodes: int | None=None, *, total: int | None=None, progress_path: str | Path | None=None) -> list[dict]:
    import torch
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True)
    if max_graphs is None or max_nodes is None:
        gpu = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else 'CPU'
        default_graphs, default_nodes = device_limits(gpu)
        max_graphs = default_graphs if max_graphs is None else max_graphs
        max_nodes = default_nodes if max_nodes is None else max_nodes
    records = []
    with torch.inference_mode():
        for batch_index, packed in enumerate(graph_batches(items, int(max_graphs), int(max_nodes))):
            graph, metadata = modules.data.collate(packed)
            graph = modules.data.move(graph, device)
            proposal = model.propose(graph, scoring=True)
            scene = modules.model.selected(proposal, labels=proposal.labels)
            direct, valid_ncr = selected_scene_descriptor_torch(scene.nucleus, scene.cell)
            ptr = graph.graph_ptr.detach().cpu().numpy()
            labels = proposal.labels.detach().cpu()
            valid = scene.valid.detach().cpu()
            for index, meta in enumerate(metadata):
                begin, end = (int(ptr[index]), int(ptr[index + 1]))
                graph_id = str(meta['graph_id'])
                destination = root / f'{safe_component(graph_id)}.pt'
                payload = {'graph_id': graph_id, 'nucleus_id': graph.nucleus_id[begin:end].detach().cpu(), 'selected_candidate_id': labels[begin:end], 'valid': valid[begin:end], 'nucleus_center': scene.nucleus.center[begin:end].detach().cpu(), 'nucleus_transform': scene.nucleus.transform[begin:end].detach().cpu(), 'cell_center': scene.cell.center[begin:end].detach().cpu(), 'cell_transform': scene.cell.transform[begin:end].detach().cpu(), 'direct_geometry9': direct[begin:end].detach().cpu(), 'valid_ncr': valid_ncr[begin:end].detach().cpu(), 'metadata': meta}
                atomic_torch(destination, payload)
                records.append({'graph_id': graph_id, 'nodes': end - begin, 'path': str(destination), 'valid_scene': int(valid[begin:end].sum()), 'invalid_ncr': int((~valid_ncr[begin:end]).sum().cpu())})
            if progress_path is not None and batch_index % 16 == 0:
                atomic_json(progress_path, {'done': len(records), 'total': total, 'batches': batch_index + 1})
    if progress_path is not None:
        atomic_json(progress_path, {'done': len(records), 'total': total, 'batches': None, 'status': 'PASS'})
    atomic_json(root / 'manifest.json', {'status': 'PASS', 'graphs': len(records), 'items': records})
    return records
