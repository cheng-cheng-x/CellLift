from celllift.runtime import resource_path as _public_resource
import numpy as np
import torch

def sparse(f, reference, valid, ig_scores=None):
    adapter = getattr(f, 'adapter', getattr(getattr(f, 'f', None), 'adapter', None))
    threshold = getattr(adapter, 'thresholds', {}).get('H3', 0.5)

    def classification(p):
        return int(p[1] >= threshold) if p.shape[-1] == 2 else int(p.argmax())
    n = len(f.rich)
    seeds = [17, 42, 73]
    initial = []
    for seed in seeds:
        gen = torch.Generator(device=f.rich.device)
        gen.manual_seed(seed)
        initial.append(torch.randn(n, generator=gen, device=f.rich.device) * 0.1)
    theta = torch.nn.Parameter(torch.stack(initial))
    opt = torch.optim.Adam([theta], lr=0.05)
    with torch.no_grad():
        original = torch.softmax(f(f.rich), -1)
        original_class = classification(original[0])
    delta = f.rich - reference
    denom = valid.sum().clamp_min(1)
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        m = torch.sigmoid(theta)
        x = reference[None] + m[:, :, None] * delta[None]
        lp = torch.log_softmax(f(x), -1)
        kl = (original * (original.clamp_min(1e-12).log() - lp)).sum(-1)
        entropy = -(m * m.clamp_min(1e-08).log() + (1 - m) * (1 - m).clamp_min(1e-08).log())
        loss = (kl + 0.01 * (m * valid).sum(1) / denom + 0.001 * (entropy * valid).sum(1) / denom).sum()
        loss.backward()
        opt.step()
    masks = torch.sigmoid(theta).detach()
    ids = torch.nonzero(valid, as_tuple=True)[0]
    records = []
    methods = []
    for j, seed in enumerate(seeds):
        methods.append(('mask', seed, ids[torch.argsort(masks[j, ids], descending=True)]))
        gen = torch.Generator(device=f.rich.device)
        gen.manual_seed(seed)
        methods.append(('random', seed, ids[torch.randperm(len(ids), generator=gen, device=ids.device)]))
    if ig_scores is not None:
        score = torch.as_tensor(ig_scores, device=ids.device)
        methods.append(('IG', 42, ids[torch.argsort(score[ids], descending=True)]))
    endpoint_probabilities = {}
    for method, seed, order in methods:
        for fraction in (0, 0.05, 0.1, 0.2, 0.4, 1):
            keep = order[:int(np.ceil(fraction * len(order)))]
            x = reference.clone()
            x[keep] = f.rich[keep]
            with torch.no_grad():
                if fraction in (0, 1) and fraction in endpoint_probabilities:
                    p = endpoint_probabilities[fraction]
                else:
                    p = torch.softmax(f(x), -1)[0]
                    if fraction in (0, 1):
                        endpoint_probabilities[fraction] = p
            kl = float((original[0] * (original[0].clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())
            drop = float(original[0, original_class] - p[original_class])
            same = classification(p) == original_class
            records.append(dict(method=method, seed=seed, fraction=fraction, probability=p.cpu().numpy(), kl=kl, probability_drop=drop, same_class=same, faithful=same and kl <= 0.02 and (drop <= 0.05), restored_nodes=len(keep), eligible_nodes=len(ids), classification_threshold=threshold if len(p) == 2 else None))
    return (masks.cpu().numpy(), records)
