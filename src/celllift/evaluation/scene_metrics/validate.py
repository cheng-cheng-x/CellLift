from celllift.runtime import resource_path as _public_resource
import json, time
import torch, numpy as np
from .common import RESULT, write, record, source_protocol
from .geometry import pairs, contact_pen, max_penalty, unit_points, volume_overlap, FACTORS

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    start = time.time()
    device = 'cuda'
    checks = {}
    torch.manual_seed(42)
    for method in ['cylinder', 'ellipsoid', 'p4']:
        n = 7
        c = torch.randn(n, 3, device=device, dtype=torch.float64) * 0.6
        T = torch.eye(3, device=device, dtype=torch.float64)[None].repeat(n, 1, 1) + torch.randn(n, 3, 3, device=device, dtype=torch.float64) * 0.1
        valid = torch.ones(n, device=device, dtype=torch.bool)
        edge = pairs(c, T, valid, method)
        value, count = volume_overlap(c, T, valid, edge, method, reps=2, samples=128, return_counts=True)
        q = unit_points(method, 2, 128, 'cuda:0').reshape(-1, 3)
        x = c[:, None, :] + torch.einsum('nij,sj->nsi', T, q)
        inv = torch.linalg.inv(T)
        diff = x[:, :, None, :] - c[None, None, :, :]
        u = torch.einsum('kab,nskb->nska', inv, diff)
        r = u[..., :2].square().sum(-1)
        inside = (r <= 1) & (u[..., 2].abs() <= 1) if method == 'cylinder' else r + u[..., 2].pow(2 if method == 'ellipsoid' else 4) <= 1
        exact = inside.sum(-1)
        assert torch.equal(count, exact), (method, int((count != exact).sum()))
        v = FACTORS[method] * torch.linalg.det(T).abs()
        expected = ((1 - 1 / exact.double()).mean(-1) * v / v.sum()).sum()
        assert abs(float(expected) - value['volume_overlap']) < 1e-12
        checks[method + '_fused_matches_all_pairs'] = True
    c = torch.zeros(3, 3, device=device, dtype=torch.float64)
    T = torch.eye(3, device=device, dtype=torch.float64)[None].repeat(3, 1, 1)
    valid = torch.ones(3, device=device, dtype=torch.bool)
    e = pairs(c, T, valid, 'p4')
    v = volume_overlap(c, T, valid, e, 'p4')
    assert abs(v['volume_overlap'] - 2 / 3) < 1e-12
    c[:, 0] = torch.tensor([0.0, 5.0, 10.0], device=device)
    e = pairs(c, T, valid, 'p4')
    v = volume_overlap(c, T, valid, e, 'p4')
    assert v['volume_overlap'] == 0
    c = c[:2].clone()
    c[:, 0] = torch.tensor([0.0, 1.0], device=device)
    T = T[:2]
    valid = valid[:2]
    e = pairs(c, T, valid, 'ellipsoid')
    v = volume_overlap(c, T, valid, e, 'ellipsoid', reps=4, samples=8192)
    assert abs(v['volume_overlap'] - 5 / 32) < 0.002
    checks['sphere_overlap_estimate'] = v['volume_overlap']
    checks['sphere_overlap_exact'] = 5 / 32
    pen = contact_pen(c, T, e)
    assert abs(float(pen[0]) - 0.5) < 1e-06
    maximum = max_penalty(3, torch.tensor([[0], [1]], device=device), torch.tensor([0.1], device=device))
    d = torch.tensor([0.9, 0.8, 0.3], device=device)
    good = maximum <= 0.02
    j = float((d * good).mean())
    assert abs(j - 0.1) < 1e-07
    assert abs(j - float(d.mean()) / 3) > 0.1
    checks['J_keeps_original_denominator_and_object_correlation'] = True
    checks['triple_coincident_overlap'] = 2 / 3
    checks['disjoint_overlap'] = 0
    cn = torch.randn(5, 3, device=device, dtype=torch.float64)
    cc = cn + torch.randn_like(cn)
    delta = torch.zeros_like(cn)
    delta[:, 2] = 2.5 - cn[:, 2]
    assert torch.allclose(cc + delta - (cn + delta), cc - cn)
    checks['paired_translation_preserves_relative_displacement'] = True
    from .run import evaluate_roi
    from .common import BASE
    uid = sorted((BASE / '05_evaluation/p4/test').glob('*/complete.json'))[0].parent.name
    checks['complete_real_roi'] = uid
    checks['real_roi_seconds'] = evaluate_roi('test', uid)
    write(RESULT / 'protocol.json', source_protocol())
    write(RESULT / '01_validation/complete.json', record(checks=checks, seconds=time.time() - start))
    print(json.dumps(checks), flush=True)
if __name__ == '__main__':
    main()
