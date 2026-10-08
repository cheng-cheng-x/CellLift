from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import sys
from celllift.runtime import ResourcePath as Path
import numpy as np
from celllift.runtime import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
from celllift.matched_geometry_controls.residual import _fit

class TinyProbe(torch.nn.Module):

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.local = torch.nn.Linear(36, 32)
        self.context = torch.nn.Linear(73, 32)
        self.head = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 9))

    def forward(self, rays, context):
        return self.head(torch.cat((self.local(rays), self.context(context)), dim=-1))

def full_loss(model, rays, context, target, valid_ncr, ids):
    with torch.inference_mode():
        pred = model(torch.as_tensor(rays[ids]), torch.as_tensor(context[ids]))
        y = torch.as_tensor(target[ids])
        mask = torch.ones_like(y)
        mask[:, 8] = torch.as_tensor(valid_ncr[ids], dtype=y.dtype)
        return float(((pred - y).square() * mask).sum() / mask.sum().clamp_min(1))

def main() -> int:
    rng = np.random.default_rng(20260909)
    anchors = 20003
    rays = rng.standard_normal((anchors, 36)).astype(np.float32)
    context = rng.standard_normal((anchors, 73)).astype(np.float32)
    target = rng.standard_normal((anchors, 9)).astype(np.float32)
    valid_ncr = rng.random(anchors) > 0.2
    train = np.ones(anchors, bool)
    validation = np.zeros(anchors, bool)
    validation[::3] = True
    model, epochs, history = _fit(TinyProbe, rays, context, target, valid_ncr, train, validation, seed=17, device='cpu', fixed_epochs=2, batch_size=4096)
    val_ids = np.flatnonzero(validation)
    reference = full_loss(model, rays, context, target, valid_ncr, val_ids)
    chunked = history[-1]
    if not np.isfinite(chunked) or not np.isfinite(reference):
        raise RuntimeError(f'non-finite loss: chunked={chunked} full={reference}')
    delta = abs(chunked - reference)
    tolerance = 0.0001 * max(1.0, abs(reference))
    print(f'epochs={epochs} history={[round(value, 6) for value in history]}')
    print(f'chunked={chunked:.9f} full={reference:.9f} delta={delta:.3e} tolerance={tolerance:.3e}')
    if delta > tolerance:
        raise RuntimeError('chunked validation loss diverges from full-set loss')
    print('PASS')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
