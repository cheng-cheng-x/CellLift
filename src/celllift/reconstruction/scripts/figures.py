from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import numpy as np
from celllift.runtime import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from celllift.reconstruction.src.common import RESULT, atomic_json, record

def main():
    rows = sorted((RESULT / '04_predictions/val').glob('*.pt'))
    chosen = [rows[j] for j in (0, len(rows) // 2, len(rows) - 1)]
    theta = np.linspace(0, 2 * np.pi, 33)
    t = np.linspace(-1, 1, 25)
    tt, th = np.meshgrid(t, theta)
    rr = np.sqrt(np.maximum(0, 1 - tt ** 4))
    unit = np.stack((rr * np.cos(th), rr * np.sin(th), tt), -1)
    for path in chosen:
        x = torch.load(path, weights_only=False, map_location='cpu')
        lab = x['labels']['main']
        ids = torch.arange(len(lab)) * 9 + lab
        valid = x['tables'].valid.numpy()
        c = x['geometry'].cell
        nc = x['geometry'].nucleus
        center = c.center[ids].numpy()
        T = c.transform[ids].numpy()
        ncenter = nc.center[ids].numpy()
        NT = nc.transform[ids].numpy()
        allids = np.flatnonzero(valid)
        med = np.median(center[valid, :2], axis=0)
        select = allids[np.argsort(((center[allids, :2] - med) ** 2).sum(1))[:24]]
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(121, projection='3d')
        bx = fig.add_subplot(122)
        points = []
        for node in select:
            color = plt.cm.tab20(int(node) % 20)
            for cc, mat, opacity in ((center[node], T[node], 0.18), (ncenter[node], NT[node], 0.5)):
                surface = np.einsum('ij,uvj->uvi', mat, unit) + cc
                points.append(surface.reshape(-1, 3))
                ax.plot_surface(*surface.transpose(2, 0, 1), color=color, alpha=opacity, linewidth=0, rstride=2, cstride=2)
            bx.scatter(ncenter[node, 0], ncenter[node, 2], color=color, s=20)
        points = np.concatenate(points)
        lo = points.min(0)
        hi = points.max(0)
        extent = hi - lo
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(extent)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_zlabel('Z (um)')
        bx.axhspan(0, 5, alpha=0.12)
        bx.set_xlabel('X (um)')
        bx.set_ylabel('Nucleus center Z (um)')
        bx.set_aspect('equal', adjustable='datalim')
        fig.suptitle(path.stem + ' | input-only, nearest 24 cells')
        fig.tight_layout()
        fig.savefig(RESULT / '06_figures' / (path.stem + '.png'), dpi=170)
        plt.close(fig)
    atomic_json(RESULT / 'runtime/figures_done.json', record(graphs=[p.stem for p in chosen]))
if __name__ == '__main__':
    main()
