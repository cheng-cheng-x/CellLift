from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
from dataclasses import asdict, dataclass, fields
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
import numpy as np
from celllift.conditional_geometry.common.mask_probe_data import MaskFoldProbeData, MaskProbeStatistics, collect_mask_probe_sample
from celllift.conditional_geometry.common.probe_data import RunningMoments
from celllift.conditional_geometry.mask_protocol import MASK_CONTEXT_DIM
from celllift.conditional_geometry.rgb_mask_protocol import RGB_DIM, RGB_MASK_CONTEXT_DIM, RGB_MASK_PROTOCOL_ID

@dataclass(frozen=True)
class RGBMaskProbeStatistics(MaskProbeStatistics):
    rgb_mean: tuple[float, ...]
    rgb_std: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'RGBMaskProbeStatistics':
        names = {field.name for field in fields(cls)}
        return cls(**{key: tuple(item) if isinstance(item, list) else item for key, item in value.items() if key in names})

class RGBMaskFoldProbeData(MaskFoldProbeData):
    statistics: RGBMaskProbeStatistics | None

    def __init__(self, cfg: Mapping[str, Any], fold: int, *, graph_limit: int | None=None) -> None:
        if str(cfg.get('probe', {}).get('conditioning_mode')) != 'rgb_plus_mask_rays':
            raise RuntimeError('RGB+mask rgb_mask_conditioning requires conditioning_mode=rgb_plus_mask_rays')
        loader_cfg = dict(cfg)
        loader_cfg['probe'] = dict(cfg['probe'])
        loader_cfg['probe']['conditioning_mode'] = 'mask_rays_only'
        super().__init__(loader_cfg, fold, graph_limit=graph_limit)
        self.cfg = dict(cfg)

    def prepare(self) -> RGBMaskProbeStatistics:
        base = super().prepare()
        moments = RunningMoments(RGB_DIM)
        moments.update(self.rgb_raw[self.roles == 'train'])
        rgb_mean, rgb_std = moments.finish()
        self.statistics = RGBMaskProbeStatistics(**base.as_dict(), rgb_mean=tuple(map(float, rgb_mean)), rgb_std=tuple(map(float, rgb_std)))
        return self.statistics

    def set_statistics(self, statistics: RGBMaskProbeStatistics, context_raw: np.ndarray) -> None:
        self.statistics = statistics
        self.context_raw = np.asarray(context_raw, dtype=np.float32)

    def transformed_context(self) -> np.ndarray:
        if self.statistics is None or self.context_raw is None:
            raise RuntimeError('prepare must run before RGB+mask context access')
        stats = self.statistics
        rgb = (self.rgb_raw - np.asarray(stats.rgb_mean)) / np.asarray(stats.rgb_std)
        mask = (self.context_raw - np.asarray(stats.context_mean)) / np.asarray(stats.context_std)
        result = np.concatenate((rgb, mask), axis=1).astype(np.float32)
        if result.shape != (len(self.graph_ids), RGB_MASK_CONTEXT_DIM) or not np.isfinite(result).all():
            raise RuntimeError('invalid RGB+mask transformed context')
        return result

def save_rgb_mask_prepared_fold(data: RGBMaskFoldProbeData, root: Path) -> None:
    if data.statistics is None or data.context_raw is None:
        raise RuntimeError('RGB+mask fold data is not prepared')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'statistics.json').write_text(json.dumps(data.statistics.as_dict(), indent=2, sort_keys=True) + '\n', encoding='utf-8')
    np.save(root / 'context_raw.npy', data.context_raw, allow_pickle=False)
    (root / 'manifest.json').write_text(json.dumps({'status': 'PASS', 'protocol_id': RGB_MASK_PROTOCOL_ID, 'fold': data.fold, 'conditioning': 'fold-specific frozen RGB512 plus raw nucleus-mask rays36 and ray-derived summaries', 'graph_ids': list(data.graph_ids), 'roles': list(map(str, data.roles)), 'official_test_touched': False}, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def load_rgb_mask_prepared_fold(data: RGBMaskFoldProbeData, root: Path) -> None:
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'PASS' or manifest.get('protocol_id') != RGB_MASK_PROTOCOL_ID or int(manifest.get('fold', -1)) != data.fold or (tuple(map(str, manifest.get('graph_ids', ()))) != data.graph_ids) or (tuple(map(str, manifest.get('roles', ()))) != tuple(map(str, data.roles))) or (manifest.get('official_test_touched') is not False):
        raise RuntimeError(f'RGB+mask prepared fold provenance mismatch: {root}')
    stats = RGBMaskProbeStatistics.from_dict(json.loads((root / 'statistics.json').read_text(encoding='utf-8')))
    context = np.load(root / 'context_raw.npy', allow_pickle=False)
    if context.shape != (len(data.graph_ids), MASK_CONTEXT_DIM) or not np.isfinite(context).all():
        raise RuntimeError(f'invalid RGB+mask prepared mask context: {root}')
    data.set_statistics(stats, context)
collect_rgb_mask_probe_sample = collect_mask_probe_sample
