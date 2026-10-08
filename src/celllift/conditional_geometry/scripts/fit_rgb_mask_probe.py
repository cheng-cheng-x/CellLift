from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any, Mapping
from celllift.conditional_geometry.common.rgb_mask_probe_data import RGBMaskFoldProbeData, collect_rgb_mask_probe_sample, load_rgb_mask_prepared_fold, save_rgb_mask_prepared_fold
from celllift.conditional_geometry.models.rgb_mask_probe import RGBMask3DProbe
from celllift.conditional_geometry.rgb_mask_protocol import RGB_MASK_PROTOCOL_ID, PROBE_SEED
from celllift.conditional_geometry.scripts import fit_mask_probe as implementation

def fit_rgb_mask_fold_probe(cfg: Mapping[str, Any], fold: int, *, device: str='cuda', graph_limit: int | None=None, sample_limit: int | None=None) -> dict[str, Any]:
    if str(cfg.get('protocol_id')) != RGB_MASK_PROTOCOL_ID:
        raise RuntimeError('RGB+mask config protocol mismatch')
    implementation.MaskFoldProbeData = RGBMaskFoldProbeData
    implementation.collect_mask_probe_sample = collect_rgb_mask_probe_sample
    implementation.load_mask_prepared_fold = load_rgb_mask_prepared_fold
    implementation.save_mask_prepared_fold = save_rgb_mask_prepared_fold
    implementation.MaskOnly3DProbe = RGBMask3DProbe
    implementation.MASK_PROTOCOL_ID = RGB_MASK_PROTOCOL_ID
    implementation.PROBE_SEED = PROBE_SEED
    return implementation.fit_mask_fold_probe(cfg, fold, device=device, graph_limit=graph_limit, sample_limit=sample_limit)
