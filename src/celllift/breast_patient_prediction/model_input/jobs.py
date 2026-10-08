from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from collections import defaultdict
from celllift.runtime import ResourcePath as Path
from typing import Any, Iterable
from .. import paths
from ..protocol import IMAGE_SIZE, PILOT_PATIENTS, PILOT_SEED, PILOT_TILES_PER_PATIENT, SUBTYPE4, TARGET_MPP
from . import layout

def load_tile_manifest(data_root: Path | None=None):
    import pandas as pd
    path = paths.tile_dir(data_root) / 'tile_manifest.parquet'
    return pd.read_parquet(path)

def load_patient_labels(data_root: Path | None=None):
    import pandas as pd
    return pd.read_parquet(paths.label_dir(data_root) / 'patient_labels.parquet')

def load_slide_inventory(data_root: Path | None=None) -> dict[str, dict[str, Any]]:
    import pandas as pd
    path = paths.inventory_dir(data_root) / 'eligible_slides.parquet'
    if not path.is_file():
        return {}
    rows = pd.read_parquet(path).to_dict('records')
    return {str(row['slide_id']): row for row in rows}

def slide_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['slide_id'])].append(dict(row))
    for slide_id in grouped:
        grouped[slide_id].sort(key=lambda item: str(item['tile_id']))
    return dict(sorted(grouped.items()))

def enrich_tile(row: dict[str, Any], inventory: dict[str, dict[str, Any]], base: Path) -> dict[str, Any]:
    out = dict(row)
    slide = inventory.get(str(row['slide_id']), {})
    out['mpp_x'] = float(slide.get('mpp_x', row.get('native_mpp', TARGET_MPP)))
    out['mpp_y'] = float(slide.get('mpp_y', out['mpp_x']))
    out['wsi_source_path'] = str(row.get('source_path') or slide.get('source_path') or '')
    out['rgb_path'] = str(layout.rgb_path(base, str(row['slide_id']), str(row['tile_id'])))
    out['target_size'] = int(row.get('target_size') or IMAGE_SIZE)
    out['target_mpp'] = float(row.get('target_mpp') or TARGET_MPP)
    out['graph_id'] = str(row.get('graph_id') or f"tcga_brca_set_encoding:{row['tile_id']}")
    return out

def select_pilot_tiles(manifest, labels, *, seed: int=PILOT_SEED, n_patients: int=PILOT_PATIENTS, tiles_per: int=PILOT_TILES_PER_PATIENT):
    import numpy as np
    fit_ids = set(manifest.loc[manifest['split'].astype(str).str.lower() == 'fit', 'patient_id'].astype(str))
    labeled = labels[labels['patient_id'].astype(str).isin(fit_ids)].copy()
    rng = np.random.default_rng(seed)
    chosen_patients: list[str] = []
    quota = n_patients // max(1, len(SUBTYPE4))
    remainder = n_patients - quota * len(SUBTYPE4)
    for index, subtype in enumerate(SUBTYPE4):
        take = quota + int(index < remainder)
        pool = labeled.loc[labeled['subtype4'] == subtype, 'patient_id'].astype(str).tolist()
        pool = sorted({pid for pid in pool if int((manifest.patient_id.astype(str) == pid).sum()) >= tiles_per})
        rng.shuffle(pool)
        for pid in pool:
            if pid not in chosen_patients and take > 0:
                chosen_patients.append(pid)
                take -= 1
        if take > 0:
            extras = [pid for pid in sorted(fit_ids) if pid not in chosen_patients]
            rng.shuffle(extras)
            chosen_patients.extend(extras[:take])
    if len(chosen_patients) < n_patients:
        extras = [pid for pid in sorted(fit_ids) if pid not in chosen_patients]
        rng.shuffle(extras)
        chosen_patients.extend(extras[:n_patients - len(chosen_patients)])
    chosen_patients = chosen_patients[:n_patients]
    tiles = []
    for pid in chosen_patients:
        part = manifest.loc[manifest['patient_id'].astype(str) == pid].sort_values(['tissue_frac', 'tile_id'])
        if len(part) == 0:
            continue
        if len(part) == 1:
            tiles.extend(part.to_dict('records'))
            continue
        tiles.append(part.iloc[0].to_dict())
        tiles.append(part.iloc[-1].to_dict())
        if tiles_per > 2:
            mid = part.iloc[len(part) // 2].to_dict()
            if mid['tile_id'] not in {tiles[-1]['tile_id'], tiles[-2]['tile_id']}:
                tiles.append(mid)
    return tiles[:n_patients * tiles_per]
