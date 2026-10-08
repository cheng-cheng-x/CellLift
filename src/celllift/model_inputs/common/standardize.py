from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import os
from celllift.runtime import ResourcePath as Path
from typing import Any
from .utils import atomic_parquet, atomic_text, read_parquet_rows, sha256_file, stable_shard

def _validated_existing(path: Path, expected_size: tuple[int, int]) -> str | None:
    from PIL import Image
    sidecar = path.with_suffix('.sha256')
    if not path.is_file() or not sidecar.is_file():
        return None
    digest = sha256_file(path)
    if digest != sidecar.read_text(encoding='utf-8').strip():
        raise RuntimeError(f'existing standardized image hash mismatch: {path}')
    with Image.open(path) as image:
        image.load()
        if image.mode != 'RGB' or image.size != expected_size:
            raise RuntimeError(f'existing standardized image schema mismatch: {path}')
    return digest

def _write_rgb(source: Path, destination: Path, size: tuple[int, int]) -> tuple[str, bool]:
    from PIL import Image
    existing = _validated_existing(destination, size)
    if existing is not None:
        return (existing, True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        rgb = image.convert('RGB')
        if rgb.size != (512, 512):
            raise RuntimeError(f'source patch is not 512x512: {source} {rgb.size}')
        output = rgb.resize(size, Image.Resampling.LANCZOS)
    temporary = destination.with_name(f'.{destination.stem}.tmp.{os.getpid()}.png')
    output.save(temporary, format='PNG', compress_level=1)
    with Image.open(temporary) as check:
        check.load()
        if check.mode != 'RGB' or check.size != size:
            raise RuntimeError(f'temporary standardized image failed validation: {temporary}')
    os.replace(temporary, destination)
    digest = sha256_file(destination)
    atomic_text(destination.with_suffix('.sha256'), digest + '\n')
    return (digest, False)

def _decode_sicap_mask(source: Path, destination: Path, expected_label_id: int) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    with Image.open(source) as image:
        gray = np.asarray(image.convert('L'), dtype=np.int16)
    palette = np.asarray([0, 50, 100, 150, 200], dtype=np.int16)
    semantic = np.abs(gray[..., None] - palette).argmin(axis=2).astype(np.uint8)
    residual = np.min(np.abs(gray[..., None] - palette), axis=2)
    non_background = semantic[semantic > 0]
    majority = int(np.bincount(non_background, minlength=5).argmax()) if len(non_background) else 0
    allowed_majorities = {0: {0}, 1: {1}, 2: {2, 3}, 3: {4}}
    expected_semantic = sorted(allowed_majorities[int(expected_label_id)])
    majority_matches = majority in allowed_majorities[int(expected_label_id)]
    resized = Image.fromarray(semantic, mode='L').resize((1024, 1024), Image.Resampling.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.stem}.tmp.{os.getpid()}.png')
    resized.save(temporary, format='PNG', compress_level=1)
    os.replace(temporary, destination)
    return {'semantic_mask_path': str(destination), 'semantic_mask_sha256': sha256_file(destination), 'semantic_majority_id': majority, 'semantic_expected_ids': json.dumps(expected_semantic), 'semantic_majority_matches_patch_label': majority_matches, 'semantic_aux_valid': bool(float(np.quantile(residual, 0.999)) <= 25.0), 'palette_residual_p999': float(np.quantile(residual, 0.999))}

def run_standardize(cfg: dict[str, Any], dataset: str, shard_id: int, num_shards: int, *, pilot: bool=False) -> dict[str, Any]:
    data_root = Path(cfg['paths']['data_root'])
    manifest_name = 'pilot_manifest.parquet' if pilot else 'patch_manifest.parquet'
    rows = read_parquet_rows(data_root / '00_manifest' / manifest_name)
    rows = [row for row in rows if stable_shard(str(row['patch_id']), num_shards) == shard_id]
    statuses: list[dict[str, Any]] = []
    for row in rows:
        patch_id = str(row['patch_id'])
        status = {'patch_id': patch_id, 'status': 'failed', 'failure_reason': ''}
        try:
            source = Path(row['source_path'])
            if sha256_file(source) != row['source_sha256']:
                raise RuntimeError('source SHA-256 changed after inventory')
            size = (int(row['output_width_px']), int(row['output_height_px']))
            digest, resumed = _write_rgb(source, Path(row['rgb_path']), size)
            status.update({'status': 'complete', 'rgb_sha256': digest, 'resumed': resumed})
            if dataset == 'sicapv2':
                semantic_path = data_root / '04_labels_splits/semantic_masks' / Path(row['rgb_path']).parent.name / f'{patch_id}.png'
                status.update(_decode_sicap_mask(Path(row['source_mask_path']), semantic_path, int(row['label_id'])))
        except Exception as exc:
            status['failure_reason'] = f'{type(exc).__name__}: {exc}'
        statuses.append(status)
    tag = 'pilot' if pilot else 'full'
    output = data_root / '00_manifest' / f'standardize_{tag}_shard_{shard_id:03d}_of_{num_shards:03d}.parquet'
    atomic_parquet(output, statuses)
    failed = [row for row in statuses if row['status'] != 'complete']
    if failed:
        raise RuntimeError(f'standardization failed for {len(failed)}/{len(statuses)} patches; see {output}')
    return {'status': 'PASS', 'processed': len(statuses), 'output': str(output)}
