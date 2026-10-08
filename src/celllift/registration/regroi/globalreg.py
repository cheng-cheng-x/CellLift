from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import json
import math
from concurrent.futures import ThreadPoolExecutor
from celllift.runtime import ResourcePath as Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from .core import read_tsv, result_root, stage_status, write_json, write_tsv

def dice(left: np.ndarray, right: np.ndarray) -> float:
    return float(2 * np.logical_and(left, right).sum() / max(1, left.sum() + right.sum()))

def warp_mask(mask: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.warpPerspective(mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0) > 0

def _euclidean(theta: float, tx: float, ty: float) -> np.ndarray:
    c, s = (math.cos(theta), math.sin(theta))
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], float)

def _parameters(matrix: np.ndarray) -> np.ndarray:
    return np.array([math.atan2(matrix[1, 0], matrix[0, 0]), matrix[0, 2], matrix[1, 2]])

def _ecc(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    left_float = left.astype(np.float32)
    right_float = right.astype(np.float32)
    lm, rm = (cv2.moments(left_float), cv2.moments(right_float))
    lcx = lm['m10'] / max(lm['m00'], 1)
    lcy = lm['m01'] / max(lm['m00'], 1)
    rcx = rm['m10'] / max(rm['m00'], 1)
    rcy = rm['m01'] / max(rm['m00'], 1)
    backward = np.array([[1, 0, rcx - lcx], [0, 1, rcy - lcy]], np.float32)
    score, backward = cv2.findTransformECC(left_float, right_float, backward, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 600, 1e-07), None, 5)
    return (np.linalg.inv(np.vstack([backward, [0, 0, 1]])).astype(float), float(score))

def _sift_affine(left_rgb: np.ndarray, right_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    left = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY))
    right = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY))
    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.01)
    kl, dl = detector.detectAndCompute(left, None)
    kr, dr = detector.detectAndCompute(right, None)
    if dl is None or dr is None:
        raise RuntimeError('SIFT descriptors missing')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = [pair[0] for pair in matcher.knnMatch(dr, dl, k=2) if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
    if len(matches) < 12:
        raise RuntimeError(f'only {len(matches)} SIFT matches')
    source = np.float32([kr[item.queryIdx].pt for item in matches])
    target = np.float32([kl[item.trainIdx].pt for item in matches])
    affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=20000, confidence=0.999, refineIters=50)
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        raise RuntimeError('SIFT-RANSAC affine failed')
    matrix = np.vstack([affine, [0, 0, 1]]).astype(float)
    return (matrix, {'sift_matches': len(matches), 'sift_inliers': int(inliers.sum())})

def _positive_dense(left: np.ndarray, right_aligned: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setVariationalRefinementIterations(10)
    raw = dis.calc(left.astype(np.uint8) * 255, right_aligned.astype(np.uint8) * 255, None)
    yy, xx = np.indices(left.shape, dtype=np.float32)
    rows = []
    for sigma in (0, 0.5, 1, 2, 4, 8, 16, 32, 64):
        field = raw.copy() if sigma == 0 else np.stack([gaussian_filter(raw[..., 0], sigma), gaussian_filter(raw[..., 1], sigma)], axis=-1).astype(np.float32)
        dx, dy = (field[..., 0], field[..., 1])
        dxy, dxx = np.gradient(dx)
        dyy, dyx = np.gradient(dy)
        jacobian = (1 + dxx) * (1 + dyy) - dxy * dyx
        map_x, map_y = (xx + dx, yy + dy)
        warped = cv2.remap(right_aligned.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        support = float(((map_x >= 0) & (map_x < left.shape[1]) & (map_y >= 0) & (map_y < left.shape[0]))[left].mean())
        post = dice(left, warped)
        row = {'sigma': sigma, 'post_dice': post, 'support_fraction': support, 'jacobian_min': float(jacobian.min()), 'jacobian_p1': float(np.percentile(jacobian, 1)), 'jacobian_p99': float(np.percentile(jacobian, 99))}
        rows.append(row)
        if row['jacobian_min'] > 0 and support >= float(cfg['global']['dense_min_support']) and (post >= float(cfg['global']['dice_pass']) or post >= float(cfg['global']['dice_recovery_floor'])):
            return ({**row, 'candidates': rows}, field)
    return ({'candidates': rows, 'post_dice': 0.0}, None)

def estimate_edge(left_rgb: np.ndarray, right_rgb: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, cfg: dict[str, Any], allow_dense: bool=True) -> dict[str, Any]:
    shape = left_mask.shape
    pre = dice(left_mask, right_mask)
    attempts = []
    candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    try:
        matrix, score = _ecc(left_mask, right_mask)
        candidates.append(('ecc_euclidean', matrix, {'ecc_score': score}))
    except cv2.error as error:
        attempts.append(f'ecc:{error}')
    try:
        matrix, metrics = _sift_affine(left_rgb, right_rgb)
        candidates.append(('sift_ransac_partial_affine', matrix, metrics))
    except RuntimeError as error:
        attempts.append(f'sift:{error}')
    if not candidates:
        candidates.append(('identity_terminal_fallback', np.eye(3), {}))
    scored = []
    for method, matrix, metrics in candidates:
        determinant = float(np.linalg.det(matrix[:2, :2]))
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        post = dice(left_mask, warp_mask(right_mask, matrix, shape))
        valid = determinant > 0 and singular.min() >= 0.85 and (singular.max() <= 1.18)
        scored.append((post if valid else -1, method, matrix, metrics, determinant, singular))
    post, method, matrix, metrics, determinant, singular = max(scored, key=lambda item: item[0])
    improvement = post - pre
    passed = post >= float(cfg['global']['dice_pass']) or (post >= float(cfg['global']['dice_recovery_floor']) and improvement >= float(cfg['global']['dice_min_improvement']))
    dense_metrics: dict[str, Any] = {}
    dense_field = None
    if not passed and allow_dense:
        aligned = warp_mask(right_mask, matrix, shape)
        dense_metrics, dense_field = _positive_dense(left_mask, aligned, cfg)
        if dense_field is not None:
            dense_post = float(dense_metrics['post_dice'])
            dense_improvement = dense_post - pre
            if dense_post >= float(cfg['global']['dice_pass']) or (dense_post >= float(cfg['global']['dice_recovery_floor']) and dense_improvement >= float(cfg['global']['dice_min_improvement'])):
                post, improvement, passed = (dense_post, dense_improvement, True)
    return {'status': 'pass' if passed else 'terminal_fail', 'method': method, 'pre_dice': pre, 'post_dice': post, 'improvement': improvement, 'determinant': determinant, 'singular_min': float(singular.min()), 'singular_max': float(singular.max()), 'forward_matrix': matrix, 'backward_matrix': np.linalg.inv(matrix), 'dense_field': dense_field, 'dense_metrics': dense_metrics, 'attempt_notes': ';'.join(attempts), **metrics}

def _sequential(pair_matrices: list[np.ndarray], reference_index: int) -> list[np.ndarray]:
    output = [np.eye(3) for _ in range(len(pair_matrices) + 1)]
    for index in range(reference_index + 1, len(output)):
        output[index] = output[index - 1] @ pair_matrices[index - 1]
    for index in range(reference_index - 1, -1, -1):
        output[index] = output[index + 1] @ np.linalg.inv(pair_matrices[index])
    return output

def optimize_pose_graph(pair_matrices: list[np.ndarray], skip_matrices: dict[int, np.ndarray], direct_matrices: list[np.ndarray | None], reference_index: int, shape: tuple[int, int]) -> tuple[list[np.ndarray], dict[str, Any]]:
    count = len(direct_matrices)
    variables = [index for index in range(count) if index != reference_index]
    slots = {slide: slot for slot, slide in enumerate(variables)}
    height, width = shape
    control = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1], [width / 2, height / 2, 1]], float).T
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add_coefficient(row: int, slide: int, axis: int, vector: np.ndarray, sign: float) -> float:
        if slide == reference_index:
            return sign * float(vector[axis])
        base = slots[slide] * 6 + axis * 3
        for offset in range(3):
            row_indices.append(row)
            col_indices.append(base + offset)
            values.append(sign * float(vector[offset]))
        return 0.0

    def add_relative(left: int, right: int, edge: np.ndarray, weight: float) -> None:
        for point in range(control.shape[1]):
            moving = control[:, point]
            fixed = edge @ moving
            for axis in range(2):
                row = len(rhs)
                constant = add_coefficient(row, left, axis, fixed, weight)
                constant += add_coefficient(row, right, axis, moving, -weight)
                rhs.append(-constant)
    for index, edge in enumerate(pair_matrices):
        add_relative(index, index + 1, edge, 1.0)
    for index, edge in skip_matrices.items():
        add_relative(index, index + 2, edge, 0.5)
    for index, direct in enumerate(direct_matrices):
        if index == reference_index or direct is None:
            continue
        weight = 0.25 / (1 + abs(index - reference_index) / 20)
        for point in range(control.shape[1]):
            source = control[:, point]
            target = direct @ source
            for axis in range(2):
                row = len(rhs)
                add_coefficient(row, index, axis, source, weight)
                rhs.append(weight * float(target[axis]))
    matrix = coo_matrix((values, (row_indices, col_indices)), shape=(len(rhs), len(variables) * 6)).tocsr()
    target = np.asarray(rhs, dtype=float)
    huber_delta = 25.0
    solution = lsqr(matrix, target, atol=1e-08, btol=1e-08, iter_lim=5000)
    vector = solution[0]
    iterations = int(solution[2])
    for _ in range(12):
        residual = matrix @ vector - target
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        outside = absolute > huber_delta
        robust[outside] = np.sqrt(huber_delta / absolute[outside])
        weighted = matrix.multiply(robust[:, None])
        updated = lsqr(weighted, target * robust, x0=vector, atol=1e-08, btol=1e-08, iter_lim=5000)
        iterations += int(updated[2])
        change = np.linalg.norm(updated[0] - vector) / max(1.0, np.linalg.norm(vector))
        vector = updated[0]
        if change < 1e-07:
            break
    poses = [np.eye(3) for _ in range(count)]
    for index in variables:
        block = vector[slots[index] * 6:slots[index] * 6 + 6]
        poses[index] = np.array([[block[0], block[1], block[2]], [block[3], block[4], block[5]], [0, 0, 1]], dtype=float)
    residual = matrix @ vector - target
    absolute = np.abs(residual)
    cost = np.where(absolute <= huber_delta, 0.5 * residual ** 2, huber_delta * (absolute - 0.5 * huber_delta)).sum()
    determinants = np.asarray([np.linalg.det(pose[:2, :2]) for pose in poses])
    finite = bool(np.isfinite(vector).all() and np.isfinite(residual).all())
    positive = bool((determinants > 0).all())
    return (poses, {'success': finite and positive, 'status': int(solution[1]), 'message': 'Huber IRLS sparse affine pose graph', 'cost': float(cost), 'optimality': float(np.max(np.abs(matrix.T @ np.clip(residual, -huber_delta, huber_delta)))), 'nfev': iterations, 'equation_count': int(matrix.shape[0]), 'variable_count': int(matrix.shape[1]), 'determinant_min': float(determinants.min()), 'determinant_max': float(determinants.max())})

def _load_inputs(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    root = result_root(cfg)
    image_root = root / '02_thumbnails_order_qc/thumbnails_10um'
    mask_root = root / '02_thumbnails_order_qc/masks_10um'
    images = [np.asarray(Image.open(image_root / f'{index:03d}.jpg').convert('RGB')) for index in range(1, 261)]
    masks = [np.asarray(Image.open(mask_root / f'{index:03d}.png')) > 0 for index in range(1, 261)]
    if len({image.shape for image in images}) != 1:
        raise RuntimeError('10 um thumbnails do not share one shape')
    return (images, masks)

def run_global_pilot(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    images, masks = _load_inputs(cfg)
    similarity = read_tsv(root / '02_thumbnails_order_qc/adjacent_similarity.tsv')
    weak_index = int(min(similarity, key=lambda row: float(row['similarity']))['edge_index']) - 1
    starts = [0, 114, 229, max(0, min(229, weak_index - 15))]
    names = ['first', 'center', 'last', 'weak']
    rows = []
    for name, start in zip(names, starts):
        for index in range(start, start + 30):
            result = estimate_edge(images[index], images[index + 1], masks[index], masks[index + 1], cfg)
            rows.append({'window': name, 'left_id': f'{index + 1:03d}', 'right_id': f'{index + 2:03d}', 'status': result['status'], 'method': result['method'], 'pre_dice': result['pre_dice'], 'post_dice': result['post_dice'], 'dense_selected': result['dense_field'] is not None})
    write_tsv(root / '03_global_pilots/pilot_edge_metrics.tsv', rows)
    failed = [row for row in rows if row['status'] != 'pass']
    write_json(root / '03_global_pilots/pilot_summary.json', {'status': 'PASS' if not failed else 'PASS_WITH_TERMINAL_FAILURES', 'edge_instances': len(rows), 'failed_instances': len(failed), 'failed_edges': [[row['left_id'], row['right_id']] for row in failed], 'automatic_fallback': 'ECC -> SIFT-RANSAC -> positive-Jacobian low-resolution dense'})
    stage_status(cfg, 'global_pilot', 'PASS' if not failed else 'PASS_WITH_TERMINAL_FAILURES', edge_instances=len(rows), failed_instances=len(failed))

def _load_cached_global_edges(cfg: dict[str, Any]) -> tuple[list[dict[str, str]], list[np.ndarray], list[dict[str, str]], dict[int, np.ndarray], list[dict[str, str]], list[np.ndarray | None]] | None:
    root = result_root(cfg)
    edge_path = root / '00_manifests/global_edge_manifest.tsv'
    skip_path = root / '04_global_registration/transforms/skip_one_manifest.tsv'
    direct_path = root / '04_global_registration/transforms/direct_reference_manifest.tsv'
    if not (edge_path.exists() and skip_path.exists() and direct_path.exists()):
        return None
    edge_rows = read_tsv(edge_path)
    skip_rows = read_tsv(skip_path)
    direct_rows = read_tsv(direct_path)
    if len(edge_rows) != 259 or len(skip_rows) != 258 or len(direct_rows) != 260:
        return None
    pair_matrices = [np.asarray(json.loads(row['forward_canvas10um_json']), dtype=float) for row in edge_rows]
    skip_matrices = {int(row['left_id']) - 1: np.asarray(json.loads(row['forward_canvas10um_json']), dtype=float) for row in skip_rows if row['status'] == 'pass'}
    direct_matrices: list[np.ndarray | None] = [None for _ in range(260)]
    for row in direct_rows:
        if row['status'] == 'pass':
            direct_matrices[int(row['moving_id']) - 1] = np.asarray(json.loads(row['forward_canvas10um_json']), dtype=float)
    return (edge_rows, pair_matrices, skip_rows, skip_matrices, direct_rows, direct_matrices)

def run_global_register(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    images, masks = _load_inputs(cfg)
    shape = masks[0].shape
    reference_index = int(cfg['global']['reference_section']) - 1
    cached = _load_cached_global_edges(cfg)
    if cached is not None:
        edge_rows, pair_matrices, _, skip_matrices, _, direct_matrices = cached
        return _finish_global_registration(cfg, images, masks, edge_rows, pair_matrices, skip_matrices, direct_matrices, reference_index)
    edge_rows, pair_matrices = ([], [])
    dense_root = root / '04_global_registration/transforms/global_dense'
    dense_root.mkdir(parents=True, exist_ok=True)
    workers = min(8, int(cfg.get('runtime', {}).get('global_cpu_workers', 8)))
    cv2.setNumThreads(1)

    def adjacent(index: int) -> tuple[int, dict[str, Any]]:
        return (index, estimate_edge(images[index], images[index + 1], masks[index], masks[index + 1], cfg))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        adjacent_results = list(pool.map(adjacent, range(259)))
    for index, result in adjacent_results:
        matrix = result.pop('forward_matrix')
        backward = result.pop('backward_matrix')
        field = result.pop('dense_field')
        dense_metrics = result.pop('dense_metrics')
        dense_path = ''
        if field is not None:
            path = dense_root / f'{index + 1:03d}__{index + 2:03d}.npz'
            np.savez_compressed(path, backward_flow_left_to_rigid_right=field, canvas_mpp_um=10.0)
            dense_path = str(path)
        pair_matrices.append(matrix)
        edge_rows.append({'edge_index': index + 1, 'left_id': f'{index + 1:03d}', 'right_id': f'{index + 2:03d}', 'forward_canvas10um_json': json.dumps(matrix.tolist(), separators=(',', ':')), 'backward_canvas10um_json': json.dumps(backward.tolist(), separators=(',', ':')), 'dense_field_path': dense_path, **result, **{f'dense_{key}': value for key, value in dense_metrics.items() if key != 'candidates'}})
    skip_matrices: dict[int, np.ndarray] = {}
    skip_rows = []

    def skip_one(index: int) -> tuple[int, dict[str, Any]]:
        return (index, estimate_edge(images[index], images[index + 2], masks[index], masks[index + 2], cfg, allow_dense=False))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        skip_results = list(pool.map(skip_one, range(258)))
    for index, result in skip_results:
        matrix = result['forward_matrix']
        if result['status'] == 'pass':
            skip_matrices[index] = matrix
        skip_rows.append({'left_id': f'{index + 1:03d}', 'right_id': f'{index + 3:03d}', 'status': result['status'], 'method': result['method'], 'post_dice': result['post_dice'], 'forward_canvas10um_json': json.dumps(matrix.tolist(), separators=(',', ':'))})

    def direct(index: int) -> tuple[int, dict[str, Any]]:
        if index == reference_index:
            return (index, {'status': 'pass', 'method': 'reference_identity', 'post_dice': 1.0, 'forward_matrix': np.eye(3)})
        result = estimate_edge(images[reference_index], images[index], masks[reference_index], masks[index], cfg, allow_dense=False)
        return (index, result)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        direct_results = list(pool.map(direct, range(260)))
    direct_matrices: list[np.ndarray | None] = [None for _ in range(260)]
    direct_rows = []
    for index, result in direct_results:
        matrix = result['forward_matrix']
        if result['status'] == 'pass':
            direct_matrices[index] = matrix
        direct_rows.append({'reference_id': f'{reference_index + 1:03d}', 'moving_id': f'{index + 1:03d}', 'status': result['status'], 'method': result['method'], 'post_dice': result['post_dice'], 'used_in_pose_graph': result['status'] == 'pass', 'forward_canvas10um_json': json.dumps(matrix.tolist(), separators=(',', ':'))})
    write_tsv(root / '00_manifests/global_edge_manifest.tsv', edge_rows)
    write_tsv(root / '04_global_registration/transforms/skip_one_manifest.tsv', skip_rows)
    write_tsv(root / '04_global_registration/transforms/direct_reference_manifest.tsv', direct_rows)
    return _finish_global_registration(cfg, images, masks, edge_rows, pair_matrices, skip_matrices, direct_matrices, reference_index)

def _finish_global_registration(cfg: dict[str, Any], images: list[np.ndarray], masks: list[np.ndarray], edge_rows: list[dict[str, Any]], pair_matrices: list[np.ndarray], skip_matrices: dict[int, np.ndarray], direct_matrices: list[np.ndarray | None], reference_index: int) -> None:
    root = result_root(cfg)
    shape = masks[0].shape
    poses, optimizer = optimize_pose_graph(pair_matrices, skip_matrices, direct_matrices, reference_index, shape)
    if not optimizer['success']:
        raise RuntimeError(f'global pose graph optimization failed: {optimizer}')
    mpp = float(cfg['physical']['mpp_um_per_px'])
    global_mpp = float(cfg['physical']['global_mpp_um_per_px'])
    scale = mpp / global_mpp
    source_level0_to_thumb = np.diag([scale, scale, 1.0])
    thumb_to_level0 = np.diag([1 / scale, 1 / scale, 1.0])
    preview_root = root / '04_global_registration/previews'
    preview_root.mkdir(parents=True, exist_ok=True)
    transform_rows = []
    for index, pose in enumerate(poses):
        preview = cv2.warpPerspective(images[index], pose, (shape[1], shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        preview_path = preview_root / f'{index + 1:03d}.jpg'
        Image.fromarray(preview).save(preview_path, quality=88)
        level0 = thumb_to_level0 @ pose @ source_level0_to_thumb
        transform_rows.append({'section_id': f'{index + 1:03d}', 'sequence_position': index + 1, 'reference_id': '130', 'source_to_global_canvas10um_json': json.dumps(pose.tolist(), separators=(',', ':')), 'global_canvas10um_to_source_json': json.dumps(np.linalg.inv(pose).tolist(), separators=(',', ':')), 'source_to_global_level0_json': json.dumps(level0.tolist(), separators=(',', ':')), 'global_level0_to_source_json': json.dumps(np.linalg.inv(level0).tolist(), separators=(',', ':')), 'determinant': float(np.linalg.det(level0[:2, :2])), 'is_reference': index == reference_index, 'preview_path': str(preview_path)})
    write_tsv(root / '00_manifests/global_transform_manifest.tsv', transform_rows)
    write_json(root / '04_global_registration/transforms/pose_graph_optimizer.json', optimizer)
    failed = [row for row in edge_rows if row['status'] != 'pass']
    write_json(root / '04_global_registration/global_registration_summary.json', {'status': 'PASS' if not failed else 'PASS_WITH_TERMINAL_FAILURES', 'slide_count': 260, 'edge_count': 259, 'failed_edge_count': len(failed), 'failed_edges': [[row['left_id'], row['right_id']] for row in failed], 'reference_identity': np.allclose(poses[reference_index], np.eye(3)), 'optimizer': optimizer})
    stage_status(cfg, 'global_register', 'PASS' if not failed else 'PASS_WITH_TERMINAL_FAILURES', slide_count=260, edge_count=259, failed_edges=len(failed))

def _panel(left: np.ndarray, right: np.ndarray, label: str) -> Image.Image:
    yy, xx = np.indices(left.shape[:2])
    choose = ((xx // 32 + yy // 32) % 2).astype(bool)
    checker = np.where(choose[..., None], left, right)
    lg = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    rg = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    redcyan = np.stack([lg, rg, rg], axis=-1)
    checker = cv2.resize(checker, (320, 240), interpolation=cv2.INTER_AREA)
    redcyan = cv2.resize(redcyan, (320, 240), interpolation=cv2.INTER_AREA)
    canvas = Image.new('RGB', (640, 266), 'white')
    canvas.paste(Image.fromarray(checker), (0, 0))
    canvas.paste(Image.fromarray(redcyan), (320, 0))
    ImageDraw.Draw(canvas).text((4, 244), label, fill='black')
    return canvas

def write_adjacent_flickers(previews: list[np.ndarray], edges: list[dict[str, Any]], qc_root: Path) -> list[dict[str, Any]]:
    cv2.setNumThreads(1)
    if len(previews) != len(edges) + 1:
        raise ValueError('adjacent flicker inputs must have one more preview than edges')
    output_root = qc_root / 'adjacent_flickers'
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, edge in enumerate(edges):
        frames = []
        for image in (previews[index], previews[index + 1]):
            resized = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(resized).convert('P'))
        path = output_root / f"{index + 1:03d}_{edge['left_id']}__{edge['right_id']}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=[600, 600], loop=0, optimize=False, disposal=2)
        rows.append({'edge_index': index + 1, 'left_id': edge['left_id'], 'right_id': edge['right_id'], 'path': str(path), 'width': 320, 'height': 240, 'frame_count': 2, 'duration_ms_per_frame': 600})
    write_tsv(qc_root / 'global_flicker_manifest.tsv', rows)
    return rows

def write_volume_qc(previews: list[np.ndarray], qc_root: Path) -> dict[str, str]:
    cv2.setNumThreads(1)
    if len(previews) != 260:
        raise ValueError(f'volume QC requires 260 previews, got {len(previews)}')
    height, width = previews[0].shape[:2]
    if any((image.shape[:2] != (height, width) for image in previews)):
        raise ValueError('registered previews have inconsistent shapes')
    weight = np.zeros((height, width), np.float32)
    weighted_z = np.zeros((height, width), np.float32)
    for index, image in enumerate(previews):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        structure = np.clip((248.0 - gray) / 248.0, 0, 1).astype(np.float32)
        weight += structure
        weighted_z += structure * (index / 259.0)
    mean_z = np.divide(weighted_z, np.maximum(weight, 1e-06), out=np.zeros_like(weighted_z), where=weight > 0)
    color = cv2.cvtColor(cv2.applyColorMap(np.rint(mean_z * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    positive = weight[weight > 0]
    normalizer = float(np.percentile(positive, 95)) if len(positive) else 1.0
    alpha = np.sqrt(np.clip(weight / max(normalizer, 1e-06), 0, 1))[..., None]
    z_projection = np.clip(color * alpha + 255 * (1 - alpha), 0, 255).astype(np.uint8)
    z_path = qc_root / 'z_color_projection.jpg'
    Image.fromarray(z_projection).save(z_path, quality=92)
    y_positions = [height // 4, height // 2, 3 * height // 4]
    xz_panels = []
    for y in y_positions:
        section = np.stack([image[y, :, :] for image in previews], axis=0)
        xz_panels.append(cv2.resize(section, (width, 520), interpolation=cv2.INTER_NEAREST))
    xz_canvas = Image.new('RGB', (width, len(xz_panels) * 542), 'white')
    xz_draw = ImageDraw.Draw(xz_canvas)
    for index, (y, panel) in enumerate(zip(y_positions, xz_panels)):
        top = index * 542
        xz_draw.text((4, top + 3), f'XZ source y={y}', fill='black')
        xz_canvas.paste(Image.fromarray(panel), (0, top + 22))
    xz_path = qc_root / 'virtual_xz_sections.jpg'
    xz_canvas.save(xz_path, quality=92)
    x_positions = [width // 4, width // 2, 3 * width // 4]
    yz_panels = []
    for x in x_positions:
        section = np.stack([image[:, x, :] for image in previews], axis=1)
        yz_panels.append(cv2.resize(section, (520, height), interpolation=cv2.INTER_NEAREST))
    yz_canvas = Image.new('RGB', (len(yz_panels) * 542, height + 22), 'white')
    yz_draw = ImageDraw.Draw(yz_canvas)
    for index, (x, panel) in enumerate(zip(x_positions, yz_panels)):
        left = index * 542
        yz_draw.text((left + 4, 3), f'YZ source x={x}', fill='black')
        yz_canvas.paste(Image.fromarray(panel), (left, 22))
    yz_path = qc_root / 'virtual_yz_sections.jpg'
    yz_canvas.save(yz_path, quality=92)
    return {'z_color_projection': str(z_path), 'virtual_xz_sections': str(xz_path), 'virtual_yz_sections': str(yz_path)}

def run_global_qc(cfg: dict[str, Any]) -> None:
    root = result_root(cfg)
    transforms = read_tsv(root / '00_manifests/global_transform_manifest.tsv')
    edges = read_tsv(root / '00_manifests/global_edge_manifest.tsv')
    if len(transforms) != 260 or len(edges) != 259:
        raise RuntimeError('global manifests are incomplete')
    previews = [np.asarray(Image.open(row['preview_path']).convert('RGB')) for row in transforms]
    mask_root = root / '02_thumbnails_order_qc/masks_10um'
    masks = []
    for index, row in enumerate(transforms):
        source = np.asarray(Image.open(mask_root / f'{index + 1:03d}.png')) > 0
        matrix = np.asarray(json.loads(row['source_to_global_canvas10um_json']), float)
        masks.append(warp_mask(source, matrix, source.shape))
    qc_root = root / '04_global_registration/qc'
    visual_root = qc_root / 'adjacent_visuals'
    visual_root.mkdir(parents=True, exist_ok=True)
    recomputed = []
    panels = []
    for index, edge in enumerate(edges):
        value = dice(masks[index], masks[index + 1])
        panel = _panel(previews[index], previews[index + 1], f"{edge['left_id']}->{edge['right_id']} pose Dice={value:.4f}")
        panel_path = visual_root / f"{index + 1:03d}_{edge['left_id']}__{edge['right_id']}.jpg"
        panel.save(panel_path, quality=88)
        panels.append(panel_path)
        recomputed.append({**edge, 'pose_graph_mask_dice': value, 'visual_path': str(panel_path)})
    write_tsv(qc_root / 'global_qc_edge_metrics.tsv', recomputed)
    flickers = write_adjacent_flickers(previews, edges, qc_root)
    contact_paths = []
    for page, start in enumerate(range(0, len(panels), 12), start=1):
        selected = [Image.open(path).convert('RGB') for path in panels[start:start + 12]]
        sheet = Image.new('RGB', (1280, math.ceil(len(selected) / 2) * 266), 'white')
        for offset, image in enumerate(selected):
            sheet.paste(image, (offset % 2 * 640, offset // 2 * 266))
        path = qc_root / f'adjacent_review_page_{page:02d}.jpg'
        sheet.save(path, quality=86)
        contact_paths.append(str(path))
    accumulator = np.zeros_like(previews[0], dtype=np.float64)
    for image in previews:
        accumulator += image
    Image.fromarray(np.clip(accumulator / len(previews), 0, 255).astype(np.uint8)).save(qc_root / 'average_stack.jpg', quality=92)
    volume_visuals = write_volume_qc(previews, qc_root)
    trajectory = []
    for row, mask in zip(transforms, masks):
        yy, xx = np.nonzero(mask)
        trajectory.append({'section_id': row['section_id'], 'tissue_area_px': int(mask.sum()), 'centroid_x': float(xx.mean()) if len(xx) else '', 'centroid_y': float(yy.mean()) if len(yy) else '', 'determinant': row['determinant']})
    write_tsv(qc_root / 'tissue_trajectory.tsv', trajectory)
    failed = [row for row in edges if row['status'] != 'pass']
    determinants = [float(row['determinant']) for row in transforms]
    reference = [row for row in transforms if str(row['is_reference']).lower() == 'true']
    reference_identity = len(reference) == 1 and np.allclose(json.loads(reference[0]['source_to_global_level0_json']), np.eye(3))
    status = 'PASS' if not failed and min(determinants) > 0 and reference_identity else 'PASS_WITH_TERMINAL_FAILURES'
    write_json(qc_root / 'global_qc_summary.json', {'status': status, 'slide_count': 260, 'edge_count': 259, 'failed_edge_count': len(failed), 'mirror_count': sum((value <= 0 for value in determinants)), 'reference_identity': reference_identity, 'visual_pages': contact_paths, 'flicker_count': len(flickers), 'flicker_manifest': str(qc_root / 'global_flicker_manifest.tsv'), 'volume_visuals': volume_visuals})
    stage_status(cfg, 'global_qc', status, failed_edges=len(failed), visual_pages=len(contact_paths))
