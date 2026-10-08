from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import csv
import gzip
import hashlib
from celllift.runtime import json
import math
import os
import time
from celllift.runtime import ResourcePath as Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
QUANTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {}
    return {f"p{str(q).replace('.', '_')}": float(np.percentile(values, q)) for q in QUANTILES}

def border_lookup(instances: list[dict[str, Any]]) -> set[int]:
    return {int(instance['instance_id']) for instance in instances if bool(instance.get('border_flag'))}

def write_histogram(path: Path, bins: np.ndarray, all_values: np.ndarray, clean_values: np.ndarray) -> None:
    all_counts, _ = np.histogram(all_values, bins=bins)
    clean_counts, _ = np.histogram(clean_values, bins=bins)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['bin_left', 'bin_right', 'all_accepted_count', 'clean_count'])
        for left, right, all_count, clean_count in zip(bins[:-1], bins[1:], all_counts, clean_counts):
            writer.writerow([f'{left:.10g}', f'{right:.10g}', int(all_count), int(clean_count)])

def plot_distribution(path: Path, all_ratios: np.ndarray, clean_ratios: np.ndarray, all_fractions: np.ndarray, clean_fractions: np.ndarray, roi_medians: np.ndarray) -> None:
    ratio_hi = float(np.percentile(all_ratios, 99.5))
    fraction_hi = min(float(np.percentile(all_fractions, 99.5)), 1.0)
    ratio_bins = np.logspace(-2.2, math.log10(max(ratio_hi, 0.02)), 100)
    fraction_bins = np.linspace(0, max(fraction_hi, 0.05), 100)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.hist(all_ratios, bins=ratio_bins, density=True, histtype='step', linewidth=1.5, label=f'all accepted (n={len(all_ratios):,})')
    ax.hist(clean_ratios, bins=ratio_bins, density=True, alpha=0.35, label=f'clean internal (n={len(clean_ratios):,})')
    ax.set_xscale('log')
    ax.set_xlabel('N:C area ratio = nucleus / cytoplasm')
    ax.set_ylabel('density')
    ax.set_title('Nuclear-to-cytoplasmic area ratio')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax = axes[0, 1]
    ax.hist(all_fractions, bins=fraction_bins, density=True, histtype='step', linewidth=1.5, label='all accepted')
    ax.hist(clean_fractions, bins=fraction_bins, density=True, alpha=0.35, label='clean internal')
    ax.set_xlabel('nuclear fraction = nucleus / (nucleus + cytoplasm)')
    ax.set_ylabel('density')
    ax.set_title('Nuclear area fraction')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax = axes[1, 0]
    ordered = np.sort(clean_ratios)
    probabilities = np.arange(1, len(ordered) + 1) / len(ordered)
    ax.plot(ordered, probabilities, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlim(max(float(np.percentile(ordered, 0.1)), 0.001), ratio_hi)
    ax.set_xlabel('N:C area ratio')
    ax.set_ylabel('cumulative fraction')
    ax.set_title('Clean-pair empirical CDF')
    for q in (10, 50, 90):
        value = float(np.percentile(ordered, q))
        ax.axvline(value, linewidth=1, linestyle='--', alpha=0.6)
        ax.text(value, q / 100, f' p{q}={value:.3f}', va='bottom')
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    roi_hi = float(np.percentile(roi_medians, 99.5))
    ax.hist(roi_medians, bins=np.linspace(0, roi_hi, 80), alpha=0.65)
    ax.axvline(float(np.median(roi_medians)), linewidth=1.5, linestyle='--', label=f'median={np.median(roi_medians):.3f}')
    ax.set_xlabel('median N:C ratio per ROI')
    ax.set_ylabel('ROI count')
    ax.set_title(f'Across-ROI distribution (n={len(roi_medians):,})')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.suptitle('Cellpose 3 dual H&E — accepted nucleus/cell pairs', fontsize=14)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    fig.savefig(temporary, dpi=180, format='png')
    plt.close(fig)
    os.replace(temporary, path)

def run(result_root: Path) -> dict[str, Any]:
    started = time.time()
    output_root = result_root / '07_analysis/nuclear_cytoplasmic_ratio_set_encoding'
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / '05_manifests/segmentation_manifest.tsv'
    audit_path = result_root / '06_audit/final_audit.json'
    audit = read_json(audit_path)
    if audit.get('status') != 'PASS':
        raise RuntimeError('segmentation final audit is not PASS')
    with manifest_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if len(rows) != int(audit['terminal_valid_roi_layer_count']):
        raise RuntimeError('segmentation manifest count mismatch')
    pair_table_path = output_root / 'accepted_pair_nc_ratio.tsv.gz'
    pair_tmp = pair_table_path.with_name(f'.{pair_table_path.name}.tmp.{os.getpid()}')
    roi_rows: list[dict[str, Any]] = []
    all_ratios: list[float] = []
    clean_ratios: list[float] = []
    all_fractions: list[float] = []
    clean_fractions: list[float] = []
    invalid_cytoplasm_count = 0
    accepted_count = 0
    clean_count = 0
    mpp = 0.46
    area_scale = mpp * mpp
    with gzip.open(pair_tmp, 'wt', newline='', encoding='utf-8') as handle:
        fields = ['roi_layer_id', 'track_id', 'section_id', 'nucleus_id', 'cell_id', 'nucleus_area_px', 'cell_area_px', 'overlap_px', 'cytoplasm_area_px', 'nucleus_area_um2', 'cytoplasm_area_um2', 'nc_area_ratio', 'nuclear_fraction', 'nucleus_containment_fraction', 'nucleus_border_flag', 'cell_border_flag', 'clean_analysis_flag']
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            pair_record = read_gzip_json(Path(row['pairs_path']))
            nucleus_record = read_gzip_json(Path(row['nucleus_instances_path']))
            cell_record = read_gzip_json(Path(row['cell_instances_path']))
            nucleus_borders = border_lookup(nucleus_record['instances'])
            cell_borders = border_lookup(cell_record['instances'])
            roi_all: list[float] = []
            roi_clean: list[float] = []
            for relation in pair_record['nucleus_relations']:
                if relation['pair_status'] != 'accepted_one_to_one':
                    continue
                accepted_count += 1
                nucleus_area = int(relation['nucleus_area_px'])
                cell_area = int(relation['cell_area_px'])
                overlap = int(relation['overlap_px'])
                cytoplasm_area = cell_area - overlap
                if cytoplasm_area <= 0:
                    invalid_cytoplasm_count += 1
                    continue
                ratio = nucleus_area / cytoplasm_area
                fraction = nucleus_area / (nucleus_area + cytoplasm_area)
                containment = float(relation['nucleus_containment_fraction'])
                nucleus_id = int(relation['nucleus_id'])
                cell_id = int(relation['cell_id'])
                nucleus_border = nucleus_id in nucleus_borders
                cell_border = cell_id in cell_borders
                clean = containment >= 0.9 and (not nucleus_border) and (not cell_border)
                all_ratios.append(ratio)
                all_fractions.append(fraction)
                roi_all.append(ratio)
                if clean:
                    clean_count += 1
                    clean_ratios.append(ratio)
                    clean_fractions.append(fraction)
                    roi_clean.append(ratio)
                writer.writerow({'roi_layer_id': row['roi_layer_id'], 'track_id': row['track_id'], 'section_id': row['section_id'], 'nucleus_id': nucleus_id, 'cell_id': cell_id, 'nucleus_area_px': nucleus_area, 'cell_area_px': cell_area, 'overlap_px': overlap, 'cytoplasm_area_px': cytoplasm_area, 'nucleus_area_um2': f'{nucleus_area * area_scale:.6f}', 'cytoplasm_area_um2': f'{cytoplasm_area * area_scale:.6f}', 'nc_area_ratio': f'{ratio:.9g}', 'nuclear_fraction': f'{fraction:.9g}', 'nucleus_containment_fraction': f'{containment:.9g}', 'nucleus_border_flag': int(nucleus_border), 'cell_border_flag': int(cell_border), 'clean_analysis_flag': int(clean)})
            roi_rows.append({'roi_layer_id': row['roi_layer_id'], 'track_id': row['track_id'], 'section_id': row['section_id'], 'accepted_valid_count': len(roi_all), 'clean_count': len(roi_clean), 'all_nc_ratio_median': float(np.median(roi_all)) if roi_all else math.nan, 'clean_nc_ratio_p10': float(np.percentile(roi_clean, 10)) if roi_clean else math.nan, 'clean_nc_ratio_median': float(np.median(roi_clean)) if roi_clean else math.nan, 'clean_nc_ratio_p90': float(np.percentile(roi_clean, 90)) if roi_clean else math.nan})
            if index % 100 == 0:
                atomic_json(output_root / 'runtime_progress.json', {'status': 'running', 'processed_roi_count': index, 'expected_roi_count': len(rows), 'accepted_pair_count': accepted_count, 'clean_pair_count': clean_count, 'elapsed_seconds': time.time() - started})
    os.replace(pair_tmp, pair_table_path)
    roi_summary_path = output_root / 'roi_nc_ratio_summary.tsv'
    roi_fields = list(roi_rows[0])
    with roi_summary_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=roi_fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(roi_rows)
    all_ratio_array = np.asarray(all_ratios, dtype=np.float64)
    clean_ratio_array = np.asarray(clean_ratios, dtype=np.float64)
    all_fraction_array = np.asarray(all_fractions, dtype=np.float64)
    clean_fraction_array = np.asarray(clean_fractions, dtype=np.float64)
    roi_medians = np.asarray([float(row['clean_nc_ratio_median']) for row in roi_rows if math.isfinite(float(row['clean_nc_ratio_median']))], dtype=np.float64)
    ratio_hist_path = output_root / 'nc_ratio_histogram.tsv'
    fraction_hist_path = output_root / 'nuclear_fraction_histogram.tsv'
    write_histogram(ratio_hist_path, np.logspace(-3, 2, 201), all_ratio_array, clean_ratio_array)
    write_histogram(fraction_hist_path, np.linspace(0, 1, 201), all_fraction_array, clean_fraction_array)
    plot_path = output_root / 'nc_ratio_distribution.png'
    plot_distribution(plot_path, all_ratio_array, clean_ratio_array, all_fraction_array, clean_fraction_array, roi_medians)
    summary = {'status': 'PASS', 'definition': 'N:C area ratio = full nucleus area / (cell-label area - nucleus/cell overlap area)', 'nuclear_fraction_definition': 'full nucleus area / (full nucleus area + cytoplasm area)', 'accepted_pair_definition': 'pair_status == accepted_one_to_one', 'clean_subset_definition': 'accepted pair; containment >= 0.90; neither nucleus nor cell touches ROI border', 'mpp_um_per_px': mpp, 'pixel_area_um2': area_scale, 'roi_count': len(rows), 'accepted_pair_count_before_area_check': accepted_count, 'invalid_nonpositive_cytoplasm_count': invalid_cytoplasm_count, 'all_valid_pair_count': int(all_ratio_array.size), 'clean_pair_count': int(clean_ratio_array.size), 'clean_pair_fraction_of_valid': float(clean_ratio_array.size / max(all_ratio_array.size, 1)), 'all_nc_ratio_quantiles': quantiles(all_ratio_array), 'clean_nc_ratio_quantiles': quantiles(clean_ratio_array), 'all_nuclear_fraction_quantiles': quantiles(all_fraction_array), 'clean_nuclear_fraction_quantiles': quantiles(clean_fraction_array), 'roi_clean_median_nc_ratio_quantiles': quantiles(roi_medians), 'source_segmentation_manifest': str(manifest_path), 'source_segmentation_manifest_sha256': sha256_file(manifest_path), 'source_final_audit': str(audit_path), 'source_algorithm_fingerprint': audit['algorithm_fingerprint'], 'elapsed_seconds': time.time() - started}
    summary_path = output_root / 'summary.json'
    atomic_json(summary_path, summary)
    manifest = {'status': 'PASS', 'files': {path.name: {'path': str(path), 'sha256': sha256_file(path), 'size_bytes': path.stat().st_size} for path in (summary_path, pair_table_path, roi_summary_path, ratio_hist_path, fraction_hist_path, plot_path)}}
    atomic_json(output_root / 'analysis_manifest.json', manifest)
    atomic_json(output_root / 'runtime_progress.json', {'status': 'complete', 'processed_roi_count': len(rows), 'expected_roi_count': len(rows), 'accepted_pair_count': accepted_count, 'clean_pair_count': clean_count, 'elapsed_seconds': time.time() - started})
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.result_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 2
if __name__ == '__main__':
    raise SystemExit(main())
