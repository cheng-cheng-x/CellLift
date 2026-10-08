from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import sys
from celllift.runtime import ResourcePath as Path
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument('stage', choices=('audit-sicap', 'stats', 'hact-prepare', 'job-list', 'train', 'fuse-sicap', 'tables'))
    value.add_argument('--device', default='cuda')
    value.add_argument('--job-id')
    return value

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.stage == 'audit-sicap':
        from celllift.benchmark_prediction.audit_prostate_baseline import run
        print(json.dumps(run(args.device), indent=2, default=str))
        return 0
    if args.stage == 'stats':
        from celllift.benchmark_prediction.paired_statistics import run
        print(json.dumps(run(), indent=2, default=str))
        return 0
    if args.stage == 'hact-prepare':
        from celllift.benchmark_prediction.hact_prepare import run
        print(json.dumps(run(), indent=2, default=str))
        return 0
    if args.stage == 'job-list':
        from celllift.benchmark_prediction.jobs import comparison_job_matrix
        jobs = comparison_job_matrix()
        print(json.dumps({'n': len(jobs), 'ids': [j['job_id'] for j in jobs]}, indent=2))
        return 0
    if args.stage == 'train':
        from celllift.benchmark_prediction.jobs import comparison_job_matrix
        from celllift.benchmark_prediction.run import run_one
        jobs = comparison_job_matrix()
        if args.job_id:
            jobs = [job for job in jobs if job['job_id'] == args.job_id]
        if not jobs:
            print(json.dumps({'status': 'FAILED', 'error': 'unknown r3 job'}))
            return 1
        for job in jobs:
            if job['family'] == 'fusion':
                continue
            print(json.dumps(run_one(job, args.device)), flush=True)
        return 0
    if args.stage == 'fuse-sicap':
        from celllift.benchmark_prediction.evaluate import _apply_locked_fusion, _load_pred, _metric_pack
        from celllift.benchmark_prediction.fuse import fuse_from_job
        from celllift.benchmark_prediction.io import job_dir
        from celllift.benchmark_prediction.jobs import comparison_job_matrix
        from celllift.benchmark_prediction.paths import baseline_result_root, comparison_result_root, result_root
        from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet
        import numpy as np
        val_scores = {}
        for job in comparison_job_matrix():
            if job['family'] != 'fusion':
                continue
            result = fuse_from_job(job)
            print(json.dumps({'job': job['job_id'], **{k: result.get(k) for k in ('status', 'best', 'weights')}}, default=str), flush=True)
            if result.get('status') in {'PASS', 'REUSED'} or result.get('weights'):
                val_scores[job['arm']] = float(result.get('best') or -1)
        primary = 'B+G3'
        if val_scores.get('B+GR', -1) > val_scores.get('B+G3', -1) + 1e-12:
            primary = 'B+GR'
        elif abs(val_scores.get('B+GR', -1) - val_scores.get('B+G3', -1)) <= 1e-12 and val_scores:
            primary = 'B+GR' if val_scores.get('B+GR', 0) >= 0 and False else 'B+G3'
        b_test = baseline_result_root() / 'sicapv2/baseline/B_paper200/full_train/seed_42/fold_00/test_predictions.parquet'
        applied = {}
        for arm in ('G2', 'G3', 'GR'):
            dest = job_dir({'dataset': 'sicapv2', 'family': 'fusion', 'arm': f'B+{arm}', 'unit': 'fold_00', 'seed': 42, 'fold': 0, 'r3': True})
            weights_path = dest / 'weights.json'
            if not weights_path.is_file() or not b_test.is_file():
                applied[f'B+{arm}'] = {'status': 'WAIT'}
                continue
            e_test = result_root() / f'sicapv2/geometry/{arm}/full_train/seed_42/fold_00/test_predictions.parquet'
            if not e_test.is_file():
                applied[f'B+{arm}'] = {'status': 'WAIT', 'e': str(e_test)}
                continue
            weights = json.loads(weights_path.read_text(encoding='utf-8'))
            fused = _apply_locked_fusion(_load_pred(b_test), _load_pred(e_test), weights, 'sicapv2')
            atomic_parquet(dest / 'test_predictions.parquet', fused)
            probs = np.stack([np.asarray(row['probability'], np.float64) for row in fused], 0)
            labels = np.asarray([int(row['label_id']) for row in fused])
            metrics = _metric_pack('sicapv2', labels, probs)
            atomic_json(dest / 'test_metrics.json', metrics)
            applied[f'B+{arm}'] = {'status': 'PASS', 'n': len(fused), 'metrics': metrics, 'weights': weights}
        payload = {'val_scores': val_scores, 'primary_3d': primary, 'test': applied}
        out = comparison_result_root() / 'analysis' / 'sicap_fusion.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.stage == 'tables':
        from celllift.benchmark_prediction.comparison_tables import write_tables
        print(json.dumps(write_tables(), indent=2, default=str))
        return 0
    return 2
if __name__ == '__main__':
    raise SystemExit(main())
