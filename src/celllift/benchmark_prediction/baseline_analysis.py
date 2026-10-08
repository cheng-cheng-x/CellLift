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
    value.add_argument('stage', choices=('diagnose-sicap', 'crc-qualify', 'bracs-check', 'infer-sicap-b', 'infer-test', 'job-list', 'train', 'fuse', 'evaluate', 'tables'))
    value.add_argument('--device', default='cuda')
    value.add_argument('--job-id')
    return value

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.stage == 'diagnose-sicap':
        from celllift.benchmark_prediction.diagnose_sicap_b import run
        print(json.dumps(run(args.device), indent=2, default=str))
        return 0
    if args.stage == 'crc-qualify':
        from celllift.benchmark_prediction.crc_score import qualification_diff
        from celllift.benchmark_prediction.evaluate import _load_pred, _metrics_from_rows
        from celllift.benchmark_prediction.io import job_dir
        from celllift.benchmark_prediction.paths import baseline_result_root
        report = qualification_diff()
        before_after = {}
        for family, arm in (('baseline', 'B'), ('geometry', 'G2'), ('geometry', 'G3'), ('geometry', 'GR'), ('route', 'B2'), ('route', 'B3'), ('fusion', 'B+G2'), ('fusion', 'B+G3'), ('fusion', 'B+GR')):
            path = job_dir({'dataset': 'tcga_crc_msi', 'family': family, 'arm': arm, 'unit': 'official', 'seed': 42, 'fold': 0}) / 'test_predictions.parquet'
            if not path.is_file():
                before_after[f'{family}/{arm}'] = {'status': 'MISSING'}
                continue
            rows = _load_pred(path)
            from celllift.benchmark_prediction.evaluate import _metric_pack
            import numpy as np
            probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0)
            labels = np.asarray([int(row['label_id']) for row in rows])
            unfiltered = _metric_pack('tcga_crc_msi', labels, probs)
            qualified = _metrics_from_rows('tcga_crc_msi', rows)
            before_after[f'{family}/{arm}'] = {'n_raw': len(rows), 'unfiltered': unfiltered, 'qualified_140': qualified, 'path': str(path)}
        report['before_after_compatibility_test'] = before_after
        out = baseline_result_root() / 'analysis' / 'c0_qualification.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        slim = {k: report[k] for k in report if k != 'qualified_ids'}
        print(json.dumps(slim, indent=2, default=str))
        print(json.dumps({'written': str(out), 'qualified_n': report['qualified_n'], 'unit_test_n': report['unit_test_n']}))
        return 0
    if args.stage == 'bracs-check':
        from celllift.benchmark_prediction.check_bracs import run
        print(json.dumps(run(), indent=2, default=str))
        return 0
    if args.stage == 'infer-sicap-b':
        from celllift.benchmark_prediction.evaluate import _metric_pack
        from celllift.benchmark_prediction.io import job_dir, write_pass
        from celllift.benchmark_prediction.predict import predict_test
        from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet
        import numpy as np
        source = {'dataset': 'sicapv2', 'family': 'baseline', 'arm': 'B', 'unit': 'full_train', 'seed': 42, 'fold': 0}
        rows = predict_test(source, args.device)
        if not rows:
            print(json.dumps({'status': 'FAILED', 'reason': 'no_rows'}))
            return 1
        dest = job_dir({**source, 'baseline_qualification': True})
        dest.mkdir(parents=True, exist_ok=True)
        atomic_parquet(dest / 'test_predictions.parquet', rows)
        probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0)
        labels = np.asarray([int(row['label_id']) for row in rows])
        metrics = _metric_pack('sicapv2', labels, probs)
        atomic_json(dest / 'test_metrics.json', metrics)
        write_pass(dest, {'dataset': 'sicapv2', 'arm': 'B', 'source_ckpt': 'compatibility_full_train', 'reused_weights': True, 're_inferred': True, 'metrics': metrics, 'n': len(rows)})
        print(json.dumps({'status': 'PASS', 'n': len(rows), 'metrics': metrics, 'path': str(dest)}, indent=2))
        return 0
    if args.stage == 'infer-test':
        from celllift.benchmark_prediction.evaluate import _complete_test_rows, _metric_pack
        from celllift.benchmark_prediction.io import job_dir
        from celllift.benchmark_prediction.jobs import baseline_job_matrix
        from celllift.benchmark_prediction.predict import predict_test
        from celllift.morphology_interaction.io_utils import atomic_json, atomic_parquet
        import numpy as np
        jobs = [job for job in baseline_job_matrix() if job['family'] != 'fusion']
        if args.job_id:
            jobs = [job for job in jobs if job['job_id'] == args.job_id]
        if not jobs:
            print(json.dumps({'status': 'FAILED', 'error': 'no infer-test job'}))
            return 1
        for job in jobs:
            dest = job_dir(job)
            rows = predict_test(job, args.device)
            if not rows:
                print(json.dumps({'job': job['job_id'], 'status': 'SKIP'}))
                continue
            complete, why = _complete_test_rows(job, rows, exact=True)
            if not complete:
                print(json.dumps({'job': job['job_id'], 'status': 'WAIT', 'reason': why, 'n': len(rows)}))
                continue
            dest.mkdir(parents=True, exist_ok=True)
            atomic_parquet(dest / 'test_predictions.parquet', rows)
            probs = np.stack([np.asarray(row['probability'], np.float64) for row in rows], 0)
            labels = np.asarray([int(row['label_id']) for row in rows])
            metrics = _metric_pack(job['dataset'], labels, probs)
            atomic_json(dest / 'test_metrics.json', metrics)
            print(json.dumps({'job': job['job_id'], 'status': 'PASS', 'n': len(rows), 'metrics': metrics}, indent=2))
        return 0
    if args.stage == 'job-list':
        from celllift.benchmark_prediction.jobs import baseline_job_matrix
        jobs = baseline_job_matrix()
        slim = [{k: v for k, v in job.items() if k not in {'official_fit', 'official_val', 'official_predict', 'official_test'}} for job in jobs]
        print(json.dumps({'n': len(jobs), 'jobs': slim}, indent=2))
        return 0
    if args.stage == 'train':
        from celllift.benchmark_prediction.jobs import baseline_job_matrix
        from celllift.benchmark_prediction.run import run_one
        jobs = baseline_job_matrix()
        if args.job_id:
            jobs = [job for job in jobs if job['job_id'] == args.job_id]
        if not jobs:
            print(json.dumps({'status': 'FAILED', 'error': 'unknown r2 job'}))
            return 1
        for job in jobs:
            if job['family'] == 'fusion':
                continue
            print(json.dumps(run_one(job, args.device)), flush=True)
        return 0
    if args.stage == 'fuse':
        from celllift.benchmark_prediction.fuse import fuse_from_job
        from celllift.benchmark_prediction.jobs import baseline_job_matrix
        from celllift.benchmark_prediction.predict import materialize_val_predictions
        print(json.dumps(materialize_val_predictions({'dataset': 'tcga_crc_msi', 'family': 'route', 'arm': 'B2', 'unit': 'official', 'seed': 42, 'fold': 0, 'baseline_qualification': True, 'route': 'b', 'refresh_val': True}, args.device)), flush=True)
        for job in baseline_job_matrix():
            if job['family'] != 'fusion':
                continue
            print(json.dumps({'job': job['job_id'], **fuse_from_job(job)}, default=str), flush=True)
        return 0
    if args.stage == 'evaluate':
        from celllift.benchmark_prediction.evaluate import evaluate_baseline
        print(json.dumps(evaluate_baseline(args.device), indent=2, default=str))
        return 0
    if args.stage == 'tables':
        from celllift.benchmark_prediction.evaluate import write_baseline_tables
        print(json.dumps(write_baseline_tables(), indent=2, default=str))
        return 0
    return 2
if __name__ == '__main__':
    raise SystemExit(main())
