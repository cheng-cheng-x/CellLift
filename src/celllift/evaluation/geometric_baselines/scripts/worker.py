from celllift.runtime import resource_path as _public_resource
import sys, time, subprocess, traceback, json
from celllift.evaluation.geometric_baselines.src.common import ROOT, RESULT, CONFIG, layout, write, record

def wait_for(files, phase):
    while not all((p.exists() for p in files)):
        write(RESULT / 'runtime' / ('wait_' + sys.argv[1] + '.json'), record(phase=phase, missing=[str(p) for p in files if not p.exists()]))
        time.sleep(5)

def run(module, args):
    log = RESULT / 'logs' / (module.split('.')[-1] + '_' + sys.argv[1] + '.log')
    with log.open('a') as f:
        p = subprocess.Popen([sys.executable, '-m', module, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        write(RESULT / 'runtime' / (module.split('.')[-1] + '_' + sys.argv[1] + '_process.json'), record(pid=p.pid, status='RUNNING'))
        code = p.wait()
    write(RESULT / 'runtime' / (module.split('.')[-1] + '_' + sys.argv[1] + '_process.json'), record(pid=p.pid, status='DONE' if code == 0 else 'FAILED', exit_code=code))
    if code:
        raise RuntimeError(module + ' failed with ' + str(code))

def main(method):
    layout()
    if method == 'cylinder' and (not (RESULT / '01_validation/geometry.json').exists()):
        run('tests.geometry', [])
    wait_for([RESULT / '01_validation/geometry.json'], 'validation')
    assert json.loads((RESULT / '01_validation/geometry.json').read_text())['status'] == 'PASSED'
    run('scripts.calibrate', [method])
    run('scripts.predict', [method])
    wait_for([RESULT / 'runtime' / ('predict_' + m + '_done.json') for m in CONFIG['methods']], 'all_input_predictions')
    if method == 'p4':
        write(RESULT / 'runtime/predictions_complete.json', record(status='ALL_INPUT_ONLY_PREDICTIONS_SAVED'))
    wait_for([RESULT / 'runtime/predictions_complete.json'], 'prediction_barrier')
    run('scripts.evaluate', [method])
    if method == 'p4':
        wait_for([RESULT / 'runtime' / ('evaluate_' + m + '_done.json') for m in CONFIG['methods']], 'all_evaluations')
        run('scripts.summarize', [])
        write(RESULT / 'runtime/complete.json', record(status='COMPLETE'))
    write(RESULT / 'runtime' / ('worker_' + method + '_done.json'), record(status='DONE'))
if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except BaseException:
        write(RESULT / 'runtime' / ('failure_' + sys.argv[1] + '_' + str(time.time_ns()) + '.json'), record(traceback=traceback.format_exc()))
        raise
