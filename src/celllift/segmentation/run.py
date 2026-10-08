from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
from celllift.runtime import json
import subprocess
from celllift.runtime import ResourcePath as Path
from celllift.segmentation.common import load_config
from celllift.segmentation.prepare import run as run_prepare
SCRIPT_ROOT = Path(__file__).resolve().parent

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=('prepare', 'audit'), required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = run_prepare(args.config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result['status'] == 'PASS' else 2
    cfg = load_config(args.config)
    return subprocess.run([str(cfg['runtime']['python']), str(SCRIPT_ROOT / 'audit.py'), '--config', str(args.config)]).returncode
if __name__ == '__main__':
    raise SystemExit(main())
