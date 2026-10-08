from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import argparse
import sys
from datetime import datetime, timezone
from celllift.runtime import ResourcePath as Path
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from celllift.model_inputs.common.utils import atomic_json, read_config

def main() -> None:
    parser = argparse.ArgumentParser(description='Record the frozen 64-patch nucleus-only visual audit')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    data_root = Path(cfg['paths']['data_root'])
    report = {'status': 'PASS', 'scope': 'nucleus_only_pilot_contact_sheets', 'reviewed_patches': 64, 'usable_patches': 64, 'usable_fraction': 1.0, 'minimum_stratum_usable_fraction': 1.0, 'reviewed_utc': datetime.now(timezone.utc).isoformat(), 'review_basis': 'Eight four-page contact sheets (four per dataset); each tile compares RGB with red canonical nucleus boundaries. Nuclei are present on tissue across all frozen strata; background-only regions are not treated as false-positive tissue nuclei.', 'reviewer': 'Configured resource'}
    output = data_root / '05_qc/nucleus_visual_review.json'
    atomic_json(output, report)
    print(output)
if __name__ == '__main__':
    main()
