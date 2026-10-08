from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from . import SEED
from .splits import load_units
GEOMETRY_ARMS = ('G2', 'G3', 'GR')
SICAP_GEOM_EXTRA = ('G2R', 'G2S')
ROUTE_A_ARMS = ('A2', 'AS')
ROUTE_C_ARMS = ('C2', 'C3')
EXPERT_ARMS = ('E2', 'ES', 'ER')

def _unit_payload(unit: dict[str, Any]) -> dict[str, Any]:
    return {'unit': unit['name'], 'official_fit': unit.get('fit') or unit.get('fit_tiles') or [], 'official_val': unit.get('val') or unit.get('val_tiles') or [], 'official_predict': unit.get('predict') or unit.get('val') or unit.get('val_tiles') or [], 'official_test': unit.get('test') or []}

def _sicap_dev_units() -> list[dict[str, Any]]:
    return [unit for unit in load_units('sicapv2') if unit['kind'] == 'sicap_official_fold']

def _one(dataset: str, family: str, arm: str, unit: dict[str, Any], **extra) -> dict[str, Any]:
    payload = {'dataset': dataset, 'family': family, 'arm': arm, 'seed': SEED, 'priority': extra.pop('priority', 50), **_unit_payload(unit), **extra}
    if dataset == 'sicapv2' and unit['kind'] == 'sicap_official_fold':
        payload['fold'] = int(unit['name'].split('_')[1])
    else:
        payload['fold'] = extra.get('fold', 0)
    return payload

def job_matrix() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    sicap_units = _sicap_dev_units()
    bracs = load_units('bracs')[0]
    crc = load_units('tcga_crc_msi')[0]
    for unit in sicap_units:
        jobs.append(_one('sicapv2', 'baseline', 'B', unit, priority=10, max_epochs=200, patience=10))
        for arm in GEOMETRY_ARMS + SICAP_GEOM_EXTRA:
            jobs.append(_one('sicapv2', 'geometry', arm, unit, encoder='deepsets', priority=12, max_epochs=60, patience=10))
        for arm in ROUTE_A_ARMS:
            jobs.append(_one('sicapv2', 'route', arm, unit, route='a', priority=40, max_epochs=60, patience=10))
        for arm in ROUTE_C_ARMS:
            jobs.append(_one('sicapv2', 'route', arm, unit, route='c', priority=40, max_epochs=60, patience=10))
        jobs.append(_one('sicapv2', 'pretrain', 'A2', unit, route='a', target='2d', priority=30))
        jobs.append(_one('sicapv2', 'pretrain', 'AS', unit, route='a', target='3d', priority=30))
    jobs.append(_one('bracs', 'baseline', 'B_rgb', bracs, priority=10, max_epochs=60, patience=10))
    for arm in GEOMETRY_ARMS:
        jobs.append(_one('bracs', 'geometry', arm, bracs, encoder='deepsets', priority=12, max_epochs=60, patience=10))
    for arm in EXPERT_ARMS:
        jobs.append(_one('bracs', 'expert', arm, bracs, priority=14, max_epochs=60, patience=10))
    jobs.append(_one('bracs', 'feature', 'H2', bracs, priority=45, max_epochs=80, patience=12))
    jobs.append(_one('bracs', 'feature', 'H3', bracs, priority=45, max_epochs=80, patience=12))
    jobs.append(_one('bracs', 'pretrain', 'D2', bracs, route='d', target='2d', priority=30))
    jobs.append(_one('bracs', 'pretrain', 'D3', bracs, route='d', target='3d', priority=30))
    jobs.append(_one('bracs', 'route', 'D2', bracs, route='d', priority=42, max_epochs=60, patience=10))
    jobs.append(_one('bracs', 'route', 'D3', bracs, route='d', priority=42, max_epochs=60, patience=10))
    jobs.append(_one('tcga_crc_msi', 'baseline', 'B', crc, priority=10, max_epochs=100, patience=10))
    for arm in GEOMETRY_ARMS:
        jobs.append(_one('tcga_crc_msi', 'geometry', arm, crc, encoder='deepsets', priority=12, max_epochs=60, patience=10))
    jobs.append(_one('tcga_crc_msi', 'geometry', 'G2', crc, encoder='meanpool', priority=13, max_epochs=60, patience=10, arm_tag='MP-G2'))
    jobs.append(_one('tcga_crc_msi', 'geometry', 'G3', crc, encoder='meanpool', priority=13, max_epochs=60, patience=10, arm_tag='MP-G3'))
    jobs.append(_one('tcga_crc_msi', 'route', 'B2', crc, route='b', priority=20, max_epochs=60, patience=10))
    jobs.append(_one('tcga_crc_msi', 'route', 'B3', crc, route='b', priority=20, max_epochs=60, patience=10))
    jobs.append(_one('tcga_crc_msi', 'feature', 'C1', crc, priority=45, max_epochs=80, patience=12))
    jobs.append(_one('tcga_crc_msi', 'feature', 'C4', crc, priority=45, max_epochs=80, patience=12))
    for dataset, units in (('sicapv2', sicap_units), ('bracs', [bracs]), ('tcga_crc_msi', [crc])):
        for unit in units:
            jobs.append(_one(dataset, 'residual', 'probe', unit, priority=11, max_epochs=80, patience=12))
    full = next((unit for unit in load_units('sicapv2') if unit['kind'] == 'sicap_full_train'))
    for arm in ('B',):
        jobs.append(_one('sicapv2', 'baseline', arm, full, priority=80, max_epochs=200, patience=10 ** 9, retrain=True))
    for arm in GEOMETRY_ARMS + SICAP_GEOM_EXTRA:
        jobs.append(_one('sicapv2', 'geometry', arm, full, encoder='deepsets', priority=80, max_epochs=60, patience=10 ** 9, retrain=True))
    jobs.append(_one('sicapv2', 'residual', 'probe', full, priority=79, max_epochs=80, patience=12, retrain=True))
    for arm in ROUTE_A_ARMS:
        jobs.append(_one('sicapv2', 'route', arm, full, route='a', priority=82, max_epochs=60, patience=10 ** 9, retrain=True))
    for arm in ROUTE_C_ARMS:
        jobs.append(_one('sicapv2', 'route', arm, full, route='c', priority=82, max_epochs=60, patience=10 ** 9, retrain=True))
    jobs.append(_one('sicapv2', 'pretrain', 'A2', full, route='a', target='2d', priority=81, retrain=True))
    jobs.append(_one('sicapv2', 'pretrain', 'AS', full, route='a', target='3d', priority=81, retrain=True))
    for unit in sicap_units:
        for arm in GEOMETRY_ARMS + SICAP_GEOM_EXTRA:
            jobs.append(_one('sicapv2', 'fusion', f'B+{arm}', unit, partner_family='baseline', partner_arm='B', expert_family='geometry', expert_arm=arm, priority=90, temperature=False))
    for arm in ('G2', 'G3', 'GR', 'E2', 'ES', 'ER'):
        family = 'expert' if arm.startswith('E') else 'geometry'
        jobs.append(_one('bracs', 'fusion', f'B_rgb+{arm}', bracs, partner_family='baseline', partner_arm='B_rgb', expert_family=family, expert_arm=arm, priority=90, temperature=False))
    for arm in ('G2', 'G3', 'GR', 'MP-G2', 'MP-G3'):
        jobs.append(_one('tcga_crc_msi', 'fusion', f'B+{arm}', crc, partner_family='baseline', partner_arm='B', expert_family='geometry', expert_arm=arm, priority=90, temperature=True))
    jobs.sort(key=lambda row: (int(row.get('priority', 50)), row['dataset'], row['family'], row.get('arm_tag') or row['arm'], row.get('unit', '')))
    for index, job in enumerate(jobs):
        job['job_id'] = f"{index:04d}__{job['dataset']}__{job['family']}__{job.get('arm_tag') or job['arm']}__{job.get('unit')}"
    return jobs

def baseline_job_matrix() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    full = next((unit for unit in load_units('sicapv2') if unit['kind'] == 'sicap_full_train'))
    crc = load_units('tcga_crc_msi')[0]
    jobs.append(_one('sicapv2', 'baseline', 'B', full, priority=10, max_epochs=200, patience=10 ** 9, paper_recipe=True, r2=True, arm_tag='B_paper200'))
    jobs.append(_one('tcga_crc_msi', 'baseline', 'B', crc, priority=12, max_epochs=100, patience=10, r2=True, arm_tag='B', crc_hard_vote=True))
    for arm in GEOMETRY_ARMS:
        jobs.append(_one('tcga_crc_msi', 'fusion', f'B+{arm}', crc, partner_family='baseline', partner_arm='B', expert_family='geometry', expert_arm=arm, priority=90, temperature=False, r2=True))
    jobs.append(_one('tcga_crc_msi', 'fusion', 'P2', crc, partner_family='route', partner_arm='B2', expert_family='route', expert_arm='B2', priority=91, temperature=False, r2=True, identity=True))
    for arm in GEOMETRY_ARMS:
        jobs.append(_one('tcga_crc_msi', 'fusion', f'P2+{arm}', crc, partner_family='route', partner_arm='B2', expert_family='geometry', expert_arm=arm, priority=92, temperature=False, r2=True))
    for index, job in enumerate(jobs):
        job['job_id'] = f"r2_{index:04d}__{job['dataset']}__{job['family']}__{job.get('arm_tag') or job['arm']}__{job.get('unit')}"
    return jobs

def comparison_job_matrix() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    fold0 = next((unit for unit in _sicap_dev_units() if unit['name'] == 'fold_00'))
    jobs.append(_one('sicapv2', 'baseline', 'B', fold0, priority=10, max_epochs=200, patience=10 ** 9, paper_recipe=True, r3=True, arm_tag='B_fold0_200'))
    for arm in GEOMETRY_ARMS:
        jobs.append(_one('sicapv2', 'fusion', f'B+{arm}', fold0, partner_family='baseline', partner_arm='B_fold0_200', expert_family='geometry', expert_arm=arm, priority=90, temperature=False, r3=True, no_temperature=True))
    for index, job in enumerate(jobs):
        job['job_id'] = f"r3_{index:04d}__{job['dataset']}__{job['family']}__{job.get('arm_tag') or job['arm']}__{job.get('unit')}"
    return jobs
