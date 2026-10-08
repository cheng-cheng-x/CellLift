from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import asdict, dataclass
import hashlib
from celllift.runtime import json
from celllift.runtime import ResourcePath as Path
from typing import Any, Mapping
PROTOCOL_ID = 'paper_rgb_decoupled_mask2d_3d_full_factorial_geometry_baselines'
TOKEN_DIM = 41
MASK_DIM = 36
THREE_D_DIM = 5
SEEDS = (17, 42, 73, 101, 137)
ENCODERS = ('meanpool', 'deepsets')
DATASETS = ('sicapv2', 'tcga_crc_msi')
RGB_MODES = ('none', 'paper')
MASK_MODES = ('none', 'real', 'shuffled')
THREE_D_MODES = ('none', 'raw', 'shuffled_raw', 'residual', 'shuffled_residual')
FUSION_LEVELS = ('geometry_only', 'patch', 'patient', 'tile')

@dataclass(frozen=True)
class Arm:
    arm_id: str
    rgb_mode: str
    mask_mode: str
    three_d_mode: str
    geometry_id: str | None

    @property
    def use_rgb(self) -> bool:
        return self.rgb_mode == 'paper'

    @property
    def object_mode(self) -> str:
        return 'rgb' if self.geometry_id is None else 'both'

    @property
    def control_mode(self) -> str:
        return 'shuffled' if self.mask_mode == 'shuffled' or self.three_d_mode.startswith('shuffled_') else 'none'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
_GEOMETRY = {'G1': ('real', 'none'), 'G1S': ('shuffled', 'none'), 'G2': ('none', 'raw'), 'G2S': ('none', 'shuffled_raw'), 'G3': ('real', 'raw'), 'G3S': ('real', 'shuffled_raw'), 'G4': ('none', 'residual'), 'G4S': ('none', 'shuffled_residual'), 'G5': ('real', 'residual'), 'G5S': ('real', 'shuffled_residual')}
ARMS: dict[str, Arm] = {'R0': Arm('R0', 'paper', 'none', 'none', None)}
for prefix, rgb in (('O', 'none'), ('R', 'paper')):
    for index, geometry_id in enumerate(_GEOMETRY, start=1):
        base = geometry_id[1:]
        arm_id = f'{prefix}{base}'
        mask, three_d = _GEOMETRY[geometry_id]
        ARMS[arm_id] = Arm(arm_id, rgb, mask, three_d, geometry_id)
GEOMETRY_ARMS = tuple((key for key in ARMS if key.startswith('O')))
RGB_GEOMETRY_ARMS = tuple((key for key in ARMS if key.startswith('R') and key != 'R0'))
if len(ARMS) != 21:
    raise AssertionError(f'geometry_baselines must contain exactly 21 arms, found {len(ARMS)}')
PRIMARY_HOLM = ('R5-R1', 'R5-R5S', 'R4-R0', 'R4-R4S')
DIRECT_3D_VS_2D_HOLM = ('R4-R1', 'R2-R1', 'O4-O1', 'O2-O1')

def validate_arm(arm: Arm, *, fusion_level: str | None=None) -> None:
    if arm.arm_id not in ARMS or ARMS[arm.arm_id] != arm:
        raise ValueError(f'unregistered geometry_baselines arm: {arm}')
    if arm.rgb_mode not in RGB_MODES or arm.mask_mode not in MASK_MODES:
        raise ValueError('invalid RGB or MASK mode')
    if arm.three_d_mode not in THREE_D_MODES:
        raise ValueError('invalid 3D mode')
    if arm.arm_id == 'R0' and arm.geometry_id is not None:
        raise ValueError('R0 cannot contain geometry')
    if arm.arm_id != 'R0' and arm.geometry_id is None:
        raise ValueError('non-R0 arms require one registered geometry expert')
    if fusion_level is not None:
        if fusion_level not in FUSION_LEVELS:
            raise ValueError(f'invalid fusion level {fusion_level!r}')
        if arm.rgb_mode == 'none' and fusion_level != 'geometry_only':
            raise ValueError('geometry-only arms require fusion_level=geometry_only')
        if arm.rgb_mode == 'paper' and arm.arm_id != 'R0' and (fusion_level == 'geometry_only'):
            raise ValueError('RGB+geometry arm requires patch/patient/tile fusion')

def validate_runtime_config(cfg: Mapping[str, Any]) -> Arm:
    dataset = str(cfg.get('dataset'))
    if dataset not in DATASETS:
        raise ValueError(f'unsupported dataset {dataset!r}')
    if str(cfg.get('set_encoder')) not in ENCODERS:
        raise ValueError('geometry_baselines permits meanpool or deepsets; Set Transformer is stopped')
    if int(cfg.get('seed', -1)) not in SEEDS:
        raise ValueError('seed is outside the frozen geometry_baselines registry')
    values = (str(cfg.get('rgb_mode')), str(cfg.get('mask_mode')), str(cfg.get('three_d_mode')))
    if values == ('none', 'none', 'none'):
        raise ValueError('empty RGB/MASK2D/3D input is illegal')
    matches = [arm for arm in ARMS.values() if (arm.rgb_mode, arm.mask_mode, arm.three_d_mode) == values]
    if len(matches) != 1:
        raise ValueError(f'configuration is not one of the 21 registered geometry_baselines arms: {values}')
    arm = matches[0]
    validate_arm(arm, fusion_level=str(cfg.get('fusion_level')))
    official = cfg.get('official_test', False)
    if official not in (False, 'gated'):
        raise ValueError("official_test must be false or 'gated'")
    return arm

def config_sha256(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def require_test_gate(result_root: str | Path) -> dict[str, Any]:
    path = Path(result_root) / 'selection_frozen.json'
    if not path.is_file():
        raise RuntimeError('official TEST is locked: selection_frozen.json is absent')
    gate = json.loads(path.read_text(encoding='utf-8'))
    required = {'status': 'PASS', 'protocol_id': PROTOCOL_ID, 'validation_complete': True, 'cross_dataset_validation_complete': True, 'paper_rgb_gates_passed': True, 'arms': list(ARMS), 'encoders': list(ENCODERS), 'seeds': list(SEEDS)}
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise RuntimeError(f'official TEST gate mismatch for {key}: {gate.get(key)!r}')
    return gate
