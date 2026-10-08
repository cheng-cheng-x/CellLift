from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
from typing import Final, Literal
import numpy as np
TEMPERATURE_BOUNDS: Final = (0.25, 4.0)
ALPHA_BOUNDS: Final = (0.0, 4.0)
OPTIMIZER_MAX_ITERATIONS: Final = 1000
OPTIMIZER_TOLERANCE: Final = 1e-10
PROBABILITY_EPSILON: Final = 1e-12
FUSION_ARM_TO_EXPERT: Final[dict[str, str]] = {'A0': 'E0', 'A1': 'E1', 'A2': 'E2', 'A3': 'E3', 'A4': 'E4', 'A5': 'E5'}
GEOMETRY_FUSION_ARMS: Final = tuple((arm for arm in FUSION_ARM_TO_EXPERT if arm != 'A0'))

@dataclass(frozen=True)
class TemperatureFit:
    value: float
    success: bool
    nll: float

    @property
    def boundary_hit(self) -> bool:
        return any((abs(self.value - bound) <= 1e-06 for bound in TEMPERATURE_BOUNDS))

@dataclass(frozen=True)
class ImageBaselineFit:
    base_temperature: TemperatureFit
    geometry_temperature: TemperatureFit
    alpha: np.ndarray
    alpha_success: bool
    fused_nll: float

    @property
    def alpha_boundary_hits(self) -> np.ndarray:
        return np.logical_or(np.isclose(self.alpha, ALPHA_BOUNDS[0], atol=1e-06), np.isclose(self.alpha, ALPHA_BOUNDS[1], atol=1e-06))

@dataclass(frozen=True)
class ScalarFusionFit:
    alpha: float
    success: bool
    fused_nll: float

    @property
    def boundary_hit(self) -> bool:
        return any((abs(self.alpha - bound) <= 1e-06 for bound in ALPHA_BOUNDS))

def expert_for_arm(arm: str) -> str:
    try:
        return FUSION_ARM_TO_EXPERT[str(arm)]
    except KeyError as exc:
        raise ValueError(f'unknown BRACS fusion arm: {arm!r}') from exc

def centered_log_probabilities(probabilities: object) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError('probabilities must be a non-empty N x K matrix with K >= 2')
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError('probabilities must be finite and lie in [0, 1]')
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-06, atol=1e-08):
        raise ValueError('probability rows must sum to one')
    logged = np.log(np.clip(values, PROBABILITY_EPSILON, 1.0))
    return logged - logged.mean(axis=1, keepdims=True)

def softmax_probabilities(logits: object) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or (not np.all(np.isfinite(values))):
        raise ValueError('logits must be a finite N x K matrix with K >= 2')
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    output = exponent / exponent.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(output)):
        raise RuntimeError('softmax produced non-finite probabilities')
    return output

def _validated_training_data(y_train: object, base_train_probabilities: object, geometry_train_probabilities: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_train, dtype=np.int64)
    base = centered_log_probabilities(base_train_probabilities)
    geometry = centered_log_probabilities(geometry_train_probabilities)
    if truth.ndim != 1 or len(truth) != len(base) or geometry.shape != base.shape:
        raise ValueError('training labels and expert predictions must be sample aligned')
    if np.any(truth < 0) or np.any(truth >= base.shape[1]):
        raise ValueError('training labels are outside the expert class range')
    return (truth, base, geometry)

def _nll(truth: np.ndarray, logits: np.ndarray) -> float:
    probabilities = softmax_probabilities(logits)
    selected = probabilities[np.arange(len(truth)), truth]
    return float(-np.log(np.clip(selected, PROBABILITY_EPSILON, 1.0)).mean())

def _fit_temperature(truth: np.ndarray, logits: np.ndarray) -> TemperatureFit:
    from scipy.optimize import minimize
    result = minimize(lambda value: _nll(truth, logits / float(value[0])), x0=np.asarray([1.0], dtype=np.float64), method='L-BFGS-B', bounds=[TEMPERATURE_BOUNDS], options={'ftol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    temperature = float(result.x[0])
    if not np.isfinite(temperature) or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]:
        raise RuntimeError('temperature optimizer returned an invalid value')
    return TemperatureFit(temperature, bool(result.success), float(result.fun))

def fit_breast_roi(y_train: object, base_train_probabilities: object, geometry_train_probabilities: object) -> ImageBaselineFit:
    from scipy.optimize import minimize
    truth, base_logits, geometry_logits = _validated_training_data(y_train, base_train_probabilities, geometry_train_probabilities)
    base_temperature = _fit_temperature(truth, base_logits)
    geometry_temperature = _fit_temperature(truth, geometry_logits)
    calibrated_base = base_logits / base_temperature.value
    calibrated_geometry = geometry_logits / geometry_temperature.value
    class_count = base_logits.shape[1]

    def objective(alpha: np.ndarray) -> float:
        return _nll(truth, calibrated_base + calibrated_geometry * alpha[None, :])
    result = minimize(objective, x0=np.full(class_count, 0.25, dtype=np.float64), method='L-BFGS-B', bounds=[ALPHA_BOUNDS] * class_count, options={'ftol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    alpha = np.asarray(result.x, dtype=np.float64)
    if alpha.shape != (class_count,) or not np.all(np.isfinite(alpha)):
        raise RuntimeError('breast_roi optimizer returned an invalid class-wise alpha')
    if np.any(alpha < ALPHA_BOUNDS[0]) or np.any(alpha > ALPHA_BOUNDS[1]):
        raise RuntimeError('breast_roi alpha is outside the registered bounds')
    return ImageBaselineFit(base_temperature=base_temperature, geometry_temperature=geometry_temperature, alpha=alpha, alpha_success=bool(result.success), fused_nll=float(result.fun))

def apply_breast_roi(base_probabilities: object, geometry_probabilities: object, fitted: ImageBaselineFit) -> np.ndarray:
    base_logits = centered_log_probabilities(base_probabilities)
    geometry_logits = centered_log_probabilities(geometry_probabilities)
    if base_logits.shape != geometry_logits.shape:
        raise ValueError('base and geometry predictions must have equal shape')
    if fitted.alpha.shape != (base_logits.shape[1],):
        raise ValueError('fitted breast_roi alpha does not match the prediction class count')
    logits = base_logits / fitted.base_temperature.value + geometry_logits / fitted.geometry_temperature.value * fitted.alpha[None, :]
    return softmax_probabilities(logits)

def fit_scalar_fusion(y_train: object, base_train_probabilities: object, geometry_train_probabilities: object) -> ScalarFusionFit:
    from scipy.optimize import minimize
    truth, base_logits, geometry_logits = _validated_training_data(y_train, base_train_probabilities, geometry_train_probabilities)

    def objective(alpha: np.ndarray) -> float:
        return _nll(truth, base_logits + float(alpha[0]) * geometry_logits)
    result = minimize(objective, x0=np.asarray([0.25], dtype=np.float64), method='L-BFGS-B', bounds=[ALPHA_BOUNDS], options={'ftol': OPTIMIZER_TOLERANCE, 'maxiter': OPTIMIZER_MAX_ITERATIONS})
    alpha = float(result.x[0])
    if not np.isfinite(alpha) or not ALPHA_BOUNDS[0] <= alpha <= ALPHA_BOUNDS[1]:
        raise RuntimeError('scalar_fusion optimizer returned an invalid alpha')
    return ScalarFusionFit(alpha=alpha, success=bool(result.success), fused_nll=float(result.fun))

def apply_scalar_fusion(base_probabilities: object, geometry_probabilities: object, fitted: ScalarFusionFit) -> np.ndarray:
    base_logits = centered_log_probabilities(base_probabilities)
    geometry_logits = centered_log_probabilities(geometry_probabilities)
    if base_logits.shape != geometry_logits.shape:
        raise ValueError('base and geometry predictions must have equal shape')
    if not ALPHA_BOUNDS[0] <= fitted.alpha <= ALPHA_BOUNDS[1]:
        raise ValueError('fitted scalar_fusion alpha is outside the registered bounds')
    return softmax_probabilities(base_logits + fitted.alpha * geometry_logits)

def fit_registered_arm(arm: str, protocol: Literal['breast_roi', 'scalar_fusion'], y_train: object, base_train_probabilities: object, geometry_train_probabilities: object | None=None) -> TemperatureFit | ImageBaselineFit | ScalarFusionFit:
    expert_for_arm(arm)
    if arm == 'A0':
        logits = centered_log_probabilities(base_train_probabilities)
        truth = np.asarray(y_train, dtype=np.int64)
        if truth.ndim != 1 or len(truth) != len(logits):
            raise ValueError('training labels and E0 predictions must be sample aligned')
        if protocol == 'breast_roi':
            return _fit_temperature(truth, logits)
        if protocol == 'scalar_fusion':
            return ScalarFusionFit(alpha=0.0, success=True, fused_nll=_nll(truth, logits))
        raise ValueError(f'unknown fusion protocol: {protocol!r}')
    if geometry_train_probabilities is None:
        raise ValueError(f'{arm} requires geometry expert predictions')
    if protocol == 'breast_roi':
        return fit_breast_roi(y_train, base_train_probabilities, geometry_train_probabilities)
    if protocol == 'scalar_fusion':
        return fit_scalar_fusion(y_train, base_train_probabilities, geometry_train_probabilities)
    raise ValueError(f'unknown fusion protocol: {protocol!r}')

def apply_registered_arm(arm: str, protocol: Literal['breast_roi', 'scalar_fusion'], base_probabilities: object, fitted: TemperatureFit | ImageBaselineFit | ScalarFusionFit, geometry_probabilities: object | None=None) -> np.ndarray:
    expert_for_arm(arm)
    if arm == 'A0':
        logits = centered_log_probabilities(base_probabilities)
        if protocol == 'breast_roi':
            if not isinstance(fitted, TemperatureFit):
                raise TypeError('breast_roi A0 requires a TemperatureFit')
            return softmax_probabilities(logits / fitted.value)
        if protocol == 'scalar_fusion':
            return softmax_probabilities(logits)
        raise ValueError(f'unknown fusion protocol: {protocol!r}')
    if geometry_probabilities is None:
        raise ValueError(f'{arm} requires geometry expert predictions')
    if protocol == 'breast_roi':
        if not isinstance(fitted, ImageBaselineFit):
            raise TypeError('breast_roi geometry arms require a ImageBaselineFit')
        return apply_breast_roi(base_probabilities, geometry_probabilities, fitted)
    if protocol == 'scalar_fusion':
        if not isinstance(fitted, ScalarFusionFit):
            raise TypeError('scalar_fusion geometry arms require a ScalarFusionFit')
        return apply_scalar_fusion(base_probabilities, geometry_probabilities, fitted)
    raise ValueError(f'unknown fusion protocol: {protocol!r}')
