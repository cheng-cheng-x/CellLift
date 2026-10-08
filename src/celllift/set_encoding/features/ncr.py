from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NCRAudit:
    nucleus_measure: np.ndarray
    cell_measure: np.ndarray
    cytoplasm_measure: np.ndarray
    raw_log_ncr: np.ndarray
    valid: np.ndarray
    reason: np.ndarray

@dataclass(frozen=True)
class NCRStatistics:
    median: float
    mean: float
    std: float
    valid_count: int
    invalid_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {'median': self.median, 'mean': self.mean, 'std': self.std, 'valid_count': self.valid_count, 'invalid_count': self.invalid_count}

@dataclass(frozen=True)
class NCRFeatures:
    standardized_log_ncr: np.ndarray
    invalid_flag: np.ndarray
    imputed_log_ncr: np.ndarray

    def channels(self) -> np.ndarray:
        return np.column_stack((self.standardized_log_ncr, self.invalid_flag)).astype(np.float32)

def compute_log_ncr(nucleus_measure: np.ndarray, cell_measure: np.ndarray) -> NCRAudit:
    nucleus = np.asarray(nucleus_measure, dtype=np.float64)
    cell = np.asarray(cell_measure, dtype=np.float64)
    if nucleus.shape != cell.shape:
        raise ValueError('nucleus and cell measures must have identical shapes')
    if nucleus.ndim != 1:
        raise ValueError('nucleus and cell measures must be one-dimensional')
    cytoplasm = cell - nucleus
    finite = np.isfinite(nucleus) & np.isfinite(cell)
    valid = finite & (nucleus > 0) & (cell > 0) & (cytoplasm > 0)
    reason = np.full(nucleus.shape, 'valid', dtype='U24')
    reason[~finite] = 'nonfinite'
    reason[finite & (nucleus <= 0)] = 'nonpositive_nucleus'
    reason[finite & (nucleus > 0) & (cell <= 0)] = 'nonpositive_cell'
    reason[finite & (nucleus > 0) & (cell > 0) & (cytoplasm <= 0)] = 'nonpositive_cytoplasm'
    raw = np.full(nucleus.shape, np.nan, dtype=np.float64)
    raw[valid] = np.log(nucleus[valid]) - np.log(cytoplasm[valid])
    return NCRAudit(nucleus, cell, cytoplasm, raw, valid, reason)

def fit_ncr_statistics(training_audit: NCRAudit, *, minimum_std: float=1e-08) -> NCRStatistics:
    values = np.asarray(training_audit.raw_log_ncr, dtype=np.float64)[training_audit.valid]
    if values.size == 0:
        raise ValueError('cannot fit NCR statistics without a valid training-fold value')
    if not np.isfinite(values).all():
        raise ValueError('valid NCR values must be finite')
    std = float(values.std(ddof=0))
    if std < minimum_std:
        std = 1.0
    return NCRStatistics(median=float(np.median(values)), mean=float(values.mean()), std=std, valid_count=int(values.size), invalid_count=int((~training_audit.valid).sum()))

def transform_ncr(audit: NCRAudit, statistics: NCRStatistics) -> NCRFeatures:
    if not np.isfinite([statistics.median, statistics.mean, statistics.std]).all() or statistics.std <= 0:
        raise ValueError('invalid NCR statistics')
    raw = np.asarray(audit.raw_log_ncr, dtype=np.float64)
    imputed = np.where(audit.valid, raw, statistics.median)
    standardized = (imputed - statistics.mean) / statistics.std
    return NCRFeatures(standardized_log_ncr=standardized.astype(np.float32), invalid_flag=(~audit.valid).astype(np.float32), imputed_log_ncr=imputed)

def fit_transform_training_ncr(training_audit: NCRAudit) -> tuple[NCRStatistics, NCRFeatures]:
    statistics = fit_ncr_statistics(training_audit)
    return (statistics, transform_ncr(training_audit, statistics))
