from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class AcceptanceProfile:
    name: str
    description: str
    min_landmarks: int = 20
    min_quadrants: int = 4
    max_median_tre_um: float = 2.0
    max_p90_tre_um: float = 4.2
    support_fraction: float = 1.0
    require_positive_jacobian: bool = True
    min_jacobian_p1: float = 0.5
    max_jacobian_p99: float = 2.0
    max_inverse_p90_um: float = 4.2
CMG = AcceptanceProfile(name='CMG', description='Cell-Matching Grade')
RSG = AcceptanceProfile(name='RSG', description='ROI Structural-Grade', max_p90_tre_um=6.0, min_jacobian_p1=0.3)
PROFILES = {profile.name: profile for profile in (CMG, RSG)}
RSG_RELAXED_SOURCE_FAILURES = {'p90_tre_above_gate', 'jacobian_p1_below_gate'}

def finite_number(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf

def gate_reasons(row: Mapping[str, Any], profile: AcceptanceProfile) -> list[str]:
    reasons: list[str] = []
    if finite_number(row, 'heldout_landmark_count') < profile.min_landmarks:
        reasons.append('insufficient_heldout_landmarks')
    if finite_number(row, 'landmark_quadrants_covered') < profile.min_quadrants:
        reasons.append('heldout_landmarks_do_not_cover_four_quadrants')
    if finite_number(row, 'median_tre_um') > profile.max_median_tre_um:
        reasons.append('median_tre_above_gate')
    if finite_number(row, 'p90_tre_um') > profile.max_p90_tre_um:
        reasons.append('p90_tre_above_gate')
    if finite_number(row, 'support_fraction') != profile.support_fraction:
        reasons.append('incomplete_source_support')
    jacobian_min = finite_number(row, 'jacobian_min')
    if profile.require_positive_jacobian and (not jacobian_min > 0.0):
        reasons.append('nonpositive_jacobian')
    if finite_number(row, 'jacobian_p1') < profile.min_jacobian_p1:
        reasons.append('jacobian_p1_below_gate')
    if finite_number(row, 'jacobian_p99') > profile.max_jacobian_p99:
        reasons.append('jacobian_p99_above_gate')
    if finite_number(row, 'inverse_consistency_p90_um') > profile.max_inverse_p90_um:
        reasons.append('forward_backward_inverse_gate_failed')
    return reasons

def passes(row: Mapping[str, Any], profile: AcceptanceProfile) -> bool:
    return not gate_reasons(row, profile)

def source_failure_reasons(row: Mapping[str, Any]) -> list[str]:
    return [reason for reason in str(row.get('failure_reasons', '')).split(';') if reason]

def candidate_gate_reasons(row: Mapping[str, Any], profile: AcceptanceProfile) -> list[str]:
    reasons = gate_reasons(row, profile)
    status = str(row.get('status', '')).lower()
    source_reasons = source_failure_reasons(row)
    if profile.name == CMG.name:
        if status != 'pass':
            reasons.append('source_status_not_pass')
        reasons.extend((f'source_failure_not_allowed:{reason}' for reason in source_reasons))
    elif profile.name == RSG.name:
        reasons.extend((f'source_failure_not_relaxed:{reason}' for reason in source_reasons if reason not in RSG_RELAXED_SOURCE_FAILURES))
        if status != 'pass' and (not source_reasons):
            reasons.append('source_status_not_pass_without_reason')
    else:
        raise ValueError(f'unsupported acceptance profile: {profile.name}')
    return list(dict.fromkeys(reasons))

def candidate_passes(row: Mapping[str, Any], profile: AcceptanceProfile) -> bool:
    return not candidate_gate_reasons(row, profile)

def profile_payload(profile: AcceptanceProfile) -> dict[str, Any]:
    return {'name': profile.name, 'description': profile.description, 'min_landmarks': profile.min_landmarks, 'min_quadrants': profile.min_quadrants, 'max_median_tre_um': profile.max_median_tre_um, 'max_p90_tre_um': profile.max_p90_tre_um, 'support_fraction': profile.support_fraction, 'require_positive_jacobian': profile.require_positive_jacobian, 'min_jacobian_p1': profile.min_jacobian_p1, 'max_jacobian_p99': profile.max_jacobian_p99, 'max_inverse_p90_um': profile.max_inverse_p90_um}
