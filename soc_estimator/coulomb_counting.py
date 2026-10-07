from __future__ import annotations

import numpy as np


def initial_soc_from_cycle_type(cycle_types) -> np.ndarray:
    """Return the Mendeley test protocol's starting SOC for each cycle type."""
    anchors = {"charge": 0.0, "discharge": 1.0}
    try:
        return np.asarray([anchors[str(value).strip().lower()] for value in cycle_types])
    except KeyError as error:
        raise ValueError(f"Unsupported Mendeley cycle type: {error.args[0]}") from error


def fit_capacity_ah(cumulative_ah, soc, initial_soc) -> float:
    """Fit capacity from training SOC labels and measured cumulative charge."""
    charge = np.asarray(cumulative_ah, dtype=np.float64).reshape(-1)
    target = np.asarray(soc, dtype=np.float64).reshape(-1)
    initial = np.asarray(initial_soc, dtype=np.float64).reshape(-1)
    if charge.size == 0 or charge.size != target.size or charge.size != initial.size:
        raise ValueError("Charge, SOC, and initial SOC must have the same non-zero length")
    if (
        not np.isfinite(charge).all()
        or not np.isfinite(target).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Capacity fitting inputs must be finite")
    if np.any((target < 0) | (target > 1)) or np.any((initial < 0) | (initial > 1)):
        raise ValueError("SOC values must be fractions in [0, 1]")

    denominator = np.dot(charge, target - initial)
    if denominator <= 0:
        raise ValueError("Training data does not identify a positive battery capacity")
    return float(np.dot(charge, charge) / denominator)


def estimate_soc_from_cumulative_ah(
    cumulative_ah, initial_soc, capacity_ah: float
) -> np.ndarray:
    """Estimate SOC from net amp-hours since a known SOC starting point."""
    charge = np.asarray(cumulative_ah, dtype=np.float64)
    initial = np.asarray(initial_soc, dtype=np.float64)
    capacity = float(capacity_ah)
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("capacity_ah must be finite and positive")
    if not np.isfinite(charge).all() or not np.isfinite(initial).all():
        raise ValueError("Charge and initial SOC inputs must be finite")
    if np.any((initial < 0) | (initial > 1)):
        raise ValueError("initial_soc values must be fractions in [0, 1]")
    if initial.ndim > 0 and initial.shape != charge.shape:
        raise ValueError("initial_soc must be scalar or match the charge array shape")
    try:
        return np.clip(initial + charge / capacity, 0.0, 1.0)
    except ValueError as error:
        raise ValueError("initial_soc must be scalar or match the charge array shape") from error
