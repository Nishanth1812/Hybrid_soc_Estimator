from __future__ import annotations

import numpy as np


def _validated_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if true.size == 0 or pred.size == 0:
        raise ValueError("Metric inputs must not be empty")
    if true.size != pred.size:
        raise ValueError("Metric inputs must contain the same number of values")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must contain only finite values")

    return true, pred


def _error(y_true, y_pred) -> np.ndarray:
    true, pred = _validated_arrays(y_true, y_pred)
    return pred - true


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_error(y_true, y_pred))))


def rmse(y_true, y_pred) -> float:
    error = _error(y_true, y_pred)
    return float(np.sqrt(np.mean(error**2)))


def max_error(y_true, y_pred) -> float:
    return float(np.max(np.abs(_error(y_true, y_pred))))


def r2(y_true, y_pred) -> float:
    true, pred = _validated_arrays(y_true, y_pred)
    total_sum_of_squares = np.sum((true - np.mean(true)) ** 2)
    if np.isclose(total_sum_of_squares, 0.0):
        return 0.0

    residual_sum_of_squares = np.sum((true - pred) ** 2)
    return float(1.0 - residual_sum_of_squares / total_sum_of_squares)


def mean_bias(y_true, y_pred) -> float:
    return float(np.mean(_error(y_true, y_pred)))


def error_std(y_true, y_pred) -> float:
    return float(np.std(_error(y_true, y_pred)))


def p95_abs_error(y_true, y_pred) -> float:
    return float(np.percentile(np.abs(_error(y_true, y_pred)), 95))


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the scalar evaluation report for reference and predicted SOC."""
    true, pred = _validated_arrays(y_true, y_pred)
    error = pred - true
    absolute_error = np.abs(error)
    total_sum_of_squares = np.sum((true - np.mean(true)) ** 2)
    r2_value = 0.0
    if not np.isclose(total_sum_of_squares, 0.0):
        r2_value = 1.0 - np.sum((true - pred) ** 2) / total_sum_of_squares

    return {
        "MAE": float(np.mean(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": float(r2_value),
        "MaxError": float(np.max(absolute_error)),
        "MeanBias": float(np.mean(error)),
        "ErrorStd": float(np.std(error)),
        "P95AbsError": float(np.percentile(absolute_error, 95)),
    }
