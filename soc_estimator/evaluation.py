from __future__ import annotations

import numpy as np
import torch

from .model import LSTMSOCEstimator
from .data import FORECAST_FEATURES, scale_sensor_features


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


def calculate_grouped_metrics(y_true, y_pred, window_metadata) -> dict:
    """Report global and condition-specific metrics for aligned test windows."""
    true, pred = _validated_arrays(y_true, y_pred)
    rows = (
        window_metadata.to_dict(orient="records")
        if hasattr(window_metadata, "to_dict")
        else list(window_metadata)
    )
    if len(rows) != true.size:
        raise ValueError("Window metadata must contain one row per prediction")

    def grouped(labels) -> dict[str, dict[str, float]]:
        indices_by_label: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            indices_by_label.setdefault(str(label), []).append(index)
        return {
            label: calculate_metrics(true[indices], pred[indices])
            for label, indices in sorted(indices_by_label.items())
        }

    cycle_numbers = [row.get("cycle_number", "unknown") for row in rows]
    cycle_types = [row.get("cycle_type", "unknown") for row in rows]
    current_direction = []
    temperature_band = []
    interval_band = []
    for row in rows:
        try:
            current = float(row["mean_current_a"])
            if not np.isfinite(current):
                raise ValueError
            current_direction.append("positive" if current > 0 else "negative" if current < 0 else "zero")
        except (KeyError, TypeError, ValueError):
            current_direction.append("unknown")
        try:
            temperature = float(row["mean_ambient_temperature_c"])
            if not np.isfinite(temperature):
                raise ValueError
            lower = np.floor(temperature / 5.0) * 5.0
            temperature_band.append(f"{lower:g}-{lower + 5:g} C")
        except (KeyError, TypeError, ValueError):
            temperature_band.append("unknown")
        try:
            interval = float(row["mean_interval_s"])
            if not np.isfinite(interval):
                raise ValueError
            if interval <= 0:
                interval_band.append("non_positive")
            else:
                lower = np.floor(interval / 5.0) * 5.0
                interval_band.append(f"{lower:g}-{lower + 5:g} s")
        except (KeyError, TypeError, ValueError):
            interval_band.append("unknown")

    soc_band = []
    for value in true:
        if value < 0:
            soc_band.append("below 0%")
        elif value >= 1:
            soc_band.append("90-100%")
        else:
            lower = int(value * 10) * 10
            soc_band.append(f"{lower}-{lower + 10}%")
    by_cycle = grouped(cycle_numbers)
    cycle_mae = {cycle: metrics["MAE"] for cycle, metrics in by_cycle.items()}
    worst_cycle = max(cycle_mae, key=cycle_mae.get)
    return {
        "overall": calculate_metrics(true, pred),
        "by_cycle": by_cycle,
        "cycle_summary": {
            "macro_MAE": float(np.mean(list(cycle_mae.values()))),
            "worst_cycle": worst_cycle,
            "worst_cycle_MAE": float(cycle_mae[worst_cycle]),
        },
        "by_cycle_type": grouped(cycle_types),
        "by_soc_band": grouped(soc_band),
        "by_current_direction": grouped(current_direction),
        "by_ambient_temperature_band_c": grouped(temperature_band),
        "by_mean_interval_band_s": grouped(interval_band),
    }


def _load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    input_features = state_dict["lstm1.weight_ih_l0"].shape[1]
    model_config = checkpoint.get("model_config", {})
    model = LSTMSOCEstimator(
        input_features=input_features,
        current_soc_feature_index=model_config.get("current_soc_feature_index"),
        max_soc_correction=model_config.get("max_soc_correction", 0.03),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def forecast(model_path, history, current_soc, device="cpu") -> dict[str, float]:
    """Predict from sensor history and a live current SOC fraction."""
    model, checkpoint = _load_model(model_path, torch.device(device))
    metadata = checkpoint.get("dataset_metadata", {})
    if metadata.get("task") != "future_soc_forecast" or metadata.get("features") != FORECAST_FEATURES:
        raise ValueError("This checkpoint does not contain a future SOC forecast contract with current SOC")
    if not np.isscalar(current_soc) or not np.isfinite(current_soc) or not 0 <= current_soc <= 1:
        raise ValueError("Current SOC estimate must be a finite fraction between zero and one")
    values = np.asarray(history, dtype=np.float64)
    length = metadata["sequence_length"]
    if values.ndim != 2 or values.shape != (length, 4) or not np.isfinite(values).all():
        raise ValueError(f"History must contain exactly {length} finite [time, voltage, current, temperature C] rows")
    intervals = np.diff(values[:, 0])
    if np.any(intervals <= 0) or np.any(intervals > metadata["max_gap_seconds"]):
        raise ValueError("History timestamps must increase without a measurement gap beyond the trained limit")
    features = np.column_stack((values[:, 1], values[:, 2], values[:, 3] + 273.15, np.r_[0.0, intervals]))
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    if mean.shape != (4,) or scale.shape != (4,) or not np.isfinite([mean, scale]).all() or np.any(scale <= 0):
        raise ValueError("Checkpoint contains invalid input scaling")
    scaled_features = scale_sensor_features(features, mean, scale)
    model_features = np.column_stack((scaled_features, np.full(length, current_soc, dtype=np.float32)))
    inputs = torch.from_numpy(model_features).to(device)[None]
    with torch.no_grad():
        soc = model(inputs).item()
    return {"soc_fraction": soc, "soc_percent": soc * 100.0,
            "target_time_s": float(values[-1, 0] + metadata["forecast_horizon_seconds"])}


def predict(model_path, test_loader, device):
    model, _ = _load_model(model_path, device)
    preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            X, y = batch[:2]
            X = X.to(device)
            pred = model(X)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    if not targets:
        raise ValueError("Cannot predict an empty dataset")
    return np.vstack(targets), np.vstack(preds)


def evaluate(model_path, test_loader, device):
    targets, preds = predict(model_path, test_loader, device)
    return calculate_metrics(targets, preds)
