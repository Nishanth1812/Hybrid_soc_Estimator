from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from config import PREPROCESSING_CONFIG
from data_pipeline.generation.dataset_generator import SimulationRecord


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(signal, kernel, mode="same")


def _smooth_signal(signal: np.ndarray) -> np.ndarray:
    window = int(PREPROCESSING_CONFIG["filter_window"])
    polyorder = int(PREPROCESSING_CONFIG["filter_polyorder"])

    if window % 2 == 0:
        window += 1
    if len(signal) < window or window <= polyorder:
        return _moving_average(signal, max(3, min(9, len(signal) // 2 * 2 + 1)))

    return savgol_filter(signal, window_length=window, polyorder=polyorder, mode="interp")


def apply_filter(record: SimulationRecord) -> SimulationRecord:
    return SimulationRecord(
        simulation_id=record.simulation_id,
        voltage_v=_smooth_signal(record.voltage_v).astype(np.float32),
        current_a=_smooth_signal(record.current_a).astype(np.float32),
        temperature_k=record.temperature_k.astype(np.float32),
        soc=record.soc.astype(np.float32),
        metadata=dict(record.metadata),
    )
