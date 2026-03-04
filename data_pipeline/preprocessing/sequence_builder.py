from __future__ import annotations

import numpy as np

from config import PREPROCESSING_CONFIG
from data_pipeline.generation.dataset_generator import SimulationRecord


def _num_sequences_for_len(signal_len: int, seq_len: int, stride: int) -> int:
    if signal_len < seq_len:
        return 0
    return ((signal_len - seq_len) // stride) + 1


def build_sequences(simulations: list[SimulationRecord]) -> tuple[np.ndarray, np.ndarray]:
    seq_len = int(PREPROCESSING_CONFIG["sequence_length"])
    stride = int(PREPROCESSING_CONFIG["stride"])

    total_sequences = 0
    for sim in simulations:
        total_sequences += _num_sequences_for_len(len(sim.soc), seq_len, stride)

    X = np.empty((total_sequences, seq_len, 3), dtype=np.float32)
    y = np.empty((total_sequences,), dtype=np.float32)

    cursor = 0
    for sim in simulations:
        feature_mat = np.column_stack((sim.voltage_v, sim.current_a, sim.temperature_k)).astype(np.float32)
        soc = sim.soc.astype(np.float32)
        n_seq = _num_sequences_for_len(len(soc), seq_len, stride)
        for n in range(n_seq):
            start = n * stride
            end = start + seq_len
            X[cursor] = feature_mat[start:end]
            y[cursor] = soc[end - 1]
            cursor += 1

    return X, y
