from __future__ import annotations

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import SCALER_PATH, SCALERS_PATH
from data_pipeline.generation.dataset_generator import SimulationRecord


def fit_scaler(train_simulations: list[SimulationRecord]) -> StandardScaler:
    scaler = StandardScaler()
    features = [
        np.column_stack((sim.voltage_v, sim.current_a, sim.temperature_k))
        for sim in train_simulations
    ]
    scaler.fit(np.vstack(features))

    SCALERS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def apply_scaler(
    simulations: list[SimulationRecord],
    scaler: StandardScaler,
) -> list[SimulationRecord]:
    scaled_records: list[SimulationRecord] = []
    for sim in simulations:
        scaled = scaler.transform(np.column_stack((sim.voltage_v, sim.current_a, sim.temperature_k)))
        scaled_records.append(
            SimulationRecord(
                simulation_id=sim.simulation_id,
                voltage_v=scaled[:, 0].astype(np.float32),
                current_a=scaled[:, 1].astype(np.float32),
                temperature_k=scaled[:, 2].astype(np.float32),
                soc=sim.soc.astype(np.float32),
                metadata=dict(sim.metadata),
            )
        )
    return scaled_records