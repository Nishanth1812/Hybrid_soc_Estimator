from __future__ import annotations

import numpy as np

from config import DEGRADATION_CONFIG, SIM_CONFIG
from .base_simulation import BaseSimulation, SimulationOutput


def run_healthy(
    current_profile_a: np.ndarray,
    ambient_temperature_k: float,
    initial_soc: float,
) -> SimulationOutput:
    simulator = BaseSimulation(
        options=SIM_CONFIG["spme_options"],
        degradation_level=DEGRADATION_CONFIG["levels"][0],
        sampling_period_s=SIM_CONFIG["sampling_period_s"],
    )
    return simulator.run(
        current_profile_a=current_profile_a,
        ambient_temperature_k=ambient_temperature_k,
        initial_soc=initial_soc,
    )
