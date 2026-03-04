from __future__ import annotations

import numpy as np

from config import DEGRADATION_CONFIG, SIM_CONFIG
from .base_simulation import BaseSimulation, SimulationOutput


def _build_degraded_options() -> dict[str, str]:
    options = dict(SIM_CONFIG["spme_options"])
    options["SEI"] = "reaction limited"
    return options


def run_degraded(
    current_profile_a: np.ndarray,
    ambient_temperature_k: float,
    initial_soc: float,
    degradation_level_id: int,
) -> SimulationOutput:
    if degradation_level_id not in DEGRADATION_CONFIG["levels"]:
        raise ValueError(f"Invalid degradation_level_id: {degradation_level_id}")
    if degradation_level_id == 0:
        raise ValueError("degradation_level_id must be >= 1 for degraded simulations")

    simulator = BaseSimulation(
        options=_build_degraded_options(),
        degradation_level=DEGRADATION_CONFIG["levels"][degradation_level_id],
        sampling_period_s=SIM_CONFIG["sampling_period_s"],
    )
    return simulator.run(
        current_profile_a=current_profile_a,
        ambient_temperature_k=ambient_temperature_k,
        initial_soc=initial_soc,
    )
