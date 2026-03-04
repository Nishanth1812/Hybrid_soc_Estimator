from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybamm


@dataclass
class SimulationOutput:
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_k: np.ndarray
    soc: np.ndarray


class BaseSimulation:
    def __init__(
        self,
        options: dict[str, Any] | None = None,
        degradation_level: dict[str, Any] | None = None,
        sampling_period_s: int = 1,
    ) -> None:
        self.options = options or {}
        self.degradation_level = degradation_level or {}
        self.sampling_period_s = sampling_period_s
        self.model = pybamm.lithium_ion.SPMe(options=self.options)
        self.parameter_values = self.model.default_parameter_values.copy()

    def _update_parameters(
        self,
        current_profile_a: np.ndarray,
        ambient_temperature_k: float,
        initial_soc: float,
    ) -> None:
        time_s = np.arange(0, len(current_profile_a) * self.sampling_period_s, self.sampling_period_s)
        current_fn = pybamm.Interpolant(time_s, current_profile_a, pybamm.t)
        self.parameter_values.update(
            {
                "Current function [A]": current_fn,
                "Ambient temperature [K]": float(ambient_temperature_k),
                "Initial SoC": float(initial_soc),
                "Lower voltage cut-off [V]": 2.0,
                "Upper voltage cut-off [V]": 4.4,
            },
            check_already_exists=False,
        )

        contact_resistance = float(self.parameter_values["Contact resistance [Ohm]"])
        lli_fraction = float(self.degradation_level.get("lli_fraction", 0.0))
        resistance_multiplier = float(self.degradation_level.get("resistance_multiplier", 1.0))

        if lli_fraction > 0:
            neg_init = float(
                self.parameter_values["Initial concentration in negative electrode [mol.m-3]"]
            )
            self.parameter_values.update(
                {
                    "Initial concentration in negative electrode [mol.m-3]": neg_init
                    * (1.0 - lli_fraction)
                },
                check_already_exists=False,
            )

        self.parameter_values.update(
            {"Contact resistance [Ohm]": contact_resistance * resistance_multiplier},
            check_already_exists=False,
        )

    def _get_temperature_entries(self, solution: pybamm.Solution) -> np.ndarray:
        for key in (
            "X-averaged cell temperature [K]",
            "Cell temperature [K]",
            "Volume-averaged cell temperature [K]",
        ):
            if key in solution.all_models[0].variables:
                return solution[key].entries
        return np.full_like(solution["Time [s]"].entries, np.nan, dtype=np.float64)

    @staticmethod
    def _pad_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) >= target_len:
            return arr[:target_len]
        pad_count = target_len - len(arr)
        return np.pad(arr, (0, pad_count), mode="edge")

    def run(
        self,
        current_profile_a: np.ndarray,
        ambient_temperature_k: float,
        initial_soc: float,
    ) -> SimulationOutput:
        self._update_parameters(
            current_profile_a=current_profile_a,
            ambient_temperature_k=ambient_temperature_k,
            initial_soc=initial_soc,
        )

        t_eval = np.arange(0, len(current_profile_a) * self.sampling_period_s, self.sampling_period_s)
        solver = pybamm.CasadiSolver(mode="safe")
        sim = pybamm.Simulation(
            model=self.model,
            parameter_values=self.parameter_values,
            solver=solver,
        )
        solution = sim.solve(t_eval=t_eval)

        time_s = solution["Time [s]"].entries
        voltage_v = solution["Terminal voltage [V]"].entries
        current_a = solution["Current [A]"].entries
        temperature_k = self._get_temperature_entries(solution)
        discharge_ah = solution["Discharge capacity [A.h]"].entries
        nominal_capacity_ah = float(self.parameter_values["Nominal cell capacity [A.h]"])
        soc = np.clip(initial_soc - (discharge_ah / nominal_capacity_ah), 0.0, 1.0)

        expected_len = len(t_eval)
        if len(time_s) != expected_len:
            time_s = t_eval.astype(np.float64)
            voltage_v = self._pad_to_length(voltage_v, expected_len)
            current_a = self._pad_to_length(current_a, expected_len)
            temperature_k = self._pad_to_length(temperature_k, expected_len)
            soc = self._pad_to_length(soc, expected_len)

        if not (
            np.isfinite(voltage_v).all()
            and np.isfinite(current_a).all()
            and np.isfinite(temperature_k).all()
            and np.isfinite(soc).all()
        ):
            raise RuntimeError("Simulation produced non-finite values")

        return SimulationOutput(
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            temperature_k=temperature_k,
            soc=soc,
        )