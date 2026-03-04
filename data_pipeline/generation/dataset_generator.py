from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import DEGRADATION_CONFIG, RAW_METADATA_PATH, RAW_PATH, SIM_CONFIG
from ..simulation.degraded_simulation import run_degraded
from ..simulation.healthy_simulation import run_healthy
from .generate_profiles import generate_current_profile


@dataclass
class SimulationRecord:
    simulation_id: str
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_k: np.ndarray
    soc: np.ndarray
    metadata: dict


def _temperature_levels_k() -> list[float]:
    return [c + 273.15 for c in SIM_CONFIG["temperature_levels_c"]]


def _build_degraded_level_schedule(num_degraded: int) -> list[int]:
    levels = [lvl for lvl in DEGRADATION_CONFIG["levels"].keys() if lvl > 0]
    repeated = (levels * ((num_degraded // len(levels)) + 1))[:num_degraded]
    return repeated


def _degraded_current_multiplier(level: int) -> float:
    return {1: 0.85, 2: 0.72, 3: 0.60}.get(level, 0.8)


def _save_raw_simulation(record: SimulationRecord, raw_dir: Path) -> None:
    out_path = raw_dir / f"{record.simulation_id}.npz"
    np.savez_compressed(
        out_path,
        voltage_v=record.voltage_v.astype(np.float32),
        current_a=record.current_a.astype(np.float32),
        temperature_k=record.temperature_k.astype(np.float32),
        soc=record.soc.astype(np.float32),
    )


def _write_metadata(records: list[SimulationRecord]) -> None:
    header = [
        "simulation_id",
        "health_state",
        "degradation_level",
        "degradation_name",
        "temperature_c",
        "temperature_k",
        "profile_type",
        "initial_soc",
        "timesteps",
    ]
    lines = [",".join(header)]
    for rec in records:
        md = rec.metadata
        lines.append(
            ",".join(
                [
                    rec.simulation_id,
                    str(md["health_state"]),
                    str(md["degradation_level"]),
                    str(md["degradation_name"]),
                    f"{md['temperature_c']:.1f}",
                    f"{md['temperature_k']:.2f}",
                    str(md["profile_type"]),
                    f"{md['initial_soc']:.4f}",
                    str(md["timesteps"]),
                ]
            )
        )
    RAW_METADATA_PATH.write_text("\n".join(lines), encoding="utf-8")


def generate_all_simulations(
    num_healthy: int | None = None,
    num_degraded: int | None = None,
    save_raw: bool = True,
) -> list[SimulationRecord]:
    num_healthy = SIM_CONFIG["num_healthy"] if num_healthy is None else int(num_healthy)
    num_degraded = SIM_CONFIG["num_degraded"] if num_degraded is None else int(num_degraded)

    rng = np.random.default_rng(SIM_CONFIG["seed"])
    duration_s = SIM_CONFIG["duration_s"]
    dt = SIM_CONFIG["sampling_period_s"]
    num_steps = (duration_s // dt) + 1
    temp_levels_k = _temperature_levels_k()
    degraded_levels = _build_degraded_level_schedule(num_degraded)

    nominal_capacity_ah = 5.0
    all_records: list[SimulationRecord] = []

    if save_raw:
        RAW_PATH.mkdir(parents=True, exist_ok=True)

    total = num_healthy + num_degraded
    for idx in range(total):
        is_healthy = idx < num_healthy
        degradation_level = 0 if is_healthy else degraded_levels[idx - num_healthy]
        simulation_id = f"sim_{idx:04d}"

        max_attempts = 6
        last_error: Exception | None = None
        for _attempt in range(max_attempts):
            profile_type = str(rng.choice(SIM_CONFIG["current_profile_types"]))
            ambient_temperature_k = float(rng.choice(temp_levels_k))
            if is_healthy:
                soc_low, soc_high = SIM_CONFIG["initial_soc_range"]
            else:
                soc_low, soc_high = 0.82, 0.98
            initial_soc = float(rng.uniform(low=soc_low, high=soc_high))

            current_profile = generate_current_profile(
                profile_type=profile_type,
                num_steps=num_steps,
                capacity_ah=nominal_capacity_ah,
                rng=rng,
            )
            try:
                if is_healthy:
                    output = run_healthy(
                        current_profile_a=current_profile.current_a,
                        ambient_temperature_k=ambient_temperature_k,
                        initial_soc=initial_soc,
                    )
                    health_state = "healthy"
                else:
                    degraded_current = (
                        current_profile.current_a * _degraded_current_multiplier(degradation_level)
                    )
                    output = run_degraded(
                        current_profile_a=degraded_current,
                        ambient_temperature_k=ambient_temperature_k,
                        initial_soc=initial_soc,
                        degradation_level_id=degradation_level,
                    )
                    health_state = "degraded"
                break
            except Exception as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Failed to simulate {simulation_id} after {max_attempts} attempts"
            ) from last_error

        record = SimulationRecord(
            simulation_id=simulation_id,
            voltage_v=output.voltage_v,
            current_a=output.current_a,
            temperature_k=output.temperature_k,
            soc=output.soc,
            metadata={
                "health_state": health_state,
                "degradation_level": degradation_level,
                "degradation_name": DEGRADATION_CONFIG["levels"][degradation_level]["name"],
                "temperature_c": ambient_temperature_k - 273.15,
                "temperature_k": ambient_temperature_k,
                "profile_type": current_profile.profile_type,
                "initial_soc": initial_soc,
                "timesteps": len(output.time_s),
            },
        )

        if save_raw:
            _save_raw_simulation(record, RAW_PATH)
        all_records.append(record)

    if save_raw:
        _write_metadata(all_records)

    return all_records


generate_all_Simulations = generate_all_simulations
