from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SIM_CONFIG


@dataclass(frozen=True)
class CurrentProfile:
    profile_type: str
    current_a: np.ndarray


def _build_constant_profile(
    rng: np.random.Generator,
    num_steps: int,
    capacity_ah: float,
) -> np.ndarray:
    c_rate = float(rng.choice(SIM_CONFIG["constant_c_rates"]))
    direction = float(rng.choice([1.0, -1.0]))
    return np.full(num_steps, direction * c_rate * capacity_ah, dtype=np.float64)


def _build_pulse_profile(
    rng: np.random.Generator,
    num_steps: int,
    capacity_ah: float,
) -> np.ndarray:
    profile = np.zeros(num_steps, dtype=np.float64)
    idx = 0
    max_current = SIM_CONFIG["max_abs_c_rate"] * capacity_ah
    while idx < num_steps:
        pulse_len = int(rng.integers(5, 120))
        amp = float(rng.uniform(-max_current, max_current))
        end = min(num_steps, idx + pulse_len)
        profile[idx:end] = amp
        idx = end
    return profile


def _build_random_profile(
    rng: np.random.Generator,
    num_steps: int,
    capacity_ah: float,
) -> np.ndarray:
    max_current = SIM_CONFIG["max_abs_c_rate"] * capacity_ah
    noise = rng.normal(loc=0.0, scale=0.45 * max_current, size=num_steps)
    kernel = np.ones(31, dtype=np.float64) / 31.0
    smooth = np.convolve(noise, kernel, mode="same")
    return np.clip(smooth, -max_current, max_current)


def _build_drive_cycle_profile(
    rng: np.random.Generator,
    num_steps: int,
    capacity_ah: float,
) -> np.ndarray:
    max_current = SIM_CONFIG["max_abs_c_rate"] * capacity_ah
    t = np.arange(num_steps, dtype=np.float64)
    base = (
        0.35 * np.sin(2.0 * np.pi * t / 180.0)
        + 0.25 * np.sin(2.0 * np.pi * t / 37.0)
        + 0.15 * np.sin(2.0 * np.pi * t / 11.0)
    ) * max_current
    stop_go = (rng.random(num_steps) > 0.82).astype(np.float64)
    stop_go = np.convolve(stop_go, np.ones(7, dtype=np.float64) / 7.0, mode="same")
    regen = -0.35 * max_current * stop_go
    perturb = rng.normal(scale=0.05 * max_current, size=num_steps)
    return np.clip(base + regen + perturb, -max_current, max_current)


def generate_current_profile(
    profile_type: str,
    num_steps: int,
    capacity_ah: float,
    rng: np.random.Generator,
) -> CurrentProfile:
    generators = {
        "constant": _build_constant_profile,
        "pulse": _build_pulse_profile,
        "random": _build_random_profile,
        "drive_cycle": _build_drive_cycle_profile,
    }
    if profile_type not in generators:
        raise ValueError(f"Unsupported profile_type: {profile_type}")

    current = generators[profile_type](rng=rng, num_steps=num_steps, capacity_ah=capacity_ah)
    return CurrentProfile(profile_type=profile_type, current_a=current)
    