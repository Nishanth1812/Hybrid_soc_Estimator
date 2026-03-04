from __future__ import annotations

import random

from config import PREPROCESSING_CONFIG
from data_pipeline.generation.dataset_generator import SimulationRecord


def split_simulations(
    simulations: list[SimulationRecord],
) -> tuple[list[SimulationRecord], list[SimulationRecord], list[SimulationRecord]]:
    sims = list(simulations)
    rng = random.Random(PREPROCESSING_CONFIG["shuffle_seed"])
    rng.shuffle(sims)

    n = len(sims)
    train_end = int(n * PREPROCESSING_CONFIG["train_ratio"])
    val_end = train_end + int(n * PREPROCESSING_CONFIG["val_ratio"])

    train = sims[:train_end]
    val = sims[train_end:val_end]
    test = sims[val_end:]
    return train, val, test
