from __future__ import annotations

import argparse

import numpy as np

from config import PROCESSED_PATH, RAW_PATH, SCALERS_PATH
from data_pipeline.generation.dataset_generator import generate_all_simulations
from data_pipeline.preprocessing.filtering import apply_filter
from data_pipeline.preprocessing.scaling import apply_scaler, fit_scaler
from data_pipeline.preprocessing.sequence_builder import build_sequences
from data_pipeline.preprocessing.splitting import split_simulations


def ensure_dirs() -> None:
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    SCALERS_PATH.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOC dataset with PyBaMM simulations")
    parser.add_argument("--num-healthy", type=int, default=None, help="Override healthy simulation count")
    parser.add_argument("--num-degraded", type=int, default=None, help="Override degraded simulation count")
    parser.add_argument(
        "--skip-raw-save",
        action="store_true",
        help="Skip writing per-simulation raw .npz files (metadata still generated if save enabled)",
    )
    return parser.parse_args()


def save_processed(X: np.ndarray, y: np.ndarray, split_name: str) -> None:
    np.save(PROCESSED_PATH / f"X_{split_name}.npy", X)
    np.save(PROCESSED_PATH / f"y_{split_name}.npy", y)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    simulations = generate_all_simulations(
        num_healthy=args.num_healthy,
        num_degraded=args.num_degraded,
        save_raw=not args.skip_raw_save,
    )

    filtered = [apply_filter(sim) for sim in simulations]
    train_sims, val_sims, test_sims = split_simulations(filtered)

    scaler = fit_scaler(train_sims)
    train_scaled = apply_scaler(train_sims, scaler)
    val_scaled = apply_scaler(val_sims, scaler)
    test_scaled = apply_scaler(test_sims, scaler)

    X_train, y_train = build_sequences(train_scaled)
    X_val, y_val = build_sequences(val_scaled)
    X_test, y_test = build_sequences(test_scaled)

    save_processed(X_train, y_train, "train")
    save_processed(X_val, y_val, "val")
    save_processed(X_test, y_test, "test")

    print("Dataset build complete")
    print(f"Train X/y: {X_train.shape} / {y_train.shape}")
    print(f"Val   X/y: {X_val.shape} / {y_val.shape}")
    print(f"Test  X/y: {X_test.shape} / {y_test.shape}")


if __name__ == "__main__":
    main()