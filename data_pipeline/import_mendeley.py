from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {
    "Cycle_Number",
    "Cycle_Type",
    "Current_measured",
    "Voltage_measured",
    "Ambient_Temperature",
    "soc",
}


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Mendeley workbook is missing columns: {sorted(missing)}")

    columns = [
        "Cycle_Number",
        "Cycle_Type",
        "Voltage_measured",
        "Current_measured",
        "Ambient_Temperature",
        "soc",
    ]
    frame = frame[columns].dropna()
    numeric = [column for column in columns if column != "Cycle_Type"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Mendeley workbook contains non-finite values")

    soc_max = float(frame["soc"].max())
    if soc_max > 1.5:
        frame["soc"] /= 100.0
    frame["soc"] = frame["soc"].clip(0.0, 1.0)
    frame["Ambient_Temperature"] += 273.15
    return frame


def _split_cycle_numbers(cycle_numbers: np.ndarray, seed: int) -> dict[str, set]:
    numbers = np.asarray(cycle_numbers)
    rng = np.random.default_rng(seed)
    shuffled = numbers.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * 0.70))
    n_val = max(1, int(len(shuffled) * 0.15))
    if n_train + n_val >= len(shuffled):
        n_val = max(1, len(shuffled) - n_train - 1)
        n_train = len(shuffled) - n_val - 1
    return {
        "train": set(shuffled[:n_train]),
        "val": set(shuffled[n_train : n_train + n_val]),
        "test": set(shuffled[n_train + n_val :]),
    }


def build_splits(
    frame: pd.DataFrame,
    sequence_length: int = 100,
    stride: int = 1,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")

    frame = _prepare_frame(frame)
    cycle_split = _split_cycle_numbers(frame["Cycle_Number"].drop_duplicates().to_numpy(), seed)
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in cycle_split}

    for (cycle_number, _cycle_type), group in frame.groupby(
        ["Cycle_Number", "Cycle_Type"], sort=False
    ):
        split_name = next(name for name, ids in cycle_split.items() if cycle_number in ids)
        features = group[["Voltage_measured", "Current_measured", "Ambient_Temperature"]].to_numpy(
            dtype=np.float32
        )
        target = group["soc"].to_numpy(dtype=np.float32)
        groups[split_name].append((features, target))

    scaler = StandardScaler()
    scaler.fit(np.vstack([features for features, _ in groups["train"]]))
    results: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_groups in groups.items():
        counts = [
            max(0, (len(target) - sequence_length) // stride + 1)
            for _, target in split_groups
        ]
        X = np.empty((sum(counts), sequence_length, 3), dtype=np.float32)
        y = np.empty((sum(counts),), dtype=np.float32)
        cursor = 0
        for (features, target), count in zip(split_groups, counts):
            if count == 0:
                continue
            scaled = scaler.transform(features).astype(np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(
                scaled, sequence_length, axis=0
            ).transpose(0, 2, 1)[::stride]
            X[cursor : cursor + count] = windows
            y[cursor : cursor + count] = target[sequence_length - 1 :: stride][:count]
            cursor += count
        results[split_name] = {"X": X, "y": y}

    results["scaler"] = scaler  # type: ignore[assignment]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Mendeley EV SOC workbook")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/mendeley/processed"))
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_excel(args.input, engine="openpyxl")
    splits = build_splits(frame, args.sequence_length, args.stride, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        np.save(args.output / f"X_{split_name}.npy", splits[split_name]["X"])
        np.save(args.output / f"y_{split_name}.npy", splits[split_name]["y"])
    joblib.dump(splits["scaler"], args.output / "input_scaler.pkl")
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "source": "Mendeley EV Lithium Ion Battery State of Charge Dataset",
                "sequence_length": args.sequence_length,
                "stride": args.stride,
                "features": ["Voltage_measured", "Current_measured", "Ambient_Temperature_K"],
                "target": "soc_fraction",
                "shapes": {
                    split: {key: list(value.shape) for key, value in splits[split].items()}
                    for split in ("train", "val", "test")
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for split_name in ("train", "val", "test"):
        print(split_name, splits[split_name]["X"].shape, splits[split_name]["y"].shape)


if __name__ == "__main__":
    main()
