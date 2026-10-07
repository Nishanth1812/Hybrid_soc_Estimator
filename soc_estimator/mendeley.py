from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .data import FORECAST_FEATURES, SCALED_FORECAST_FEATURES, scale_sensor_features

REQUIRED_COLUMNS = {
    "Cycle_Number",
    "Cycle_Type",
    "Current_measured",
    "Voltage_measured",
    "Ambient_Temperature",
    "soc",
}
SENSOR_FEATURE_COLUMNS = ["Voltage_measured", "Current_measured", "Ambient_Temperature"]
TIME_STEP_COLUMN = "Time_Step_s"
PUBLISHED_SAMPLE_COUNT = 604_750


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Workbook contains duplicate columns after trimming whitespace")
    return frame


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.empty(0, dtype=np.float64)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def data_quality_report(frame: pd.DataFrame) -> dict:
    """Summarize source quality without changing or rejecting measurements."""
    frame = _normalize_columns(frame)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Mendeley workbook is missing columns: {sorted(missing_columns)}")

    numeric = [column for column in REQUIRED_COLUMNS if column != "Cycle_Type"]
    if "Time" in frame:
        numeric.append("Time")
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    usable = frame[list(REQUIRED_COLUMNS)].notna().all(axis=1)
    usable &= np.isfinite(numeric_frame.to_numpy(dtype=np.float64)).all(axis=1)

    soc_values = _numeric_column(frame, "soc")
    soc_scale = 100.0 if soc_values.size and np.nanmax(soc_values) > 1.5 else 1.0
    soc_fraction = soc_values / soc_scale
    report = {
        "rows_in_source": int(len(frame)),
        "rows_usable": int(usable.sum()),
        "rows_dropped_for_missing_or_non_finite_required_values": int(len(frame) - usable.sum()),
        "missing_required_values_by_column": {
            column: int(frame[column].isna().sum()) for column in sorted(REQUIRED_COLUMNS)
        },
        "non_numeric_required_values_by_column": {
            column: int((frame[column].notna() & numeric_frame[column].isna()).sum())
            for column in numeric
        },
        "non_finite_required_values_by_column": {
            column: int(
                np.count_nonzero(
                    ~np.isfinite(
                        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
                    )
                )
            )
            for column in numeric
        },
        "cycle_count": int(frame.loc[usable, "Cycle_Number"].nunique()),
        "cycle_id_order_decreases": int(
            np.count_nonzero(
                np.diff(
                    pd.to_numeric(frame["Cycle_Number"], errors="coerce")
                    .dropna()
                    .drop_duplicates()
                    .to_numpy(dtype=np.float64)
                )
                < 0
            )
        ),
        "cycle_type_row_counts": {
            str(key): int(value)
            for key, value in frame.loc[usable, "Cycle_Type"].value_counts(dropna=False).items()
        },
        "soc_fraction_range_before_clipping": {
            "min": float(np.min(soc_fraction)) if soc_fraction.size else None,
            "max": float(np.max(soc_fraction)) if soc_fraction.size else None,
        },
        "soc_values_outside_zero_to_one_count": int(
            np.count_nonzero((soc_fraction < 0.0) | (soc_fraction > 1.0))
        ),
        "soc_values_below_zero_count": int(np.count_nonzero(soc_fraction < 0.0)),
        "soc_values_above_one_count": int(np.count_nonzero(soc_fraction > 1.0)),
        "source_value_ranges": {
            column: _distribution(_numeric_column(frame, column))
            for column in [*SENSOR_FEATURE_COLUMNS, "soc"]
        },
    }

    if "Time" in frame:
        times = pd.to_numeric(frame["Time"], errors="coerce")
        intervals = times.groupby(
            [frame["Cycle_Number"], frame["Cycle_Type"]], sort=False
        ).diff().to_numpy(dtype=np.float64)
        finite_intervals = intervals[np.isfinite(intervals)]
        positive_intervals = finite_intervals[finite_intervals > 0]
        report["time_interval_seconds"] = {
            **_distribution(positive_intervals),
            "zero_count": int(np.count_nonzero(finite_intervals == 0)),
            "negative_count": int(np.count_nonzero(finite_intervals < 0)),
        }
        report["time_missing_or_non_finite_count"] = int(
            np.count_nonzero(~np.isfinite(times.to_numpy(dtype=np.float64)))
        )
    else:
        report["time_interval_seconds"] = None
        report["time_missing_or_non_finite_count"] = None

    if "Delta t" in frame:
        delta_t = pd.to_numeric(frame["Delta t"], errors="coerce").to_numpy(dtype=np.float64)
        finite_delta_t = delta_t[np.isfinite(delta_t)]
        report["delta_t_seconds"] = {
            **_distribution(finite_delta_t),
            "non_positive_count": int(np.count_nonzero(finite_delta_t <= 0)),
            "missing_or_non_finite_count": int(np.count_nonzero(~np.isfinite(delta_t))),
        }
    else:
        report["delta_t_seconds"] = None
    return report


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_columns(frame).reset_index(drop=True)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Mendeley workbook is missing columns: {sorted(missing)}")

    time_columns = ["Time"] if "Time" in frame else []
    columns = ["Cycle_Number", "Cycle_Type", *SENSOR_FEATURE_COLUMNS, "soc", *time_columns]
    frame = frame[columns].copy()
    required_numeric = [column for column in columns if column != "Cycle_Type"]
    frame[required_numeric] = frame[required_numeric].apply(pd.to_numeric, errors="raise")
    frame = frame.dropna(subset=columns)
    if not np.isfinite(frame[required_numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError("Mendeley workbook contains non-finite values")

    soc_max = float(frame["soc"].max())
    if soc_max > 1.5:
        frame["soc"] /= 100.0
    frame["soc"] = frame["soc"].clip(0.0, 1.0)
    frame["Ambient_Temperature"] += 273.15
    if "Time" in frame:
        frame[TIME_STEP_COLUMN] = frame.groupby(
            ["Cycle_Number", "Cycle_Type"], sort=False
        )["Time"].diff().fillna(0.0)
    return frame


def _split_cycle_numbers(
    cycle_numbers: np.ndarray, seed: int, strategy: str
) -> dict[str, set]:
    numbers = np.asarray(cycle_numbers)
    if numbers.size < 3:
        raise ValueError("At least three cycles are required for train/validation/test splits")
    if strategy == "random":
        numbers = numbers.copy()
        np.random.default_rng(seed).shuffle(numbers)
    elif strategy == "chronological":
        numbers = np.sort(numbers)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    n_train = max(1, int(len(numbers) * 0.70))
    n_val = max(1, int(len(numbers) * 0.15))
    if n_train + n_val >= len(numbers):
        n_val = max(1, len(numbers) - n_train - 1)
        n_train = len(numbers) - n_val - 1
    return {
        "train": set(numbers[:n_train]),
        "val": set(numbers[n_train : n_train + n_val]),
        "test": set(numbers[n_train + n_val :]),
    }


def _contiguous_segments(group: pd.DataFrame, max_gap_seconds: float) -> list[pd.DataFrame]:
    """Keep windows within source-contiguous, forward-moving time segments."""
    breaks_before = np.diff(group.index.to_numpy()) != 1
    if TIME_STEP_COLUMN in group:
        intervals = group[TIME_STEP_COLUMN].to_numpy(dtype=np.float64)[1:]
        breaks_before |= (intervals <= 0) | (intervals > max_gap_seconds)
    breaks = np.flatnonzero(breaks_before) + 1
    bounds = [0, *breaks.tolist(), len(group)]
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        segment = group.iloc[start:end].copy()
        if not segment.empty and TIME_STEP_COLUMN in segment:
            segment.iloc[0, segment.columns.get_loc(TIME_STEP_COLUMN)] = 0.0
        segments.append(segment)
    return segments


def build_splits(
    frame: pd.DataFrame,
    sequence_length: int = 100,
    stride: int | None = None,
    seed: int = 42,
    split_strategy: str = "chronological",
    train_stride: int | None = None,
    eval_stride: int | None = None,
    forecast_horizon_seconds: float = 60.0,
    max_gap_seconds: float = 60.0,
) -> dict[str, dict[str, np.ndarray]]:
    default_train_stride, default_eval_stride = (10, 1) if stride is None else (stride, stride)
    train_stride = default_train_stride if train_stride is None else train_stride
    eval_stride = default_eval_stride if eval_stride is None else eval_stride
    if sequence_length < 1 or min(train_stride, eval_stride) < 1:
        raise ValueError("sequence_length and window strides must be positive")
    if not np.isfinite([forecast_horizon_seconds, max_gap_seconds]).all() or min(
        forecast_horizon_seconds, max_gap_seconds
    ) <= 0:
        raise ValueError("Forecast horizon and maximum measurement gap must be finite and positive")

    frame = _prepare_frame(frame)
    if "Time" not in frame:
        raise ValueError("Future SOC forecasting requires the Time column in seconds")
    sensor_feature_columns = SENSOR_FEATURE_COLUMNS + [TIME_STEP_COLUMN]
    feature_columns = FORECAST_FEATURES
    cycle_split = _split_cycle_numbers(
        frame["Cycle_Number"].drop_duplicates().to_numpy(), seed, split_strategy
    )
    assignment = {
        cycle_number: split_name
        for split_name, cycle_numbers in cycle_split.items()
        for cycle_number in cycle_numbers
    }
    groups: dict[str, list[tuple[object, object, pd.DataFrame]]] = {
        name: [] for name in cycle_split
    }

    for (cycle_number, cycle_type), group in frame.groupby(
        ["Cycle_Number", "Cycle_Type"], sort=False
    ):
        for segment in _contiguous_segments(group, max_gap_seconds):
            groups[assignment[cycle_number]].append((cycle_number, cycle_type, segment))

    scaler = StandardScaler()
    train_features = [
        segment[sensor_feature_columns].to_numpy(dtype=np.float32)
        for _, _, segment in groups["train"]
        if len(segment) >= sequence_length
    ]
    if not train_features:
        raise ValueError("No training cycle is long enough for the requested sequence length")
    scaler.fit(np.vstack(train_features))
    zero_interval_input = scale_sensor_features(
        np.zeros((1, len(sensor_feature_columns)), dtype=np.float32), scaler.mean_, scaler.scale_
    )[0, -1]

    results: dict[str, dict[str, object]] = {}
    for split_name, split_groups in groups.items():
        split_stride = train_stride if split_name == "train" else eval_stride
        window_starts = []
        for _, _, segment in split_groups:
            starts = np.arange(0, max(0, len(segment) - sequence_length + 1), split_stride)
            times = segment["Time"].to_numpy(dtype=np.float64)
            starts = starts[times[starts + sequence_length - 1] + forecast_horizon_seconds <= times[-1]]
            window_starts.append(starts)
        counts = [len(starts) for starts in window_starts]
        total_windows = sum(counts)
        if total_windows == 0:
            raise ValueError(f"No {split_name} windows have sufficient history and future SOC coverage")
        X = np.empty((total_windows, sequence_length, len(feature_columns)), dtype=np.float32)
        y = np.empty((total_windows,), dtype=np.float32)
        metadata_parts = []
        cursor = 0

        for (cycle_number, cycle_type, segment), starts in zip(split_groups, window_starts):
            count = len(starts)
            if count == 0:
                continue
            features = segment[sensor_feature_columns].to_numpy(dtype=np.float32)
            target = segment["soc"].to_numpy(dtype=np.float32)
            scaled = scale_sensor_features(features, scaler.mean_, scaler.scale_)
            windows = np.lib.stride_tricks.sliding_window_view(
                scaled, sequence_length, axis=0
            ).transpose(0, 2, 1)
            ends = starts + sequence_length - 1
            X[cursor : cursor + count, :, : len(sensor_feature_columns)] = windows[starts]
            X[cursor : cursor + count, :, -1] = target[ends, None]
            # A standalone history has no observation preceding its first sample.
            X[cursor : cursor + count, 0, len(sensor_feature_columns) - 1] = zero_interval_input

            times = segment["Time"].to_numpy(dtype=np.float64)
            start_times, end_times = times[starts], times[ends]
            target_times = end_times + forecast_horizon_seconds
            y[cursor : cursor + count] = np.interp(target_times, times, target)
            right = np.searchsorted(times, target_times, side="left")
            left = np.where(times[right] == target_times, right, right - 1)
            mean_interval = (end_times - start_times) / max(1, sequence_length - 1)
            mean_current = np.lib.stride_tricks.sliding_window_view(
                features[:, 1], sequence_length
            )[starts].mean(axis=1)
            mean_temperature = np.lib.stride_tricks.sliding_window_view(
                features[:, 2], sequence_length
            )[starts].mean(axis=1) - 273.15
            row_numbers = segment.index.to_numpy()
            metadata_parts.append(
                pd.DataFrame(
                    {
                        "cycle_number": cycle_number,
                        "cycle_type": cycle_type,
                        "source_start_index": row_numbers[starts],
                        "source_end_index": row_numbers[ends],
                        "start_time_s": start_times,
                        "end_time_s": end_times,
                        "target_time_s": target_times,
                        "forecast_horizon_s": forecast_horizon_seconds,
                        "target_left_source_index": row_numbers[left],
                        "target_right_source_index": row_numbers[right],
                        "reference_soc_at_end": target[ends],
                        "mean_interval_s": mean_interval,
                        "mean_ambient_temperature_c": mean_temperature,
                        "mean_current_a": mean_current,
                    }
                )
            )
            cursor += count

        metadata = pd.concat(metadata_parts, ignore_index=True)
        results[split_name] = {"X": X, "y": y, "metadata": metadata}

    results["scaler"] = scaler
    results["features"] = feature_columns
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Mendeley EV SOC workbook")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/mendeley/forecast_60s_with_current_soc_v1")
    )
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=None, help="Set both window strides")
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forecast-horizon-seconds", type=float, default=60.0)
    parser.add_argument("--max-gap-seconds", type=float, default=60.0)
    parser.add_argument(
        "--split-strategy", choices=("chronological", "random"), default="chronological"
    )
    args = parser.parse_args()
    if args.stride is not None and (args.train_stride is not None or args.eval_stride is not None):
        parser.error("--stride cannot be combined with --train-stride or --eval-stride")
    train_stride = args.train_stride if args.train_stride is not None else 10
    eval_stride = args.eval_stride if args.eval_stride is not None else 1
    if args.stride is not None:
        train_stride = eval_stride = args.stride

    frame = pd.read_excel(args.input, engine="openpyxl")
    if "Time" not in frame.columns:
        raise ValueError("Mendeley workbook is missing the Time column required for Δt input")
    quality = data_quality_report(frame)
    splits = build_splits(
        frame,
        sequence_length=args.sequence_length,
        seed=args.seed,
        split_strategy=args.split_strategy,
        train_stride=train_stride,
        eval_stride=eval_stride,
        forecast_horizon_seconds=args.forecast_horizon_seconds,
        max_gap_seconds=args.max_gap_seconds,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split = splits[split_name]
        np.save(args.output / f"X_{split_name}.npy", split["X"])
        np.save(args.output / f"y_{split_name}.npy", split["y"])
        split["metadata"].to_csv(
            args.output / f"window_metadata_{split_name}.csv", index=False
        )
    joblib.dump(splits["scaler"], args.output / "input_scaler.pkl")
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "source": "Mendeley EV Lithium Ion Battery State of Charge Dataset",
                "source_record": {
                    "url": "https://data.mendeley.com/datasets/7dkmgbspfg/2",
                    "version": 2,
                    "reported_sample_count": PUBLISHED_SAMPLE_COUNT,
                    "local_workbook_rows": quality["rows_in_source"],
                    "sample_count_difference": PUBLISHED_SAMPLE_COUNT - quality["rows_in_source"],
                    "workbook_version_verified": False,
                },
                "sequence_length": args.sequence_length,
                "train_stride": train_stride,
                "eval_stride": eval_stride,
                "split_strategy": args.split_strategy,
                "seed": args.seed,
                "task": "future_soc_forecast",
                "forecast_horizon_seconds": args.forecast_horizon_seconds,
                "max_gap_seconds": args.max_gap_seconds,
                "target_alignment": "linear interpolation of reference SOC at input end + horizon; no extrapolation",
                "input_units": ["V", "A", "K", "s"],
                "first_window_interval_seconds": 0.0,
                "input_scaler": {
                    "mean": splits["scaler"].mean_.tolist(),
                    "scale": splits["scaler"].scale_.tolist(),
                },
                "features": splits["features"],
                "scaled_features": SCALED_FORECAST_FEATURES,
                "current_soc_feature_index": len(SCALED_FORECAST_FEATURES),
                "current_soc_source": "reference SOC at the final input timestamp; deployment supplies the live BMS SOC estimate",
                "target": "soc_fraction",
                "cycle_numbers": {
                    split: splits[split]["metadata"]["cycle_number"].drop_duplicates().tolist()
                    for split in ("train", "val", "test")
                },
                "window_metadata": {
                    split: f"window_metadata_{split}.csv" for split in ("train", "val", "test")
                },
                "data_quality": quality,
                "shapes": {
                    split: {
                        "X": list(splits[split]["X"].shape),
                        "y": list(splits[split]["y"].shape),
                    }
                    for split in ("train", "val", "test")
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for split_name in ("train", "val", "test"):
        print(split_name, splits[split_name]["X"].shape, splits[split_name]["y"].shape)
    print("Data quality:", json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
