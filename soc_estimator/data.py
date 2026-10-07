from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCALED_FORECAST_FEATURES = [
    "Voltage_measured", "Current_measured", "Ambient_Temperature_K", "Time_Step_s"
]
FORECAST_FEATURES = [*SCALED_FORECAST_FEATURES, "Current_SOC_Estimate"]


def scale_sensor_features(values, mean, scale) -> np.ndarray:
    """Match StandardScaler's float32 arithmetic in preparation and inference."""
    features = np.asarray(values, dtype=np.float32).copy()
    features -= mean
    features /= scale
    return features


class SOCDataset(Dataset):
    def __init__(self, X, y, sample_weights=None):
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
        if X.ndim != 3 or y.ndim != 1 or len(X) != len(y) or min(X.shape) == 0:
            raise ValueError("SOC data requires nonempty [windows, steps, features] inputs and aligned 1D targets")
        if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError("SOC data must be finite with target fractions between zero and one")
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)
        if self.X.dtype != torch.float32:
            self.X = self.X.float()
        if self.y.dtype != torch.float32:
            self.y = self.y.float()
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.sample_weights is not None:
            return self.X[idx], self.y[idx], self.sample_weights[idx]
        return self.X[idx], self.y[idx]


def load_dataloaders(
    dataset_path,
    batch_size=64,
    num_workers=2,
    pin_memory=None,
):
    X_train = np.load(f"{dataset_path}/X_train.npy")
    y_train = np.load(f"{dataset_path}/y_train.npy")
    X_val = np.load(f"{dataset_path}/X_val.npy")
    y_val = np.load(f"{dataset_path}/y_val.npy")
    X_test = np.load(f"{dataset_path}/X_test.npy")
    y_test = np.load(f"{dataset_path}/y_test.npy")

    train_dataset = SOCDataset(X_train, y_train)
    test_dataset = SOCDataset(X_test, y_test)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_options = {
        "batch_size": batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2

    metadata_path = Path(dataset_path) / "window_metadata_train.csv"
    sampler = None
    if metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as file:
            cycle_numbers = [row["cycle_number"] for row in csv.DictReader(file)]
        if len(cycle_numbers) != len(train_dataset):
            raise ValueError(
                "Training metadata must contain one cycle number per training window"
            )
        if not cycle_numbers or any(not cycle for cycle in cycle_numbers):
            raise ValueError("Training metadata contains a missing cycle number")
        cycle_counts = Counter(cycle_numbers)
        weights = torch.tensor(
            [1.0 / cycle_counts[cycle] for cycle in cycle_numbers], dtype=torch.double
        )
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)

    val_weights = None
    val_metadata_path = Path(dataset_path) / "window_metadata_val.csv"
    if val_metadata_path.exists():
        with val_metadata_path.open(newline="", encoding="utf-8") as file:
            val_cycles = [row["cycle_number"] for row in csv.DictReader(file)]
        if len(val_cycles) != len(y_val):
            raise ValueError("Validation metadata must contain one cycle number per window")
        if not val_cycles or any(not cycle for cycle in val_cycles):
            raise ValueError("Validation metadata contains a missing cycle number")
        val_counts = Counter(val_cycles)
        val_weights = torch.tensor(
            [1.0 / val_counts[cycle] for cycle in val_cycles], dtype=torch.float32
        )
    val_dataset = SOCDataset(X_val, y_val, sample_weights=val_weights)

    train_loader = DataLoader(
        train_dataset, sampler=sampler, shuffle=sampler is None, **loader_options
    )
    val_loader = DataLoader(val_dataset, **loader_options)
    test_loader = DataLoader(test_dataset, **loader_options)
    return train_loader, val_loader, test_loader
