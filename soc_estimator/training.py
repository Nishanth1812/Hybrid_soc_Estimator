from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .data import FORECAST_FEATURES, load_dataloaders
from .model import LSTMSOCEstimator


def configure_logging(log_path: str | Path, logger_name: str) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.lr_scheduler = lr_scheduler

    def train_epoch(
        self,
        dataloader,
        logger: logging.Logger | None = None,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_batches = len(dataloader)
        progress_interval = max(1, total_batches // 10)
        started_at = time.perf_counter()

        for batch_index, (X, y) in enumerate(dataloader, start=1):
            non_blocking = self.device.type == "cuda"
            X = X.to(self.device, non_blocking=non_blocking)
            y = y.to(self.device, non_blocking=non_blocking)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; checkpoint has not been replaced")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

            if logger is not None and (
                batch_index == 1
                or batch_index % progress_interval == 0
                or batch_index == total_batches
            ):
                epoch_label = (
                    f"{epoch}/{total_epochs}"
                    if epoch is not None and total_epochs is not None
                    else str(epoch or "?")
                )
                logger.info(
                    "Epoch %s | batch %d/%d | loss=%.6f | elapsed=%.1fs",
                    epoch_label,
                    batch_index,
                    total_batches,
                    loss.item(),
                    time.perf_counter() - started_at,
                )

        if total_samples == 0:
            raise ValueError("Training data must not be empty")
        return total_loss / total_samples

    def step_scheduler(self, val_loss):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for batch in dataloader:
                X, y = batch[:2]
                weights = batch[2] if len(batch) == 3 else None
                non_blocking = self.device.type == "cuda"
                X = X.to(self.device, non_blocking=non_blocking)
                y = y.to(self.device, non_blocking=non_blocking)
                pred = self.model(X)
                losses = nn.functional.mse_loss(pred, y, reduction="none").view(-1)
                if weights is None:
                    weights = torch.ones_like(losses)
                else:
                    weights = weights.to(self.device, non_blocking=non_blocking)
                total_loss += torch.sum(losses * weights).item()
                total_weight += weights.sum().item()
        if total_weight <= 0 or not torch.isfinite(torch.tensor(total_loss)):
            raise ValueError("Validation requires finite losses and positive sample weights")
        return total_loss / total_weight


def _log_loader_summary(logger: logging.Logger, name: str, loader) -> None:
    logger.info(
        "%s dataset | samples=%d | batches=%d | batch_size=%d",
        name,
        len(loader.dataset),
        len(loader),
        loader.batch_size,
    )


def save_training_history(history: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_model(
    dataset_path,
    epochs=300,
    batch_size=64,
    num_workers=2,
    lr=3e-4,
    seed=42,
    early_stop_patience=40,
    model_path="models/best_model.pt",
    history_path="logs/training_history.json",
    log_path="logs/training.log",
):
    if epochs < 1 or early_stop_patience < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Epochs, stopping patience and learning rate must be positive")
    dataset_path = Path(dataset_path)
    metadata = json.loads((dataset_path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("task") != "future_soc_forecast" or metadata.get("features") != FORECAST_FEATURES:
        raise ValueError("Rebuild the sensor-only future SOC dataset before training")
    scaler = metadata["input_scaler"]
    if len(scaler["mean"]) != len(FORECAST_FEATURES) - 1 or len(scaler["scale"]) != len(FORECAST_FEATURES) - 1 or not all(
        math.isfinite(value) for value in scaler["mean"] + scaler["scale"]
    ) or any(value <= 0 for value in scaler["scale"]):
        raise ValueError("Dataset contains invalid input scaling")
    torch.manual_seed(seed)
    logger = configure_logging(Path(log_path), "soc_training")
    training_started_at = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 22))

    logger.info("Training started")
    logger.info("Dataset path: %s", dataset_path)
    logger.info(
        "Configuration | epochs=%d | batch_size=%d | learning_rate=%.6g | num_workers=%d",
        epochs,
        batch_size,
        lr,
        num_workers,
    )
    logger.info(
        "Hardware | device=%s | cuda_available=%s | cuda_device_count=%d",
        device,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    train_loader, val_loader, test_loader = load_dataloaders(
        dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    _log_loader_summary(logger, "Train", train_loader)
    _log_loader_summary(logger, "Validation", val_loader)
    _log_loader_summary(logger, "Test", test_loader)

    input_features = train_loader.dataset.X.shape[-1]
    for loader in (train_loader, val_loader, test_loader):
        if list(loader.dataset.X.shape[1:]) != [metadata["sequence_length"], len(FORECAST_FEATURES)]:
            raise ValueError("Dataset shape does not match the forecast metadata")
    model = LSTMSOCEstimator(
        input_features=input_features,
        current_soc_feature_index=metadata["current_soc_feature_index"],
    ).to(device)
    logger.info("Using device: %s", device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    trainer = Trainer(model, optimizer, nn.MSELoss(), device, lr_scheduler=scheduler)

    best_val = trainer.validate(val_loader)
    best_epoch = 0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_path)
    history = {
        "initial_val_loss": best_val,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "forecast_horizon_seconds": metadata["forecast_horizon_seconds"],
        "configuration": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                          "seed": seed, "early_stop_patience": early_stop_patience},
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }

    def save_checkpoint():
        checkpoint = {
            "format_version": 2, "model_state_dict": model.state_dict(),
            "model_config": {"current_soc_feature_index": model.current_soc_feature_index,
                             "max_soc_correction": model.max_soc_correction},
            "dataset_metadata": metadata, "scaler_mean": scaler["mean"],
            "scaler_scale": scaler["scale"], "best_epoch": best_epoch,
            "best_val_loss": best_val,
        }
        temporary_path = model_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(model_path)

    save_checkpoint()

    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        logger.info("Epoch %d/%d started", epoch + 1, epochs)
        train_loss = trainer.train_epoch(
            train_loader,
            logger=logger,
            epoch=epoch + 1,
            total_epochs=epochs,
        )
        val_loss = trainer.validate(val_loader)
        trainer.step_scheduler(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_started_at
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)
        history["epoch_seconds"].append(epoch_seconds)
        logger.info(
            "Epoch %d/%d complete | train_loss=%.9g | val_loss=%.9g | "
            "learning_rate=%.6g | elapsed=%.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            current_lr,
            epoch_seconds,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            save_checkpoint()
            logger.info(
                "New best checkpoint saved | path=%s | val_loss=%.9g",
                model_path,
                best_val,
            )
        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val
        save_training_history(history, history_path)
        if (epoch + 1) - best_epoch >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %d | best_val_loss=%.6f | best_epoch=%d",
                epoch + 1,
                best_val,
                best_epoch,
            )
            break

    logger.info(
        "Training finished | best_val_loss=%.6f | best_epoch=%d | total_elapsed=%.1fs",
        best_val,
        best_epoch,
        time.perf_counter() - training_started_at,
    )
    save_training_history(history, history_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)["model_state_dict"])
    model.eval()
    return model
