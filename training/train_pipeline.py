from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm_soc_model import LSTMSOCEstimator
from training.dataset_loader import load_dataloaders
from training.trainer import Trainer
from utils.logging_utils import configure_logging


def _should_use_data_parallel() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def _state_dict_for_save(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


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
    epochs=60,
    batch_size=64,
    train_subsample=1,
    val_subsample=1,
    test_subsample=1,
    lr=1e-3,
    seed=42,
    early_stop_patience=12,
    model_path="models/best_model.pt",
    history_path="logs/training_history.json",
    log_path="logs/training.log",
):

    torch.manual_seed(seed)
    logger = configure_logging(Path(log_path), "soc_training")
    training_started_at = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 22))

    logger.info("Training started")
    logger.info("Dataset path: %s", dataset_path)
    logger.info(
        "Configuration | epochs=%d | batch_size=%d | learning_rate=%.6g | "
        "train_subsample=%d | val_subsample=%d | test_subsample=%d",
        epochs,
        batch_size,
        lr,
        train_subsample,
        val_subsample,
        test_subsample,
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
        train_subsample=train_subsample,
        val_subsample=val_subsample,
        test_subsample=test_subsample,
    )

    _log_loader_summary(logger, "Train", train_loader)
    _log_loader_summary(logger, "Validation", val_loader)
    _log_loader_summary(logger, "Test", test_loader)

    model = LSTMSOCEstimator().to(device)

    if _should_use_data_parallel():
        model = nn.DataParallel(model)
        logger.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
    else:
        logger.info("Using device: %s", device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn, device, lr_scheduler=scheduler)

    best_val = float("inf")
    best_epoch = 0
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_path)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "epoch_seconds": [],
    }

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
        save_training_history(history, history_path)
        logger.info(
            "Epoch %d/%d complete | train_loss=%.6f | val_loss=%.6f | "
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

            torch.save(_state_dict_for_save(model), model_path)
            logger.info(
                "New best checkpoint saved | path=%s | val_loss=%.6f",
                model_path,
                best_val,
            )

        elif (epoch + 1) - best_epoch >= early_stop_patience:

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

    return model
