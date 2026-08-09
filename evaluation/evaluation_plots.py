from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.metrics import mae, r2


def plot_soc(true_soc, pred_soc, output_path=None):

    figure = plt.figure(figsize=(10, 4))

    plt.plot(true_soc, label="True SOC")
    plt.plot(pred_soc, label="Predicted SOC")

    plt.legend()

    plt.title("SOC Tracking")

    if output_path is None:
        plt.show()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(figure)


def plot_training_history(history: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epochs, history["train_loss"], label="Train loss")
    axis.plot(epochs, history["val_loss"], label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_diagnostics(
    true_soc: np.ndarray,
    pred_soc: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_flat = np.asarray(true_soc).reshape(-1)
    pred_flat = np.asarray(pred_soc).reshape(-1)
    if true_flat.size == 0 or pred_flat.size == 0:
        raise ValueError("Cannot plot empty prediction arrays")
    if true_flat.size != pred_flat.size:
        raise ValueError("Prediction arrays must contain the same number of values")
    if not np.isfinite(true_flat).all() or not np.isfinite(pred_flat).all():
        raise ValueError("Prediction arrays must contain only finite values")

    errors = pred_flat - true_flat
    model_r2 = r2(true_flat, pred_flat)
    model_mae = mae(true_flat, pred_flat)

    track_count = min(2000, len(true_flat))
    plot_soc(
        true_flat[:track_count],
        pred_flat[:track_count],
        output_dir / "soc_tracking.png",
    )

    sample_count = min(50000, len(true_flat))
    sample_indices = np.linspace(0, len(true_flat) - 1, sample_count, dtype=int)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        true_flat[sample_indices],
        pred_flat[sample_indices],
        s=5,
        alpha=0.2,
    )
    axis.plot([0, 1], [0, 1], "r--", label="Perfect prediction")
    axis.set_xlabel("True SOC")
    axis.set_ylabel("Predicted SOC")
    axis.set_title(f"True versus predicted SOC (R² = {model_r2:.4f})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(
        output_dir / "prediction_scatter.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(errors, bins=60)
    axis.axvline(0.0, color="red", linestyle="--")
    axis.set_xlabel("Prediction error (predicted - true)")
    axis.set_ylabel("Count")
    axis.set_title("SOC prediction error distribution")
    axis.grid(alpha=0.25)
    figure.savefig(
        output_dir / "prediction_error_histogram.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(
        true_flat[sample_indices],
        errors[sample_indices],
        s=5,
        alpha=0.2,
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("True SOC")
    axis.set_ylabel("Prediction residual (predicted - true)")
    axis.set_title("Prediction residuals versus reference SOC")
    axis.grid(alpha=0.25)
    figure.savefig(
        output_dir / "prediction_residuals.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(sample_indices, errors[sample_indices], linewidth=0.8)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.axhline(model_mae, color="tab:orange", linestyle=":", label="±MAE")
    axis.axhline(-model_mae, color="tab:orange", linestyle=":")
    axis.set_xlabel("Test sample order")
    axis.set_ylabel("Prediction error (predicted - true)")
    axis.set_title("Prediction error over test samples")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(
        output_dir / "prediction_error_over_samples.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)
