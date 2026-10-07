from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import modal


REMOTE_ROOT = Path("/data")
DATASET_PATH = REMOTE_ROOT / "forecast_60s_with_current_soc_v1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch>=2.2.0", "numpy>=1.26.0", "joblib>=1.5.0", "scikit-learn>=1.5.0")
    .add_local_python_source("soc_estimator")
)
volume = modal.Volume.from_name("mahindra-mendeley-bms", create_if_missing=True)
app = modal.App("mahindra-mendeley-soc-forecast", image=image)


@app.function(gpu="A10G", volumes={str(REMOTE_ROOT): volume}, timeout=24 * 60 * 60)
def train_remote(epochs: int = 300) -> dict:
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge

    from soc_estimator.data import load_dataloaders
    from soc_estimator.evaluation import calculate_grouped_metrics, calculate_metrics, predict
    from soc_estimator.training import train_model

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = REMOTE_ROOT / "outputs_forecast_60s_with_current_soc" / run_id
    output_path.mkdir(parents=True, exist_ok=False)
    print(f"Run artifacts: {output_path}", flush=True)
    try:
        model_path = output_path / "best_model.pt"
        train_model(
            DATASET_PATH, epochs=epochs, batch_size=512, num_workers=2,
            lr=3e-4, early_stop_patience=40, model_path=model_path,
            history_path=output_path / "training_history.json", log_path=output_path / "training.log",
        )
        train_loader, val_loader, test_loader = load_dataloaders(
            DATASET_PATH, batch_size=1024, num_workers=2,
        )
        # All features, including current SOC, are available at forecast time.
        baseline = Ridge(alpha=1.0)
        sampler_weights = getattr(train_loader.sampler, "weights", None)
        weights = sampler_weights.numpy() if sampler_weights is not None else None
        if weights is not None:
            weights = weights * len(weights) / weights.sum()
        baseline.fit(train_loader.dataset.X[:, -1, :].numpy(),
                     train_loader.dataset.y[:, 0].numpy(), sample_weight=weights)

        reports = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            targets, predictions = predict(model_path, loader, torch.device("cuda"))
            with (DATASET_PATH / f"window_metadata_{split}.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            linear_predictions = np.clip(baseline.predict(loader.dataset.X[:, -1, :].numpy()), 0.0, 1.0)
            oracle_persistence = np.array([float(row["reference_soc_at_end"]) for row in rows])
            persistence_metrics = calculate_grouped_metrics(targets, oracle_persistence, rows)
            reports[split] = {
                "lstm": calculate_grouped_metrics(targets, predictions, rows),
                "linear_current_soc_baseline": calculate_grouped_metrics(targets, linear_predictions, rows),
                "current_soc_persistence_baseline": persistence_metrics,
            }
            with (output_path / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]) + ["target_soc", "predicted_soc", "linear_baseline_soc", "current_soc_persistence_soc"])
                writer.writeheader()
                for row, target, prediction, linear in zip(rows, targets.reshape(-1), predictions.reshape(-1), linear_predictions):
                    writer.writerow({**row, "target_soc": float(target), "predicted_soc": float(prediction),
                                     "linear_baseline_soc": float(linear),
                                     "current_soc_persistence_soc": float(row["reference_soc_at_end"])})
        history = json.loads((output_path / "training_history.json").read_text(encoding="utf-8"))
        result = {
            "device": torch.cuda.get_device_name(0), "output_path": str(output_path),
            "task": "future_soc_forecast", "forecast_horizon_seconds": history["forecast_horizon_seconds"],
            "best_epoch": history["best_epoch"], "epochs_completed": len(history["epoch"]),
            "beats_current_soc_persistence_on_validation_macro_mae":
                reports["val"]["lstm"]["cycle_summary"]["macro_MAE"] <
                reports["val"]["current_soc_persistence_baseline"]["cycle_summary"]["macro_MAE"],
            "metrics_unit": "SOC fraction; multiply errors by 100 for percentage points",
            "persistence_baseline_note": "Predicts future SOC by holding the supplied current SOC estimate constant",
            "reports": reports,
        }
        (output_path / "forecast_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
        print(json.dumps(reports["test"]["lstm"]["overall"], indent=2))
        return result
    finally:
        volume.commit()


@app.local_entrypoint()
def main(epochs: int = 300) -> None:
    result = train_remote.remote(epochs)
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
