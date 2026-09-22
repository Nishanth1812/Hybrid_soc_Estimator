from __future__ import annotations

import json
from pathlib import Path

import modal


REMOTE_ROOT = Path("/data")
DATASET_PATH = REMOTE_ROOT / "processed"
OUTPUT_PATH = REMOTE_ROOT / "outputs"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.13.0",
        "joblib>=1.5.0",
        "matplotlib>=3.8.0",
    )
    .add_local_python_source(
        "config",
        "evaluation",
        "models",
        "training",
        "utils",
    )
)

volume = modal.Volume.from_name("mahindra-mendeley-bms", create_if_missing=True)
app = modal.App("mahindra-mendeley-bms-training", image=image)


@app.function(
    gpu="T4",
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
)
def train_remote() -> dict:
    import torch

    from evaluation.evaluate_model import predict
    from evaluation.metrics import calculate_metrics
    from training.dataset_loader import load_dataloaders
    from training.train_pipeline import train_model

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_PATH / "best_model.pt"
    history_path = OUTPUT_PATH / "training_history.json"
    log_path = OUTPUT_PATH / "training.log"

    train_model(
        DATASET_PATH,
        epochs=60,
        batch_size=1024,
        num_workers=2,
        model_path=model_path,
        history_path=history_path,
        log_path=log_path,
    )

    _, _, test_loader = load_dataloaders(
        DATASET_PATH,
        batch_size=1024,
        num_workers=2,
    )
    targets, predictions = predict(model_path, test_loader, torch.device("cuda"))
    metrics = calculate_metrics(targets, predictions)
    (OUTPUT_PATH / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    volume.commit()
    print(json.dumps({"device": torch.cuda.get_device_name(0), **metrics}, indent=2))
    return {"device": torch.cuda.get_device_name(0), **metrics}


@app.local_entrypoint()
def main() -> None:
    print(train_remote.remote())
