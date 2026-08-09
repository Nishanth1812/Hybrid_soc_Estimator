import argparse
import json
import logging
from pathlib import Path

import torch

from validation.dataset_validator import DatasetValidator
from evaluation.evaluate_model import predict
from evaluation.evaluation_plots import (
    plot_prediction_diagnostics,
    plot_training_history,
)
from evaluation.metrics import calculate_metrics
from training.dataset_loader import load_dataloaders
from training.train_pipeline import train_model


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset and train the SOC model")
    parser.add_argument("--dataset-path", type=str, default="datasets/processed")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-subsample", type=int, default=1)
    parser.add_argument("--val-subsample", type=int, default=1)
    parser.add_argument("--test-subsample", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="models/best_model.pt")
    parser.add_argument(
        "--history-path",
        type=str,
        default="logs/training_history.json",
    )
    parser.add_argument("--log-path", type=str, default="logs/training.log")
    parser.add_argument(
        "--evaluation-output-dir",
        type=str,
        default="evaluation_outputs",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate the existing checkpoint",
    )
    return parser.parse_args()


def evaluate_trained_model(args: argparse.Namespace) -> None:
    logger = logging.getLogger("soc_training")
    output_dir = _project_path(args.evaluation_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _project_path(args.model_path)
    history_path = _project_path(args.history_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Starting evaluation using checkpoint %s", model_path)
    _, _, test_loader = load_dataloaders(
        _project_path(args.dataset_path),
        batch_size=args.batch_size,
        test_subsample=args.test_subsample,
    )
    targets, predictions = predict(model_path, test_loader, device)

    metrics = calculate_metrics(targets, predictions)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        plot_training_history(history, output_dir / "training_history.png")

    plot_prediction_diagnostics(targets, predictions, output_dir)

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Evaluation plots saved to %s", output_dir)

    print("Evaluation complete")
    print(f"Metrics: {metrics}")
    print(f"Metrics file: {metrics_path}")
    print(f"Plots directory: {output_dir}")


def main() -> None:
    args = parse_args()

    dataset_path = _project_path(args.dataset_path)
    evaluation_output_dir = _project_path(args.evaluation_output_dir)
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)
    validator = DatasetValidator(dataset_path)
    validator.run_full_validation(evaluation_output_dir / "dataset_distributions.png")

    if args.evaluate_only:
        print("Skipping training and evaluating the existing checkpoint...")
    else:
        print("Starting training...")

        train_model(
            dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_subsample=args.train_subsample,
            val_subsample=args.val_subsample,
            test_subsample=args.test_subsample,
            model_path=_project_path(args.model_path),
            history_path=_project_path(args.history_path),
            log_path=_project_path(args.log_path),
        )

        print("Training finished")
    evaluate_trained_model(args)


if __name__ == "__main__":
    main()
