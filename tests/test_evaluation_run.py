import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from run_system import evaluate_trained_model


class EvaluationRunTests(unittest.TestCase):
    def test_evaluation_writes_metrics_and_plots_to_requested_directory(self):
        history = {
            "epoch": [1, 2],
            "train_loss": [0.1, 0.05],
            "val_loss": [0.12, 0.08],
            "learning_rate": [0.001, 0.001],
            "epoch_seconds": [1.0, 1.1],
        }
        targets = np.array([[0.8], [0.7], [0.6], [0.5]])
        predictions = np.array([[0.81], [0.68], [0.59], [0.52]])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "evaluation_outputs"
            history_path = root / "logs" / "training_history.json"
            history_path.parent.mkdir(parents=True)
            history_path.write_text(json.dumps(history), encoding="utf-8")
            args = argparse.Namespace(
                dataset_path="datasets/processed",
                batch_size=2,
                test_subsample=1,
                evaluation_output_dir=str(output_dir),
                model_path=str(root / "models" / "best_model.pt"),
                history_path=str(history_path),
            )

            with patch("run_system.load_dataloaders", return_value=(None, None, None)):
                with patch(
                    "run_system.predict",
                    return_value=(targets, predictions),
                ):
                    evaluate_trained_model(args)

            metrics = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertAlmostEqual(metrics["MAE"], 0.015)
            self.assertIn("R2", metrics)

            expected_files = [
                "metrics.json",
                "training_history.png",
                "soc_tracking.png",
                "prediction_scatter.png",
                "prediction_error_histogram.png",
                "prediction_residuals.png",
                "prediction_error_over_samples.png",
            ]
            for filename in expected_files:
                artifact = output_dir / filename
                self.assertTrue(artifact.exists(), filename)
                self.assertGreater(artifact.stat().st_size, 0, filename)


if __name__ == "__main__":
    unittest.main()
