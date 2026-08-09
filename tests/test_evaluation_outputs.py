from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evaluation.evaluation_plots import (
    plot_prediction_diagnostics,
    plot_training_history,
)
from training.train_pipeline import save_training_history


class EvaluationOutputTests(unittest.TestCase):
    def test_history_and_diagnostic_plots_are_saved(self):
        history = {
            "epoch": [1, 2],
            "train_loss": [0.10, 0.05],
            "val_loss": [0.12, 0.08],
            "learning_rate": [0.001, 0.001],
            "epoch_seconds": [2.0, 2.1],
        }
        true_soc = np.array([[0.8], [0.7], [0.6]])
        pred_soc = np.array([[0.81], [0.68], [0.59]])

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_training_history(history, output_dir / "history.json")
            plot_training_history(history, output_dir / "training_history.png")
            plot_prediction_diagnostics(true_soc, pred_soc, output_dir)

            expected = [
                "history.json",
                "training_history.png",
                "soc_tracking.png",
                "prediction_scatter.png",
                "prediction_error_histogram.png",
                "prediction_residuals.png",
                "prediction_error_over_samples.png",
            ]
            for filename in expected:
                artifact = output_dir / filename
                self.assertTrue(artifact.exists())
                self.assertGreater(artifact.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
