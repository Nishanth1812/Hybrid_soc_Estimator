from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluation.evaluate_model import evaluate, predict
from models.lstm_soc_model import LSTMSOCEstimator


class EvaluationModelTests(unittest.TestCase):
    def test_predict_returns_targets_and_predictions_for_plain_checkpoint(self):
        model = LSTMSOCEstimator()
        inputs = torch.zeros(2, 100, 3)
        targets = torch.tensor([[0.8], [0.7]], dtype=torch.float32)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            actual_targets, predictions = predict(
                checkpoint,
                loader,
                torch.device("cpu"),
            )
            results = evaluate(checkpoint, loader, torch.device("cpu"))

        self.assertEqual(actual_targets.shape, (2, 1))
        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(np.isfinite(predictions).all())

        self.assertEqual(
            set(results),
            {"MAE", "RMSE", "R2", "MaxError", "MeanBias", "ErrorStd", "P95AbsError"},
        )


if __name__ == "__main__":
    unittest.main()
