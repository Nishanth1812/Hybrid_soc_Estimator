from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from soc_estimator.evaluation import evaluate, predict
from soc_estimator.model import LSTMSOCEstimator


class EvaluationModelTests(unittest.TestCase):
    def test_predict_returns_targets_and_predictions_for_plain_checkpoint(self):
        model = LSTMSOCEstimator(input_features=3)
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

    def test_forecast_checkpoint_reproduces_preprocessing_and_rejects_gaps(self):
        import pandas as pd
        from soc_estimator.evaluation import forecast
        from soc_estimator.mendeley import build_splits

        torch.manual_seed(42)
        model = LSTMSOCEstimator(input_features=5, current_soc_feature_index=4).eval()
        metadata = {
            "task": "future_soc_forecast", "sequence_length": 3,
            "forecast_horizon_seconds": 60.0, "max_gap_seconds": 60.0,
            "features": ["Voltage_measured", "Current_measured", "Ambient_Temperature_K", "Time_Step_s", "Current_SOC_Estimate"],
            "current_soc_feature_index": 4,
        }
        model_config = {"current_soc_feature_index": 4, "max_soc_correction": 0.03}
        history = np.array([[100, 3.3, -1, 25], [110, 3.2, -1, 25], [125, 3.1, -1, 25]])
        rows = []
        for cycle in (1, 2, 3):
            for time, voltage, current, temperature in np.vstack((history, [[145, 3.0, -1, 25], [165, 2.9, -1, 25], [185, 2.8, -1, 25]])):
                rows.append({"Cycle_Number": cycle, "Cycle_Type": "discharge", "Time": time,
                             "Voltage_measured": voltage, "Current_measured": current,
                             "Ambient_Temperature": temperature, "soc": 0.8 - time / 1000})
        splits = build_splits(pd.DataFrame(rows), sequence_length=3, stride=1)
        mean, scale = splits["scaler"].mean_, splits["scaler"].scale_
        current_soc = 0.49
        raw = np.column_stack((history[:, 1], history[:, 2], history[:, 3] + 273.15, [0, 10, 15]))
        scaled = (raw.astype(np.float32) - mean) / scale
        inputs = torch.tensor(np.column_stack((scaled, np.full(3, current_soc))), dtype=torch.float32)[None]
        loader = DataLoader(TensorDataset(inputs, torch.zeros(1, 1)), batch_size=1)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"format_version": 2, "model_state_dict": model.state_dict(),
                        "model_config": model_config,
                        "dataset_metadata": metadata, "scaler_mean": mean.tolist(),
                        "scaler_scale": scale.tolist()}, checkpoint)
            result = forecast(checkpoint, history, current_soc)
            _, predictions = predict(checkpoint, loader, torch.device("cpu"))
            self.assertAlmostEqual(result["soc_fraction"], predictions[0, 0], places=6)
            self.assertEqual(result["target_time_s"], 185.0)
            with self.assertRaisesRegex(ValueError, "Current SOC"):
                forecast(checkpoint, history, 1.2)
            history[-1, 0] = 300
            with self.assertRaisesRegex(ValueError, "gap"):
                forecast(checkpoint, history, current_soc)


if __name__ == "__main__":
    unittest.main()
