import unittest

import numpy as np

from evaluation.metrics import calculate_metrics, r2


class MetricsTests(unittest.TestCase):
    def test_calculate_metrics_returns_expected_values(self):
        y_true = np.array([[1.0], [2.0], [4.0]])
        y_pred = np.array([[2.0], [2.0], [3.0]])

        actual = calculate_metrics(y_true, y_pred)

        self.assertAlmostEqual(actual["MAE"], 2.0 / 3.0)
        self.assertAlmostEqual(actual["RMSE"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(actual["R2"], 4.0 / 7.0)
        self.assertAlmostEqual(actual["MaxError"], 1.0)
        self.assertAlmostEqual(actual["MeanBias"], 0.0)
        self.assertAlmostEqual(actual["ErrorStd"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(actual["P95AbsError"], 1.0)

    def test_r2_returns_zero_for_constant_reference(self):
        self.assertEqual(r2([0.5, 0.5], [0.5, 0.6]), 0.0)

    def test_metrics_reject_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0.1, 0.2], [0.1])


if __name__ == "__main__":
    unittest.main()
