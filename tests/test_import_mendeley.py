import unittest

import numpy as np
import pandas as pd

from soc_estimator.mendeley import build_splits


class MendeleyImportTests(unittest.TestCase):
    def test_forecast_uses_only_past_sensors_and_exact_future_time(self):
        times = np.array([0, 7, 19, 30, 44, 58, 73, 90, 108, 125, 146, 170])
        frame = pd.DataFrame([
            {"Cycle_Number": cycle, "Cycle_Type": "charge", "Time": time,
             "Voltage_measured": 3.2 + time / 1000, "Current_measured": 1.0,
             "Ambient_Temperature": 25.0, "soc": 0.2 + time / 1000}
            for cycle in range(1, 7) for time in times
        ])
        splits = build_splits(frame, sequence_length=3, stride=1)
        self.assertEqual(splits["features"], [
            "Voltage_measured", "Current_measured", "Ambient_Temperature_K",
            "Time_Step_s", "Current_SOC_Estimate"
        ])
        cycle_sets = []
        for name in ("train", "val", "test"):
            split = splits[name]
            metadata = split["metadata"]
            self.assertEqual(split["X"].shape[1:], (3, 5))
            np.testing.assert_allclose(
                split["y"], 0.2 + (metadata["end_time_s"] + 60) / 1000, atol=1e-7
            )
            np.testing.assert_allclose(metadata["target_time_s"] - metadata["end_time_s"], 60)
            self.assertTrue((metadata["target_left_source_index"] > metadata["source_end_index"]).all())
            self.assertTrue((metadata["target_time_s"] <= times[-1]).all())
            raw = splits["scaler"].inverse_transform(split["X"][0, :, :4])
            np.testing.assert_allclose(raw[:, 0], 3.2 + times[:3] / 1000, atol=1e-6)
            np.testing.assert_allclose(raw[:, 3], [0, 7, 12], atol=2e-6)
            self.assertTrue(np.all(split["X"][..., 4] == metadata["reference_soc_at_input_end"].to_numpy()[:, None]))
            cycle_sets.append(set(metadata["cycle_number"]))
        self.assertFalse(cycle_sets[0] & cycle_sets[1] or cycle_sets[0] & cycle_sets[2] or cycle_sets[1] & cycle_sets[2])
        changed_labels = frame.copy()
        changed_labels.loc[changed_labels["Time"] >= 58, "soc"] = 0.9
        changed = build_splits(changed_labels, sequence_length=3, stride=1)
        np.testing.assert_array_equal(changed["train"]["X"][0], splits["train"]["X"][0])
        self.assertNotEqual(changed["train"]["y"][0], splits["train"]["y"][0])

    def test_windows_and_targets_do_not_cross_measurement_gaps(self):
        times = [0, 20, 40, 60, 80, 100, 300, 320, 340, 360, 380, 400]
        frame = pd.DataFrame([
            {"Cycle_Number": cycle, "Cycle_Type": "discharge", "Time": time,
             "Voltage_measured": 4.0, "Current_measured": -1.0,
             "Ambient_Temperature": 25.0, "soc": 0.8 - time / 1000}
            for cycle in range(1, 4) for time in times
        ])
        splits = build_splits(frame, sequence_length=2, stride=1)
        for name in ("train", "val", "test"):
            metadata = splits[name]["metadata"]
            self.assertFalse(((metadata["end_time_s"] < 300) & (metadata["target_time_s"] > 100)).any())
            self.assertFalse(((metadata["start_time_s"] < 300) & (metadata["end_time_s"] >= 300)).any())


if __name__ == "__main__":
    unittest.main()
