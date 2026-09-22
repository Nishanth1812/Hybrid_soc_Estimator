import unittest

import numpy as np
import pandas as pd

from data_pipeline.import_mendeley import build_splits


class MendeleyImportTests(unittest.TestCase):
    def test_build_splits_normalizes_units_and_keeps_cycles_separate(self):
        rows = []
        for cycle in range(1, 7):
            for cycle_type in ("charge", "discharge"):
                for step in range(4):
                    rows.append(
                        {
                            "Cycle_Number": cycle,
                            "Cycle_Type": cycle_type,
                            "Voltage_measured": 4.0 + step / 100,
                            "Current_measured": 1.0,
                            "Ambient_Temperature": 25.0,
                            "soc": 0.2 + step / 10,
                        }
                    )

        splits = build_splits(pd.DataFrame(rows), sequence_length=3)

        self.assertEqual({key for key in splits if key != "scaler"}, {"train", "val", "test"})
        self.assertEqual(splits["train"]["X"].shape[1:], (3, 3))
        self.assertEqual(splits["train"]["X"].dtype, np.float32)
        self.assertAlmostEqual(splits["scaler"].mean_[2], 298.15, places=4)
        self.assertAlmostEqual(splits["train"]["X"][0, 0, 2], 0.0, places=5)
        self.assertTrue(np.all(np.isclose(splits["train"]["y"], 0.4) | np.isclose(splits["train"]["y"], 0.5)))
        self.assertTrue(np.all((splits["train"]["y"] >= 0) & (splits["train"]["y"] <= 1)))


if __name__ == "__main__":
    unittest.main()
