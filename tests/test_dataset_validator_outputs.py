from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from validation.dataset_validator import DatasetValidator


class DatasetValidatorOutputTests(unittest.TestCase):
    def test_validation_distribution_plot_is_saved(self):
        with TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "processed"
            dataset_dir.mkdir()
            X = np.zeros((4, 100, 3), dtype=np.float32)
            y = np.full((4,), 0.75, dtype=np.float32)
            for split in ("train", "val", "test"):
                np.save(dataset_dir / f"X_{split}.npy", X)
                np.save(dataset_dir / f"y_{split}.npy", y)

            output_path = Path(directory) / "evaluation_outputs" / "dataset_distributions.png"
            DatasetValidator(str(dataset_dir)).run_full_validation(output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
