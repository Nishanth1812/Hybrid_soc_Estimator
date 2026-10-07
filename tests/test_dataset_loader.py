from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from soc_estimator.data import SOCDataset, load_dataloaders


class DatasetLoaderTests(unittest.TestCase):
    def test_float32_dataset_tensor_shares_numpy_storage(self):
        X = np.zeros((2, 100, 3), dtype=np.float32)
        y = np.array([0.8, 0.7], dtype=np.float32)
        dataset = SOCDataset(X, y)

        X[0, 0, 0] = 3.5

        self.assertEqual(dataset.X[0, 0, 0].item(), 3.5)

    def test_loader_accepts_worker_and_pin_memory_options(self):
        with TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            X = np.zeros((4, 100, 3), dtype=np.float32)
            y = np.full((4,), 0.75, dtype=np.float32)
            for split in ("train", "val", "test"):
                np.save(dataset_dir / f"X_{split}.npy", X)
                np.save(dataset_dir / f"y_{split}.npy", y)

            loaders = load_dataloaders(
                dataset_dir,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
            )

            self.assertEqual([len(loader.dataset) for loader in loaders], [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
