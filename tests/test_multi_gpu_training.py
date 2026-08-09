import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from models.lstm_soc_model import LSTMSOCEstimator
from training.train_pipeline import (
    _should_use_data_parallel,
    _state_dict_for_save,
)


class MultiGPUTrainingTests(unittest.TestCase):
    @patch("training.train_pipeline.torch.cuda.device_count", return_value=2)
    @patch("training.train_pipeline.torch.cuda.is_available", return_value=True)
    def test_uses_data_parallel_when_two_cuda_devices_exist(self, _available, _count):
        self.assertTrue(_should_use_data_parallel())

    @patch("training.train_pipeline.torch.cuda.device_count", return_value=1)
    @patch("training.train_pipeline.torch.cuda.is_available", return_value=True)
    def test_does_not_use_data_parallel_with_one_cuda_device(self, _available, _count):
        self.assertFalse(_should_use_data_parallel())

    def test_saved_parallel_state_dict_loads_into_plain_model(self):
        model = torch.nn.DataParallel(LSTMSOCEstimator())
        state_dict = _state_dict_for_save(model)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(state_dict, checkpoint)
            restored = LSTMSOCEstimator()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))


if __name__ == "__main__":
    unittest.main()
