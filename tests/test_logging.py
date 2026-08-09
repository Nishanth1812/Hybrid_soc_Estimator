from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import Trainer
from utils.logging_utils import configure_logging


class LoggingTests(unittest.TestCase):
    @staticmethod
    def _close_handlers(logger):
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    def test_logger_writes_timestamped_message_to_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            logger = configure_logging(path, "test_logging_file")
            logger.info("dataset stage complete")

            for handler in logger.handlers:
                handler.flush()

            contents = path.read_text(encoding="utf-8")
            self.assertIn("INFO", contents)
            self.assertIn("dataset stage complete", contents)
            self._close_handlers(logger)

    def test_train_epoch_emits_progress_when_logger_is_supplied(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, torch.nn.MSELoss(), torch.device("cpu"))
        inputs = torch.ones(4, 1)
        targets = torch.ones(4, 1)
        dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

        with TemporaryDirectory() as directory:
            logger = configure_logging(Path(directory) / "train.log", "test_logging_train")
            loss = trainer.train_epoch(
                dataloader,
                logger=logger,
                epoch=1,
                total_epochs=1,
            )

            for handler in logger.handlers:
                handler.flush()

            contents = (Path(directory) / "train.log").read_text(encoding="utf-8")
            self._close_handlers(logger)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn("batch", contents)


if __name__ == "__main__":
    unittest.main()
