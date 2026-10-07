# Pipeline Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add timestamped console and file logs for PyBaMM dataset generation and LSTM training without changing Kaggle commands.

**Architecture:** Add `utils/logging_utils.py` with a shared logger factory. The dataset generator will report completed futures as they finish and the build pipeline will log each preprocessing/output stage. The trainer will accept an optional logger and emit periodic batch progress; `train_pipeline.py` will log epoch summaries, device information, checkpoints, and early stopping.

**Tech Stack:** Python standard `logging`, `concurrent.futures.as_completed`, PyBaMM, NumPy, PyTorch, `unittest`.

## Global Constraints

- Keep existing Kaggle commands unchanged.
- Keep the current simulation, preprocessing, model, and optimizer behavior unchanged.
- Write logs under `logs/` relative to the project working directory.
- Preserve fail-fast behavior when a simulation fails after its retry budget.
- Preserve the existing plain-model checkpoint format.

---

### Task 1: Add shared logging and logging tests

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/logging_utils.py`
- Create: `tests/test_logging.py`

**Interfaces:**
- Produces: `configure_logging(log_path, logger_name) -> logging.Logger`.

- [ ] **Step 1: Write the failing logger test**

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from utils.logging_utils import configure_logging


class LoggingTests(unittest.TestCase):
    def test_logger_writes_timestamped_message_to_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            logger = configure_logging(path, "test_logging")
            logger.info("dataset stage complete")

            for handler in logger.handlers:
                handler.flush()

            contents = path.read_text(encoding="utf-8")
            self.assertIn("INFO", contents)
            self.assertIn("dataset stage complete", contents)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm it fails because the helper is missing**

```text
python -m unittest tests.test_logging -v
```

- [ ] **Step 3: Implement the shared logger factory**

Create a logger with a timestamp formatter, a stdout handler, and a UTF-8 file handler. Remove/close existing handlers for the same logger name so repeated notebook runs do not duplicate every message.

- [ ] **Step 4: Run the logger test and confirm it passes**

```text
python -m unittest tests.test_logging -v
```

### Task 2: Add dataset-generation progress logs

**Files:**
- Modify: `data_pipeline/generation/dataset_generator.py`
- Modify: `data_pipeline/build_dataset.py`

**Interfaces:**
- Consumes: existing simulation task tuples and `SimulationRecord` metadata.
- Produces: completion-order progress logs and dataset-stage summaries; no new CLI arguments.

- [ ] **Step 1: Add a testable simulation timing/attempt metadata assertion**

Extend the existing logging test or add a focused test using a small synthetic record to verify the progress formatter includes simulation ID and completed/total counts.

- [ ] **Step 2: Run the focused test and confirm it fails**

```text
python -m unittest tests.test_logging -v
```

- [ ] **Step 3: Implement generation logging**

Record attempt count and elapsed seconds in each simulation's metadata. Replace ordered `executor.map` collection with `submit`/`as_completed`, preserve output order by task index, and log `[completed/total]` after every finished simulation. Log exceptions with `logger.exception` before re-raising.

- [ ] **Step 4: Implement build-stage logging**

Configure `logs/dataset_generation.log` in `build_dataset.main`, then log configuration, filtering, split sizes, scaling, sequence shapes, output paths, and total elapsed time. Keep the existing final shape prints for compatibility.

- [ ] **Step 5: Run focused tests and static checks**

```text
python -m unittest tests.test_logging tests.test_multi_gpu_training -v
git diff --check -- data_pipeline/generation/dataset_generator.py data_pipeline/build_dataset.py utils tests
```

### Task 3: Add training progress logs

**Files:**
- Modify: `training/trainer.py`
- Modify: `training/train_pipeline.py`

**Interfaces:**
- Consumes: existing dataloaders, model, optimizer, and scheduler.
- Produces: optional logger-aware `Trainer.train_epoch(..., logger=None, epoch=None, total_epochs=None)` with unchanged loss return value.

- [ ] **Step 1: Add a test for logger-aware epoch execution**

Use a tiny real `TensorDataset` and a test logger to verify `Trainer.train_epoch` returns a finite loss and emits at least one progress message when a logger is supplied.

- [ ] **Step 2: Run the focused trainer test and confirm it fails**

```text
python -m unittest tests.test_logging -v
```

- [ ] **Step 3: Implement periodic batch logs**

Add optional logging parameters to `train_epoch`. Log approximately ten progress updates per epoch, including epoch number, batch number, total batches, and current loss. Do not change optimizer, gradient clipping, or returned average loss.

- [ ] **Step 4: Implement pipeline logs**

Configure `logs/training.log`, log device/GPU count and loader sizes, pass the logger into `train_epoch`, and log each epoch's train loss, validation loss, learning rate, elapsed time, best-checkpoint updates, early stopping, and total duration. Preserve automatic `DataParallel` and unwrapped checkpoint saving.

- [ ] **Step 5: Run all available tests and static checks**

```text
python -m unittest discover -s tests -v
git diff --check -- training utils tests
```

### Task 4: Document and rebuild the uploadable ZIP

**Files:**
- Modify: `README.md`
- Rebuild: `Hybrid_soc_Estimator_code.zip`

**Interfaces:**
- Produces: documentation of generated log paths and the unchanged Kaggle command.

- [ ] **Step 1: Document log locations and example messages**

Explain that running the existing generation and training commands automatically creates the two log files and prints the same messages to the Kaggle output.

- [ ] **Step 2: Rebuild the ZIP**

Include source code, tests, utilities, configuration, and README; exclude generated datasets, `.venv`, caches, and model checkpoints.

- [ ] **Step 3: Inspect the archive**

Confirm `utils/logging_utils.py`, `data_pipeline/generation/dataset_generator.py`, `training/trainer.py`, and `tests/test_logging.py` are present and excluded artifacts are absent.
